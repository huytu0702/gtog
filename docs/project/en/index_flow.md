# Indexing Flow

This document explains the **GraphRAG indexing pipeline** — how raw documents become a queryable knowledge graph stored in **Azure Cosmos DB**. It covers both the backend job lifecycle (queue → worker → status updates) and the GraphRAG core workflow steps.

> This deployment runs in **cosmos_pipeline mode**: `settings.yaml` has `output.type: cosmosdb`, so pipeline datasets are written directly to Cosmos inside per-version `pipeline-{collection}-{version}` containers. The backend also overrides the GraphRAG Cosmos vector store at runtime so all vector embeddings are written into the shared `vectors` container and scoped by partition key. **No parquet files are written to disk.**

## 1. Two-Layer View

The indexing pipeline runs in two layers:

| Layer | Component | Responsibility |
|---|---|---|
| **Job orchestration** | `backend/app/services/indexing_service.py` + `backend/app/worker.py` | Job queueing, leasing, retries, status reporting |
| **Pipeline execution** | `graphrag.api.build_index` (calls `graphrag.index.run.run_pipeline`) | Workflow steps that produce pipeline datasets, written to Cosmos via `CosmosDBPipelineStorage` |

```mermaid
graph LR
    A[POST /collections/:id/index] --> B[IndexingService.start_indexing]
    B --> C[Cosmos: create job<br/>status=pending]
    B --> D[Queue: enqueue message]
    D --> E[worker.py poll loop]
    E --> F[Lease job<br/>status=running]
    F --> G[graphrag.api.build_index]
    G --> H[Workflow steps<br/>1..N]
    H --> I[Pipeline datasets in<br/>pipeline-{collection}-{version}]
    H --> J1[Vector embeddings in<br/>shared vectors container]
    I --> V[Verify pipeline datasets +<br/>verify vector scopes +<br/>upsert artifactManifest]
    J1 --> V
    V --> P[set_active_version<br/>in collections container]
    P --> J[Cosmos: status=completed]
    F -. failure .-> K[Retry up to<br/>max_attempts]
    K --> J2[status=failed]
```

## 2. Job Lifecycle (Backend)

```mermaid
stateDiagram-v2
    [*] --> pending: enqueue
    pending --> running: worker leases job
    running --> completed: pipeline succeeds
    running --> retrying: transient failure
    retrying --> running: re-leased after backoff
    running --> failed: max_attempts exceeded
    running --> cancelled: manual cancel
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

**Job document fields** (Cosmos `indexingJobs` container):
- `id`, `collectionId` (partition key), `status`, `attempt`, `maxAttempts`
- `leaseOwnerId`, `leaseExpiresAt`, `heartbeatAt`
- `startedAt`, `completedAt`, `retryAt`, `progress`, `message`, `error`

**Lease semantics** (configurable in `Settings`):
- `indexing_worker_lease_duration_seconds` = 300
- `indexing_worker_heartbeat_interval_seconds` = 30
- `indexing_worker_recovery_interval_seconds` = 30
- `indexing_job_max_attempts` = 3

## 3. Worker Loop

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker (worker.py)
    participant Q as Storage Queue
    participant C as Cosmos (jobs)
    participant Core as graphrag.api.build_index
    participant S as Storage (blob/cosmos)

    loop forever
        W->>Q: receive_messages(batch=4)
        alt no messages
            W->>W: sleep poll_interval
        else messages received
            W->>C: lease job (CAS on leaseOwnerId)
            alt lease acquired
                W->>C: status=running, attempt++
                W->>S: write input docs to pipeline storage
                W->>Core: build_index(config, callbacks)
                loop per workflow
                    Core-->>W: PipelineRunResult (workflow done)
                    W->>C: update progress, message
                end
                alt success
                    W->>C: status=completed, completedAt=now
                    W->>Q: delete_message
                else failure
                    alt attempt < max
                        W->>C: status=retrying, retryAt=now+backoff
                        W->>Q: delete_message (will re-enqueue with delay)
                    else attempt >= max
                        W->>C: status=failed, error=...
                        W->>Q: delete_message
                    end
                end
            else lease lost
                W->>Q: skip (visibility timeout will requeue)
            end
        end
    end
```

## 4. GraphRAG Core Pipeline

`graphrag.api.build_index` calls `graphrag.index.run.run_pipeline.run_pipeline` which executes a sequence of workflows.

### Standard pipeline (default)

```mermaid
flowchart TB
    Start([Start build_index]) --> Load[load_input_documents<br/>read from input storage]
    Load --> Chunk[create_base_text_units<br/>split into chunks]
    Chunk --> FinalDocs[create_final_documents<br/>finalize doc metadata]
    FinalDocs --> ExtractGraph[extract_graph<br/>LLM: entities + relationships]
    ExtractGraph --> Finalize[finalize_graph<br/>compute degree, centrality]
    Finalize --> Covariates[extract_covariates<br/>LLM: claims]
    Covariates --> Communities[create_communities<br/>Leiden clustering]
    Communities --> FinalTU[create_final_text_units<br/>link entities/rels to text units]
    FinalTU --> Reports[create_community_reports<br/>LLM: summarize each community]
    Reports --> Embed[generate_text_embeddings<br/>embed entities, relationships, reports]
    Embed --> End([Done — write datasets to Cosmos])

    classDef llm fill:#fde68a,stroke:#000,stroke-width:2px
    class ExtractGraph,Covariates,Reports,Embed llm
```

LLM-powered steps are highlighted (yellow). NLP/graph-only steps are deterministic and CPU-bound.

### Fast pipeline (`IndexingMethod.Fast`)

```mermaid
flowchart TB
    Start([Start]) --> Load[load_input_documents]
    Load --> Chunk[create_base_text_units]
    Chunk --> FinalDocs[create_final_documents]
    FinalDocs --> NLP[extract_graph_nlp<br/>NLP only — no LLM]
    NLP --> Prune[prune_graph<br/>reduce graph size]
    Prune --> Covariates[extract_covariates]
    Covariates --> Communities[create_communities]
    Communities --> FinalTU[create_final_text_units]
    FinalTU --> ReportsText[create_community_reports_text<br/>text-based — no LLM]
    ReportsText --> Embed[generate_text_embeddings]
    Embed --> End([Done])
```

Fast mode trades quality for speed by skipping LLM-based extraction and report generation.

### Update pipeline

When `is_update_run=True`, additional `*_update` workflows run after the base pipeline:
- `update_entities_relationships`
- `update_communities`
- `update_community_reports`
- `update_covariates`
- `update_text_units`
- `update_text_embeddings`
- `update_clean_state`

Update workflows merge new artifacts with the previous index using timestamped storage.

## 5. Workflow Step Detail

### 5.1 `load_input_documents`

- Reads files from input storage matching `config.input.file_pattern` (e.g., `.*\.txt$`).
- Produces a `pd.DataFrame` of documents (id, title, text).

### 5.2 `create_base_text_units`

- Splits documents into chunks per `config.chunks.size` and `config.chunks.overlap`.
- Strategy: `tokens` (default) or `sentence`.
- Output schema: `id, document_ids, text, n_tokens`.

### 5.3 `extract_graph` (LLM)

- Uses `graphrag.index.operations.extract_graph` with the **entity extraction prompt**.
- For each text unit, the LLM emits a structured list of `(entity_name, entity_type, description)` and `(source, target, description, weight)`.
- Multiple `gleanings` rounds (default 1) re-prompt the LLM to find missed entities.
- Results are merged: duplicate entities are summarized via the **summarize descriptions prompt**.

### 5.4 `finalize_graph`

- Builds a NetworkX graph.
- Computes `degree` and ranks nodes/edges.
- Adds `human_readable_id` to each entity/relationship.

### 5.5 `extract_covariates` (LLM, optional)

- Extracts claims (subject, object, type, status, period, source text) per text unit.
- Disabled by default unless `config.extract_claims.enabled=true`.

### 5.6 `create_communities`

- Hierarchical Leiden clustering over the entity graph.
- Produces communities at multiple levels (level 0 = top-level clusters; level 1+ = sub-communities).
- Each community holds: `entity_ids`, `relationship_ids`, `text_unit_ids`, `parent`, `children`, `size`.

### 5.7 `create_final_text_units`

- Back-links text units to the entities and relationships they contain.
- Adds `entity_ids`, `relationship_ids`, `covariate_ids` columns to text units.

### 5.8 `create_community_reports` (LLM)

- For each community, the LLM produces a structured report (`title`, `summary`, `findings`, `rating`, `rating_explanation`).
- Reports are stored as `full_content` (Markdown) plus a structured `attributes` dict.

### 5.9 `generate_text_embeddings`

- Embeds (depending on `config.embed_text.target`):
  - `entity.description` and/or `entity.title`
  - `relationship.description`
  - `community_report.full_content`
  - `text_unit.text`
- Default backend: `openai_embedding` (`text-embedding-3-small`) via LiteLLM.
- Embeddings are required for Local, DRIFT, Basic, and ToG search.

## 6. Output Artifacts

After indexing completes, the following datasets exist as parquet payloads inside the per-version Cosmos container `pipeline-{collection}-{version}` (one Cosmos document per dataset, body = parquet bytes):

```mermaid
erDiagram
    DOCUMENTS ||--o{ TEXT_UNITS : "splits into"
    TEXT_UNITS }o--o{ ENTITIES : "mentions"
    TEXT_UNITS }o--o{ RELATIONSHIPS : "evidences"
    TEXT_UNITS }o--o{ COVARIATES : "evidences"
    ENTITIES ||--o{ RELATIONSHIPS : "source/target"
    ENTITIES }o--o{ COMMUNITIES : "member of"
    COMMUNITIES ||--|| COMMUNITY_REPORTS : "summarized by"
    COMMUNITIES ||--o{ COMMUNITIES : "parent/child"

    DOCUMENTS {
        string id PK
        string title
        string text
        list text_unit_ids
    }
    TEXT_UNITS {
        string id PK
        string text
        int n_tokens
        list entity_ids
        list relationship_ids
        list covariate_ids
        list document_ids
    }
    ENTITIES {
        string id PK
        string title
        string type
        string description
        list description_embedding
        int community
        int degree
        list text_unit_ids
    }
    RELATIONSHIPS {
        string id PK
        string source FK
        string target FK
        string description
        float weight
        list text_unit_ids
    }
    COMMUNITIES {
        string id PK
        int level
        string parent
        list children
        list entity_ids
        list relationship_ids
        list text_unit_ids
    }
    COMMUNITY_REPORTS {
        string id PK
        string community FK
        string summary
        string full_content
        list full_content_embedding
        float rank
    }
    COVARIATES {
        string id PK
        string type
        string subject_id FK
        string object_id
        string text_unit_id FK
    }
```

See [database_schema.md](database_schema.md) for the full column schema of each dataset.

## 7. Output Storage Layout

In **cosmos_pipeline mode** (this deployment), pipeline datasets for one indexing run live inside a versioned pipeline container, while vector embeddings for all versions share one physical Cosmos vector container:

| Container | Partition Key | Purpose |
|---|---|---|
| `pipeline-{collection}-{version}` | implementation-defined by GraphRAG cosmos pipeline storage | GraphRAG pipeline datasets for one collection version |
| `artifactManifest` | `/collectionId` | Verified dataset counts per `(collection, version)` |
| `collections` | `/collectionId` | Active version pointer per collection (`activeVersion`) |
| `vectors` | `/partitionKey` | Shared Cosmos vector container; logical scopes are `{collectionId}:{version}|{embeddingKind}` |

The backend worker verifies both the pipeline datasets and the expected vector scopes before calling `set_active_version(...)`. The query layer reads `collections.activeVersion` for the collection, then loads from the matching `pipeline-{collection}-{version}` container and the matching scoped partition inside `vectors`.

> File-backed development is not supported in this deployment — `settings.yaml` pins `output.type: cosmosdb`. The GraphRAG core supports `file` and `blob` modes, but they are not used here.

## 8. Progress Reporting

The worker injects `WorkflowCallbacks` into `build_index` to report progress.

```mermaid
sequenceDiagram
    participant Core as build_index
    participant CB as WorkflowCallbacks
    participant W as worker.py
    participant C as Cosmos

    loop per workflow
        Core->>CB: workflow_start(name)
        CB->>W: log + heartbeat
        W->>C: heartbeatAt = now
        Core-->>W: yield PipelineRunResult
        W->>C: progress = (i / total) * 100<br/>message = "completed: <name>"
    end
```

The frontend polls `GET /api/collections/:id/index` every 2s while status is `running` or `pending`, displaying `progress` and `message` to the user.

## 9. Failure Handling & Retries

```mermaid
flowchart TB
    Start[Workflow exception] --> Check{attempt < max_attempts?}
    Check -- yes --> Retry[status=retrying<br/>retryAt = now + backoff]
    Retry --> Sleep[Worker re-enqueues<br/>with visibility delay]
    Sleep --> Lease[Re-lease job]
    Lease --> Run[Re-run pipeline]
    Run --> CheckSuccess{success?}
    CheckSuccess -- yes --> Done[status=completed]
    CheckSuccess -- no --> Check
    Check -- no --> Failed[status=failed<br/>error=last exception]
```

**Backoff:** exponential with jitter, capped at `AZURE_STORAGE_QUEUE_VISIBILITY_TIMEOUT_SECONDS` (300s default).

**Idempotency:** each indexing run targets a fresh `pipeline-{collection}-{version}` container (version = first 16 chars of `jobId`) and a fresh logical vector scope inside `vectors` for each embedding kind. Only after `_verify_pipeline_output(...)` and `_verify_vector_output(...)` succeed does `set_active_version(...)` swap the pointer in the `collections` container (`activeVersion`), so a retry from scratch writes a clean version without affecting the currently-served one.

## 10. Cache & Update Strategy

- **Cache** (`config.cache`): LLM responses cached by hash to avoid re-paying for the same prompts on retry.
- **Update runs** (`is_update_run=True`): pipeline writes a new version container, then `set_active_version` swaps the pointer atomically once verification passes. Used by `graphrag update` CLI.
- **Cache warming**: when `serving_cache_warm_on_index_complete=true`, the dataset cache is pre-loaded after a successful indexing run so the first query hits a warm cache.

## Related Docs

- [architecture.md](architecture.md) — System overview
- [database_schema.md](database_schema.md) — Parquet column schemas
- [query_flow.md](query_flow.md) — How pipeline datasets are consumed at query time
