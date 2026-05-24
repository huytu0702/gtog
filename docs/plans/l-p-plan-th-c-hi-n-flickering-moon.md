# Plan: Consolidate Cosmos vector storage without modifying `graphrag/`

## Scope constraint

Allowed to change:

- `backend/`
- `scripts/`
- `tests/unit/scripts/`
- `docs/`

Do **not** change:

- `graphrag/`

This means the implementation must not edit `graphrag/vector_stores/cosmosdb.py`, `graphrag/index/operations/embed_text/embed_text.py`, or `graphrag/storage/cosmosdb_pipeline_storage.py` directly. Any behavior change for GraphRAG internals must be injected from backend startup/runtime configuration, or deferred until modifying `graphrag/` is allowed.

## Context

Current production storage uses Azure Cosmos DB in `cosmos_pipeline` mode. The pipeline does **not** write parquet files to disk/blob; `CosmosDBPipelineStorage` receives parquet bytes from GraphRAG, expands each dataset into row-level Cosmos items inside `pipeline-{collection}-{version}`, and reconstructs parquet bytes on read.

The current pain point is vector container proliferation. GraphRAG currently derives a physical Cosmos vector container per embedding namespace, so indexing creates containers such as `test-2305-entity-description`, `test-2305-community-full_content`, and `test-2305-text_unit-text`. If vectors are also versioned by container, container count grows too quickly.

Recommended outcome under the scope constraint: keep the current versioned pipeline container model, but override the Cosmos vector store implementation from `backend/` so all Cosmos vectors go into one fixed `vectors` container scoped by `collectionVersion + embeddingKind`. This avoids changing vendored/core GraphRAG files while preserving active-version publish semantics.

## Recommended target design

### Cosmos fixed containers

Keep existing fixed containers:

- `collections`
- `documents`
- `indexingJobs`
- `jobEvents`
- `artifactManifest`
- `conversationSessions`
- `conversationTurns`

Add/use one fixed vector container:

- `vectors`

### Cosmos dynamic pipeline containers

Keep one pipeline output container per collection/version:

- `pipeline-{collection}-{version}`

`CosmosDBPipelineStorage` continues storing pipeline datasets row-by-row inside these containers. Because `graphrag/storage/cosmosdb_pipeline_storage.py` cannot be changed in this scope, pipeline-storage hardening is **out of scope** for this plan unless implemented by a backend-side subclass/registration hook that safely replaces the GraphRAG storage factory.

### Vector document model

Use one physical container `vectors` with partition key `/partitionKey`.

Each vector item should include:

- `id`: stable unique item id, preferably `{partitionKey}|{originalId}` or another deterministic id that avoids cross-partition collisions
- `sourceId`: original vector/document id from GraphRAG
- `partitionKey`: `{collectionId}:{version}|{embeddingKind}`
- `collectionId`
- `version`
- `collectionVersion`: `{collectionId}:{version}`
- `embeddingKind`: e.g. `entity.description`, `community.full_content`, `text_unit.text`
- `text`
- `vector`
- `attributes`

`overwrite=True` for Cosmos vectors must delete only the current `partitionKey`, not the whole `vectors` container.

Assumption for this implementation: all configured embeddings use the same vector dimension, currently `3072`, so a single `/vector` policy in one container is valid.

## Implementation steps

### 1. Add a backend-owned scoped Cosmos vector store adapter

Add a new backend module, for example:

- `backend/app/vector_stores/scoped_cosmosdb.py`

Implementation approach:

1. Define `ScopedCosmosDBVectorStore`, either subclassing `graphrag.vector_stores.cosmosdb.CosmosDBVectorStore` or implementing `BaseVectorStore` directly.
2. Do not edit `graphrag/vector_stores/cosmosdb.py`.
3. Use config kwargs from `connect(...)`:
   - `container_name`, defaulting to `vectors`
   - `collection_id`
   - `version`
   - `collection_version`
   - `embedding_kind`, falling back to `self.index_name` when GraphRAG appends schema suffixes
4. Create/use physical container `vectors` with:
   - partition key `/partitionKey`
   - vector path `/vector`
   - `diskANN` vector index when supported, with emulator fallback if needed
5. Validate required scope metadata when Cosmos cloud vectors are enabled. Fail fast if `collection_id` or `version` is missing.
6. Compute `partitionKey = f"{collection_id}:{version}|{embedding_kind}"`.
7. In `load_documents(..., overwrite=True)`, delete only documents for the current `partitionKey`.
8. Upsert documents with deterministic unique item ids and retain original GraphRAG ids in `sourceId`.
9. In `similarity_search_by_vector(...)`, restrict search to the current `partitionKey`.
10. In emulator fallback/local scoring, fetch only the current `partitionKey`.
11. In `search_by_id(...)`, read by deterministic item id and current partition key, or query by `sourceId` plus `partitionKey`.
12. In `filter_by_id(...)`, include current scope filtering and avoid SQL string injection from raw ids.
13. In `clear(...)`, delete only current scope documents or make it a no-op for backend serving/indexing. Do not delete the database.

### 2. Register the backend adapter without modifying GraphRAG

Add a small registration module, for example:

- `backend/app/vector_stores/registration.py`

Changes:

1. Import `VectorStoreFactory` and `VectorStoreType` from GraphRAG.
2. Register `ScopedCosmosDBVectorStore` for `VectorStoreType.CosmosDB.value`.
3. Make registration idempotent.
4. Call registration before any indexing/query operation can create vector stores. Good locations:
   - FastAPI lifespan startup in `backend/app/main.py`
   - worker/indexing entrypoint if indexing can run outside FastAPI lifespan
   - `load_graphrag_config(...)` as a defensive fallback before returning config

This preserves the public GraphRAG API while replacing the runtime implementation only for this backend app.

### 3. Add vector scope/runtime config helpers

Critical file:

- `backend/app/utils/helpers.py`

Reuse existing helpers:

- `_build_vector_index_name(...)`
- `_vector_store_cli_overrides(...)`
- `load_graphrag_config(...)`

Changes:

1. Introduce a fixed Cosmos vector container helper returning `vectors`.
2. Extend `_vector_store_cli_overrides(...)` to accept:
   - `collection_id`
   - `version`
3. When `CLOUD_VECTOR_STORE_TYPE=cosmosdb` and `use_cloud_vectors=True`, set:
   - `vector_store.default_vector_store.container_name = "vectors"`
   - `vector_store.default_vector_store.collection_id = collection_id`
   - `vector_store.default_vector_store.version = version`
   - `vector_store.default_vector_store.collection_version = "{collection_id}:{version}"`
4. Continue passing Cosmos endpoint, connection string, database name, and client kwargs as today.
5. Keep Azure AI Search support in generic code only if existing tests depend on it, but make the production/default path Cosmos-compatible.

Important limitation: because `graphrag/index/operations/embed_text/embed_text.py` cannot be changed, `embedding_kind` may have to be inferred from GraphRAG's schema/index name in the adapter. The adapter should normalize known suffixes into stable values such as `entity.description`, `community.full_content`, and `text_unit.text`.

### 4. Preserve activeVersion publish flow and add vector verification

Critical files:

- `backend/app/services/indexing_service.py`
- `backend/app/services/query_service.py`
- `backend/app/repositories/control_plane_repository.py`
- `backend/app/repositories/pipeline_output_repository.py`

Current flow to preserve:

1. Write pipeline output to `pipeline-{collection}-{version}`.
2. Verify required pipeline datasets.
3. Upsert artifact manifest.
4. Set `collections.activeVersion`.
5. Query reads `activeVersion` and loads that version.

Changes:

1. Keep `_verify_pipeline_output(...)` for required datasets.
2. Add lightweight vector-scope verification before `set_active_version(...)`:
   - verify `vectors` exists,
   - verify expected active embedding scopes are non-empty for configured embeddings.
3. Ensure every query path that calls `load_graphrag_config(collection_id, version=active_version, use_cloud_vectors=True)` gets vector config scoped to that active version.
4. Do not delete old vector partitions during publish; cleanup should be a separate retention operation.

### 5. Update provisioning scripts and script contract tests

Critical files:

- `scripts/provision-azure-db.sh`
- `scripts/provision-azure-db.ps1`
- `tests/unit/scripts/test_provision_azure_db_serverless_contract.py`

Recommended implementation:

1. Keep account capability checks:
   - `EnableServerless`
   - `EnableNoSQLVectorSearch`
2. Stop saying vector containers are created on-demand per collection/embedding.
3. Prefer letting the backend Cosmos vector adapter create `vectors` on first use, because it owns the vector embedding/index policy.
4. Update script output/docs to say:
   - fixed vector container: `vectors`
   - created by backend on first Cosmos vector indexing if not pre-created.
5. Update tests that currently assert “does not preprovision vector containers” so they assert no per-collection/per-embedding vector container provisioning and mention fixed `vectors` behavior instead.

### 6. Update backend settings and docs

Critical files:

- `backend/settings.yaml`
- `docs/project/en/database_schema.md`
- `docs/project/en/index_flow.md`
- `docs/project/en/architecture.md`
- `docs/project/en/query_flow.md`

Changes:

1. Set/default Cosmos vector physical container to `vectors` where backend config controls it.
2. Keep embeddings schema entries:
   - `entity.description`
   - `community.full_content`
   - `text_unit.text`
3. Document true pipeline storage behavior:
   - one container per collection/version,
   - row-level Cosmos items per dataset,
   - item ids like `{dataset}:{rowId}`,
   - not one Cosmos document containing `{dataset}.parquet` bytes.
4. Document vector design:
   - one fixed `vectors` container,
   - partition key `/partitionKey`,
   - scoped by `collectionVersion + embeddingKind`,
   - old versions retained until cleanup.
5. Update diagrams so query flow shows:
   - active version lookup,
   - pipeline container hydration,
   - vector search in `vectors` partition.
6. Explicitly document that this project overrides GraphRAG's built-in Cosmos vector store at backend runtime and does not modify files under `graphrag/`.

### 7. Defer GraphRAG core hardening

Out of scope while `graphrag/` cannot be changed:

- Editing `graphrag/vector_stores/cosmosdb.py`
- Editing `graphrag/index/operations/embed_text/embed_text.py`
- Editing `graphrag/storage/cosmosdb_pipeline_storage.py`

Deferred improvements if core changes are allowed later:

1. Move the scoped Cosmos vector implementation into `graphrag/vector_stores/cosmosdb.py`.
2. Pass explicit `embedding_kind` from `embed_text.py` instead of inferring it in the backend adapter.
3. Harden `CosmosDBPipelineStorage` directly:
   - `STARTSWITH(c.id, "{dataset}:")`
   - parameterized queries
   - persisted row metadata
   - deterministic reconstruction order
   - re-raise write failures

## Tests to add/update

### Backend runtime config tests

File:

- `backend/tests/unit/test_helpers_runtime_config.py`

Verify:

- Cosmos vector config uses physical `container_name = vectors`.
- Version/scoping metadata is passed when `version` is supplied.
- Cosmos cloud vector mode does not require Azure AI Search settings.
- Backend vector-store registration is invoked before config is used.

### Backend scoped Cosmos vector adapter tests

Add focused tests under backend tests, for example:

- `backend/tests/unit/vector_stores/test_scoped_cosmosdb.py`

Verify with fakes/mocks:

- `connect(...)` creates/uses `vectors`, not `index_name` as the container.
- Container partition key is `/partitionKey`.
- `overwrite=True` deletes only current partition scope.
- Same source id in two versions does not collide.
- Vector search filters to current `partitionKey`.
- Emulator fallback filters to current `partitionKey` before local scoring.
- `search_by_id(...)` uses current partition key/scope.
- `clear(...)` does not delete the database.

### Indexing publish verification tests

Representative files:

- existing tests under `backend/tests/unit/services/`

Verify:

- active version is not set when required vector scopes are empty.
- active version is set after pipeline datasets and vector scopes are present.
- vector verification checks the fixed `vectors` container and expected partition keys.

### Provisioning script tests

File:

- `tests/unit/scripts/test_provision_azure_db_serverless_contract.py`

Verify:

- scripts still require serverless + NoSQL vector search capability.
- scripts do not define per-embedding vector containers.
- scripts mention/use the fixed `vectors` container model.
- scripts still do not echo secret values.

### Existing backend tests to run

- `backend/tests/unit/repositories/test_pipeline_output_repository.py`
- `backend/tests/unit/services/test_query_service_cosmos_context.py`
- `backend/tests/unit/test_query_service_cosmos_only.py`
- `backend/tests/integration/test_cosmos_pipeline_emulator.py` when emulator env is available.

## Verification commands

From `F:\KL\gtog`:

```powershell
python -m pytest backend/tests/unit/test_helpers_runtime_config.py
```

```powershell
python -m pytest backend/tests/unit/vector_stores/test_scoped_cosmosdb.py
```

```powershell
python -m pytest backend/tests/unit/repositories/test_pipeline_output_repository.py
```

```powershell
python -m pytest tests/unit/scripts/test_provision_azure_db_serverless_contract.py
```

If Cosmos emulator is configured:

```powershell
python -m pytest backend/tests/integration/test_cosmos_pipeline_emulator.py
```

Broader regression:

```powershell
python -m pytest backend/tests/unit tests/unit/scripts/test_provision_azure_db_serverless_contract.py
```

Static checks if available:

```powershell
python -m ruff check backend tests/unit/scripts
```

```powershell
python -m pyright
```

Manual Azure validation after implementation and deploy/reindex:

```powershell
az cosmosdb sql container list --account-name cdb-gtog-prod-alt --resource-group rg-gtog-prod --database-name gtog-control --query "[].name" --output table
```

Expected after reindex:

- fixed `vectors` exists,
- new `pipeline-{collection}-{version}` exists,
- no new per-embedding vector containers are created.

## Deployment/reset approach for live data

Recommended approach: breaking deploy with database reset/reprovision and full reindex.

Steps:

1. Deploy backend code that registers the scoped Cosmos vector adapter.
2. Reprovision or clean the Cosmos database so legacy pipeline/vector containers are removed.
3. Recreate fixed control-plane containers and allow the backend adapter to create/use the fixed `vectors` container.
4. Re-upload or preserve source documents in Blob, then reindex each collection.
5. Confirm new index writes pipeline rows to `pipeline-{collection}-{newVersion}` and vectors to the fixed `vectors` container.
6. Confirm `collections.activeVersion` only flips after pipeline and vector verification.

No backfill or compatibility migration is required.

## Risks and mitigations

1. **Runtime override fragility**
   - Mitigation: register the backend adapter in both startup and `load_graphrag_config(...)`, make registration idempotent, and add tests proving `VectorStoreFactory` points Cosmos DB to the backend adapter.

2. **Embedding kind inference is less explicit than changing GraphRAG core**
   - Mitigation: normalize known GraphRAG schema/index-name suffixes in one backend function and test all configured embeddings.

3. **Single vector dimension constraint**
   - Mitigation: validate all configured Cosmos embedding dimensions are equal; current settings use `3072`.

4. **Large logical partitions in `vectors`**
   - Mitigation: start with `{collectionVersion}|{embeddingKind}`; revisit bucketed partition keys only if Cosmos partition limits or RU metrics require it.

5. **Accidental cross-version vector search**
   - Mitigation: require active version scope in runtime config and add tests that assert partition filtering.

6. **Pipeline storage bugs remain in GraphRAG core**
   - Mitigation: keep existing backend `_verify_pipeline_output(...)`; defer direct `CosmosDBPipelineStorage` hardening until `graphrag/` edits are allowed.

7. **Database reset is destructive**
   - Mitigation: confirm source documents are preserved/re-uploadable before reprovisioning, then run full reindex.

8. **Reindex cost/time**
   - Mitigation: run one non-critical collection first, then batch reindex remaining collections.
