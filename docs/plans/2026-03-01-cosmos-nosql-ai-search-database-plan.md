# Cosmos DB (NoSQL) + Azure AI Search Database Plan for GraphRAG (gtog)

**Date**: 2026-03-01
**Status**: Proposed
**Scope**: Production-ready database layer and query/indexing data flow (phased rollout)

## 1) Objective

Build a production-grade database architecture for this project using:

- **Azure Cosmos DB for NoSQL** as the primary application database (control plane + serving context)
- **Azure AI Search** for vector/hybrid retrieval

while eliminating query-time dependence on downloading parquet files for every request.

### Implementation compatibility constraints (must be honored during rollout)

- GraphRAG query APIs currently accept pandas DataFrames for local/global/drift/tog inputs, not repository abstractions (`graphrag/api/query.py`).
- This repository’s vector-store config model expects `embeddings_schema` for per-embedding schema overrides (`graphrag/config/models/vector_store_config.py`).
- Current Azure AI Search vector-store implementation in this codebase uses generic fields (`id`, `vector`, `text`, `attributes`) and does not directly apply collection/version filters in vector similarity calls (`graphrag/vector_stores/azure_ai_search.py`).
- Migration phases must therefore include explicit compatibility decisions and adapters before hard cutover.

## 2) Current Problems to Resolve

Current backend state still has production gaps:

1. Query path loads parquet from blob per request in cloud mode
   - `backend/app/services/query_service.py:36`
   - Called by global/local/tog/drift loading points (`:140`, `:205`, `:280`, `:382`)
2. Runtime prompt file generation is still present
   - `backend/app/utils/helpers.py:61` and call in `:160`
3. Debug endpoint behavior should be disabled in production
   - `backend/app/routers/search.py:95`
4. Duplicate/unreachable exception block in collections router
   - `backend/app/routers/collections.py:51`

## 3) Target Architecture

```text
Frontend (Next.js)
   |
   v
Backend API (FastAPI, stateless)
   |---- Azure AI Search (vector/hybrid retrieval)
   |---- Azure Cosmos DB NoSQL (control + serving data)
   |---- Service Bus (indexing job queue)

Indexing Worker (separate process/service)
   |---- runs GraphRAG indexing
   |---- writes serving docs to Cosmos (versioned)
   |---- writes vector docs to AI Search (same version)
   |---- atomically flips active version pointer
```

### Responsibility split

- **Cosmos DB**
  - collections/documents metadata
  - indexing jobs + job events
  - serving context (entities, relationships, text units, communities, reports)
  - active serving version per collection
- **Azure AI Search**
  - vector index and hybrid retrieval
  - top-k candidate retrieval for local/global/drift/tog
- **Blob (optional/archival)**
  - raw uploads and/or archival artifacts only
  - not required in hot query path once migration completes

## 4) Data Model (Cosmos DB NoSQL)

Recommended partition key for application-owned collection-scoped containers: `/collectionId`.

**Compatibility note:** built-in GraphRAG Cosmos implementations in this repository use `/id` for internal storage/vector containers. Keep app-owned containers and GraphRAG-internal containers intentionally separate, or apply a tested adapter strategy before unifying partition keys.

### 4.1 Control-plane containers

1. `collections`
   - id, collectionId, name, description, status, createdAt, updatedAt, activeVersion
2. `documents`
   - id, collectionId, sourcePath, mimeType, sizeBytes, sha256, uploadedAt, status
3. `indexingJobs`
   - id, collectionId, status, attempt, maxAttempts, requestedAt, startedAt, finishedAt, error
4. `jobEvents`
   - id, collectionId, jobId, fromStatus, toStatus, timestamp, metadata
5. `artifactManifest`
   - id, collectionId, version, artifactName, counts, checksum, createdAt

### 4.2 Serving containers

1. `entities`
2. `relationships`
3. `textUnits`
4. `communities`
5. `communityReports`
6. `covariates` (optional)

Each serving document should include at minimum:

- `id`
- `collectionId`
- `version`
- domain fields needed by query assembly and citation

## 5) AI Search Index Model

### Target retrieval contract

Preferred filterable fields:

- `id` (key)
- `collectionId` (filterable)
- `version` (filterable)
- `content` (searchable)
- `embedding` (vector, dimensions aligned to model output)

Preferred query filter behavior:

- `collectionId == <collection_id>`
- `version == <activeVersion from Cosmos>`

### Compatibility decision required before cutover

Current repository implementation uses generic vector-store fields (`id`, `vector`, `text`, `attributes`) and does not yet enforce collection/version filtering in vector similarity calls. Before enabling production cutover, choose one of the following and document it in code:

1. **Schema + query extension path**: extend Azure AI Search index schema and query call path to include and apply `collectionId` + `version` filters.
2. **Index isolation path**: isolate by index naming convention (e.g., per-collection/per-version) and rely on active index selection rather than runtime filters.

Whichever path is selected must guarantee retrieval/context version consistency for every request.

## 6) Versioned Serving Strategy (No Partial Reads)

### Write path

Worker writes all outputs to `version = vNext` first:

1. Upsert serving docs to Cosmos with `vNext`
2. Upsert retrieval docs/vectors to AI Search with `vNext`
3. Validate counts/checksums against manifest
4. Atomically update `collections.activeVersion = vNext` (optimistic concurrency)

### Read path

API request:

1. Read `activeVersion` from Cosmos
2. Retrieve candidate IDs from AI Search filtered by collection + activeVersion
3. Batch point-read context docs from Cosmos for that same version
4. Build context objects for GraphRAG API call via an explicit Cosmos→DataFrame assembly layer (until/if query engine abstractions change)

Result: zero request-level dependency on blob parquet downloads.

## 7) Phased Implementation Plan

## Phase 0 — Stabilize existing production gaps

- Remove runtime prompt generation, enforce fail-fast validation of prompt files
- Disable ToG debug endpoint in production
- Clean duplicate exception block in collections router
- Keep blob/parquet route as temporary migration fallback only
- Add a configuration compatibility checkpoint against the active GraphRAG version used in this repo:
  - validate vector-store schema key naming (`embeddings_schema` vs any legacy docs naming)
  - validate storage/cache/output/reporting key names and type values from the installed package
  - validate Azure AI Search and CosmosDB option support in current code paths

**Done when**
- No runtime prompt writes
- Debug endpoint disabled in prod
- Router cleanup complete
- Config compatibility checklist is documented and passing

## Phase 1 — Control plane migration to Cosmos

- Implement Cosmos repositories for collections/documents/jobs/events/manifests
- Move metadata operations from blob JSON/local assumptions to Cosmos
- Add idempotent job enqueue + explicit state transitions

**Done when**
- Collection/document/indexing state is persisted in Cosmos and restart-safe

## Phase 2 — Serving model in Cosmos

- Materialize entities/relationships/text_units/communities/reports into Cosmos serving containers
- Build versioned writes and activeVersion pointer update

**Done when**
- Query context can be fully reconstructed from Cosmos for one active version

## Phase 3 — Query path cutover (AI Search + Cosmos)

- Replace `_blob_parquet` query hot path in `backend/app/services/query_service.py`
- Introduce retriever + context reader abstraction
- Add a `DataFrameAssembler` (or equivalent adapter) that reconstructs the DataFrames required by current GraphRAG query APIs from Cosmos serving documents
- Route local/global/drift/tog to new retrieval pipeline

**Done when**
- No per-request blob parquet read in normal query traffic
- All query methods (local/global/drift/tog) run from Cosmos-backed assembled context without behavior regressions

## Phase 4 — Workerized indexing and reliability

- Move indexing execution to worker (queue-driven)
- Retry/dead-letter policy, idempotent upserts
- Add health/metrics and structured job events

**Done when**
- Indexing jobs are asynchronous, observable, retryable, and recoverable

## Phase 5 — Security and operations hardening

- Private endpoints and network restrictions
- Managed identity + Key Vault integration
- RU autoscale policy and 429 backoff tuning
- SLO/alert dashboards, load test, DR runbooks

**Done when**
- Meets production security posture and performance targets

## 8) Validation Checklist Per Phase

### Functional
- Collection CRUD works end-to-end
- Upload + index + query works for local/global/tog/drift
- New index version becomes visible only after atomic switch

### Reliability
- API/worker restart during indexing does not corrupt state
- Failed job retries up to maxAttempts then dead-letter
- Idempotent reprocessing does not duplicate data

### Performance
- No parquet download in hot query path
- Stable p95/p99 under expected concurrency
- Acceptable RU consumption and AI Search latency

### Security
- Production debug routes disabled
- Secrets externalized and rotated
- Private network access policy enforced

## 9) Risks and Mitigations

1. **Dual-store consistency (Cosmos + AI Search)**
   - Mitigation: versioned writes + atomic activeVersion switch
2. **RU spikes in Cosmos**
   - Mitigation: autoscale, batch reads, projection minimization
3. **Schema drift during migration**
   - Mitigation: explicit versioned contracts and migration scripts
4. **Cutover regressions**
   - Mitigation: shadow read parity tests + canary rollout + rollback flag

## 10) Cutover and Rollback Strategy

### Cutover

1. Deploy repositories and worker in dual-write mode
2. Backfill Cosmos serving docs from current artifacts
3. Run shadow read parity tests between legacy and new paths
4. Enable canary traffic (10% -> 50% -> 100%)
5. Remove legacy parquet hot-path once stable

### Rollback

- Keep runtime flag to switch query path back to legacy for one release window
- Preserve previous activeVersion until rollback validation completes

## 11) Notes for This Repository

- Prompt files already exist at `backend/prompts`; no prompt seeding needed.
- Current `backend/settings.yaml` already points vector store to Azure AI Search.
- In this repository’s GraphRAG code path, prefer `embeddings_schema` (not `index_schema`) for per-embedding vector schema overrides.
- Current `.env` contains cloud credentials; production should move secrets to Key Vault and managed identity.

---

This plan is intended as the execution baseline for implementing a production-ready database layer with **Cosmos DB for NoSQL + Azure AI Search** in this repository.
