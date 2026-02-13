# CosmosDB Emulator Full-Stack Development Design

Date: 2026-02-13
Status: Approved (brainstorming)
Decision: Option C (full local platform package), with one full-stack Docker Compose file as the primary development entrypoint.

## 1. Context and Goals

The project will use Azure Cosmos DB long-term. Phase 1 should implement a local-first path using the Azure Cosmos DB Emulator (Docker), while ensuring:

- GraphRAG uses CosmosDB for indexing artifacts, cache, and vector store.
- Backend uses CosmosDB for collection metadata, uploaded documents, prompts, and query/index data access.
- Local development uses a single full-stack compose workflow, while allowing quick Cosmos-only startup for fast testing.
- End-to-end backend smoke path validates create collection → upload document → index → query.

## 2. Scope

### In scope

1. Full Cosmos-backed collection domain:
   - Collection metadata in CosmosDB.
   - Uploaded documents in CosmosDB.
   - Collection prompts in CosmosDB.
2. GraphRAG configuration profile for Cosmos emulator mode.
3. Backend indexing and query path updates to work from GraphRAG storage abstraction in Cosmos mode (not direct local parquet file reads).
4. Single full-stack Docker Compose for development.
5. Scripts/docs for bootstrap, smoke tests, troubleshooting, and reset.

### Out of scope

- Production Cosmos account deployment and IaC.
- Performance tuning beyond emulator constraints.
- CI/CD cloud environment rollout.

## 3. Current-State Findings

### Existing Cosmos support in GraphRAG

- Cosmos pipeline storage exists: `graphrag/storage/cosmosdb_pipeline_storage.py`.
- Cosmos vector store exists: `graphrag/vector_stores/cosmosdb.py`.
- Storage/vector factories already register Cosmos types:
  - `graphrag/storage/factory.py`
  - `graphrag/vector_stores/factory.py`
- Index pipeline already instantiates storage/cache from config:
  - `graphrag/index/run/run_pipeline.py`

### Current backend limitations

- Backend config loader currently forces file storage overrides (`backend/app/utils/helpers.py`), which blocks Cosmos mode.
- Query service reads local parquet files directly (`backend/app/services/query_service.py`) instead of storage abstraction.
- Collection/doc/prompt management is filesystem-centric (`backend/app/services/storage_service.py`).

## 4. Architecture Decisions

## 4.1 Single full-stack Docker Compose

Use one compose file as source of truth for local development.

- Default run: all services (Cosmos emulator, backend, optional frontend).
- Fast test run: service-targeted startup for emulator only (same compose file, same network/env source).

Rationale:
- Prevents config drift between quick and full runs.
- Simplifies onboarding and operational documentation.

## 4.2 Cosmos profile activation model

Use a separate Cosmos emulator settings profile (not default replacement).

- Keep existing defaults stable.
- Add dedicated Cosmos profile and explicit selection mechanism in backend settings.

Rationale:
- Safe migration and easier troubleshooting.
- No accidental behavior changes in existing local workflows.

## 4.3 Full collection domain in Cosmos

Everything collection-related is stored in Cosmos:

- Collections metadata
- Documents
- Prompts
- Index artifacts/output
- Cache
- Vectors

Rationale:
- Matches future architecture intent.
- Eliminates split-brain storage behavior during development.

## 5. High-Level Data Flow

1. Developer starts stack with the single compose file.
2. Backend loads Cosmos emulator profile.
3. Collection create API writes collection record to Cosmos.
4. Document upload API writes document payload + metadata to Cosmos.
5. Prompt templates are created/read from Cosmos for each collection.
6. Indexing endpoint triggers `graphrag.api.build_index(...)` using Cosmos-based input/output/cache/vector config.
7. Query endpoint loads required tables through storage abstraction (Cosmos), then calls GraphRAG query API.
8. Responses are returned through existing API contract.

## 6. Component Design

## 6.1 Infrastructure

Planned files:
- `docker-compose.dev.yaml` (single full-stack compose)
- `.env.cosmos-emulator.example`

Compose includes:
- Cosmos emulator service (persistent volume)
- Backend service
- Frontend service (optional but present in full-stack compose)

## 6.2 Configuration

Planned files/changes:
- `backend/settings.cosmos-emulator.yaml` (new GraphRAG profile)
- `backend/app/config.py` (support selecting settings profile path)
- `backend/app/utils/helpers.py` (remove forced file-only overrides when Cosmos profile selected)

## 6.3 Backend data services

Refactor `backend/app/services/storage_service.py` into a storage-access layer that supports Cosmos-backed repositories in Cosmos mode:

- Collection repository (Cosmos)
- Document repository (Cosmos)
- Prompt repository (Cosmos)

Maintain endpoint contracts in:
- `backend/app/routers/collections.py`
- `backend/app/routers/documents.py`

## 6.4 Indexing and query path

- Indexing service keeps `api.build_index(...)` flow but with Cosmos profile values.
- Query service replaces direct `pd.read_parquet(path)` with storage abstraction-based loads.
- Indexed-state validation moves from local file existence checks to storage-based checks.

## 7. Error Handling Strategy

1. Fail fast on missing/invalid Cosmos configuration.
2. No silent fallback from Cosmos profile to file profile.
3. Explicit errors for emulator connectivity/certificate issues.
4. Respect emulator limitations and rely on existing GraphRAG fallbacks in vector store implementation.
5. Provide explicit reset command/script for destructive cleanup.

## 8. Testing and Validation Plan

## 8.1 Required smoke flow (backend API)

1. Create collection.
2. Upload document.
3. Start indexing and verify completion.
4. Run at least one query method successfully.

All steps must execute against Cosmos-backed storage.

## 8.2 Additional checks

- Persistence check across restart.
- Negative checks:
  - emulator unavailable
  - bad connection string
  - missing required config

## 8.3 Test artifacts

- New backend integration test for end-to-end Cosmos path.
- Storage/service unit tests for Cosmos repositories.
- Bootstrap + smoke scripts to automate local validation.

## 9. Rollout Sequence

Phase 1: Compose + profile plumbing
- add single full-stack compose
- add Cosmos emulator env/profile
- backend profile selection

Phase 2: Collection domain migration to Cosmos
- metadata/documents/prompts repositories
- router/service integration

Phase 3: GraphRAG config alignment
- remove forced file overrides
- ensure index pipeline runs with Cosmos input/output/cache/vector

Phase 4: Query/validation abstraction
- replace local parquet reads with storage abstraction
- convert indexed checks to storage-based logic

Phase 5: DX + verification hardening
- scripts, troubleshooting docs, reset workflow
- integration and smoke tests

## 10. Risks and Mitigations

1. Emulator feature gaps vs cloud service
- Mitigation: document limitations; keep behavior explicit.

2. Migration complexity from filesystem assumptions
- Mitigation: phase rollout and parity tests.

3. Developer friction from cert/connectivity setup
- Mitigation: bootstrap script and clear troubleshooting guide.

4. Data reset confusion with persistent volumes
- Mitigation: explicit reset command and warnings.

## 11. Success Criteria

Phase 1 is successful when all conditions are met:

1. Collections, documents, prompts are stored in Cosmos.
2. Index artifacts/cache and vectors are stored in Cosmos.
3. Backend smoke API flow passes end-to-end using Cosmos emulator profile.
4. GraphRAG index + at least one query method pass using Cosmos emulator profile.
5. Data persists across backend/emulator restart.
6. One full-stack compose file supports both full runs and Cosmos-only quick runs via service targeting.
