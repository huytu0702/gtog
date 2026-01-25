# PostgreSQL Storage Design (GraphRAG Backend)

Date: 2026-01-24

## Summary
Move all backend storage to PostgreSQL (Docker Compose) using pgvector for embeddings, SQLAlchemy ORM + Alembic for migrations, and a Redis-backed worker queue for indexing jobs. Filesystem storage is removed; all GraphRAG outputs are persisted in the database.

---

## Architecture
- **FastAPI API layer** becomes a thin ORM-backed service (collections/documents/index/search).
- **PostgreSQL + pgvector** stores collections, documents (bytea + extracted text), indexing runs, graph tables, and embeddings.
- **Worker queue** (Redis + Celery/RQ) runs indexing jobs and writes outputs to DB.
- **GraphRAG integration layer** writes indexing outputs to DB and reads query inputs from DB.
- **Docker Compose** runs `postgres`, `redis`, `backend`, and `worker`.

---

## Components
- **FastAPI routers**: same endpoints, now backed by ORM calls and job enqueueing.
- **SQLAlchemy ORM + Alembic**: schema and migrations.
- **Worker**: executes indexing pipeline; writes outputs; updates `index_runs` status.
- **GraphRAG adapter**: maps GraphRAG outputs into DB tables.

---

## Data Model
All GraphRAG output tables include shared fields:
- `id` (UUID)
- `human_readable_id` (int, per-run)

Operational tables:
- `collections`: id, name, description, created_at
- `index_runs`: id, collection_id, status, started_at, finished_at, error

GraphRAG output tables (aligned with `docs/index/outputs.md`):
- `documents`: id, human_readable_id, collection_id, index_run_id, title, text, metadata (jsonb), plus storage columns filename, content_type, bytes (bytea), created_at
- `entities`: id, human_readable_id, collection_id, index_run_id, title, type, description, frequency, degree, x, y
- `relationships`: id, human_readable_id, collection_id, index_run_id, source_entity_id, target_entity_id, source, target, description, weight, combined_degree
- `communities`: id, human_readable_id, collection_id, index_run_id, community (int), parent (int), level, title, period, size
- `community_reports`: id, human_readable_id, collection_id, index_run_id, community (int), parent (int), level, title, summary, full_content, rank, rating_explanation, findings (jsonb), full_content_json (jsonb), period, size
- `text_units`: id, human_readable_id, collection_id, index_run_id, text, n_tokens
- `covariates`: id, human_readable_id, collection_id, index_run_id, covariate_type, type, description, subject_id, object_id, status, start_date, end_date, source_text, text_unit_id
- `embeddings`: id, collection_id, index_run_id, type (text_unit/entity/community_report), ref_id, vector (pgvector)

Join tables (normalized list fields):
- `document_text_units` (document_id, text_unit_id) — documents.text_unit_ids and text_units.document_ids
- `text_unit_entities` (text_unit_id, entity_id) — text_units.entity_ids / entities.text_unit_ids
- `text_unit_relationships` (text_unit_id, relationship_id) — text_units.relationships_ids / relationships.text_unit_ids
- `community_entities` (community_id, entity_id) — communities.entity_ids
- `community_relationships` (community_id, relationship_id) — communities.relationship_ids
- `community_text_units` (community_id, text_unit_id) — communities.text_unit_ids
- `community_hierarchy` (parent_id, child_id) — communities.children (and community_reports.children)

Indexing writes all outputs with the current `index_run_id`.
Queries read from the latest completed `index_run_id` for a collection.

---

## Data Flow
**Indexing**
1) Upload document -> store in `documents` (bytea + text).
2) Start indexing -> create `index_runs` (status=queued).
3) Worker runs pipeline -> writes graph outputs + embeddings.
4) Worker updates `index_runs` to completed/failed.

**Query**
Search endpoints read from DB tables (latest completed run), no parquet files.

---

## Error Handling
- **409** on collection/document conflicts (unique constraints).
- **202** for accepted indexing jobs; **409** if already running.
- `index_runs` tracks `queued/running/completed/failed` with error detail.
- Writes use transactions; batch inserts for large tables.

---

## Testing Strategy
- **Unit**: ORM models, service logic, conflict cases.
- **Worker**: job lifecycle, failure paths, idempotent reindex runs.
- **Integration**: docker compose (Postgres + Redis), end-to-end indexing + search.
- **API contract**: response shapes unchanged.
- **Performance**: batch insert sanity, vector index usage.
