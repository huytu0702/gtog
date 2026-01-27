# Postgres Backend Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the PostgreSQL-backed backend flow for indexing and search, removing filesystem dependencies and enforcing correct job lifecycle behavior.

**Architecture:** Implement DB ingestion via a GraphRAG-to-DB adapter and repositories, make the worker write outputs into Postgres, and make search endpoints read only from Postgres. Remove or quarantine filesystem-based helpers and services to avoid drift.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, RQ/Redis, PostgreSQL (pgvector)

---

### Task 1: Enforce index-run conflict rules

**Files:**
- Modify: `backend/app/services/indexing_service_db.py:1-120`
- Modify: `backend/app/routers/indexing.py:1-160`
- Test: `backend/tests/unit/services/test_indexing_service_db.py` (create)

**Step 1: Write the failing test**

```python
# backend/tests/unit/services/test_indexing_service_db.py

def test_start_indexing_rejects_if_run_in_progress(db_session, collection):
    service = IndexingServiceDB(db_session)
    service._index_run_repo.create(
        collection_id=collection.id,
        status=IndexRunStatus.RUNNING,
    )

    with pytest.raises(HTTPException) as exc:
        service.start_indexing(collection.id)

    assert exc.value.status_code == 409
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/services/test_indexing_service_db.py::test_start_indexing_rejects_if_run_in_progress -v`
Expected: FAIL (no conflict handling)

**Step 3: Write minimal implementation**

```python
# backend/app/services/indexing_service_db.py
latest = self._index_run_repo.get_latest_for_collection(collection_id)
if latest and latest.status in {IndexRunStatus.QUEUED, IndexRunStatus.RUNNING}:
    raise HTTPException(status_code=409, detail="Indexing already running")
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/services/test_indexing_service_db.py::test_start_indexing_rejects_if_run_in_progress -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/tests/unit/services/test_indexing_service_db.py backend/app/services/indexing_service_db.py backend/app/routers/indexing.py
git commit -m "fix(backend): reject concurrent index runs"
```

---

### Task 2: Implement GraphRAGDbAdapter skeleton and repositories for graph outputs

**Files:**
- Modify: `backend/app/services/graphrag_db_adapter.py:1-200`
- Create: `backend/app/repositories/entities.py`
- Create: `backend/app/repositories/relationships.py`
- Create: `backend/app/repositories/communities.py`
- Create: `backend/app/repositories/community_reports.py`
- Create: `backend/app/repositories/text_units.py`
- Create: `backend/app/repositories/covariates.py`
- Create: `backend/app/repositories/embeddings.py`
- Modify: `backend/app/repositories/__init__.py:1-200`
- Test: `backend/tests/unit/services/test_graphrag_db_adapter.py` (create)

**Step 1: Write the failing test**

```python
# backend/tests/unit/services/test_graphrag_db_adapter.py

def test_adapter_inserts_entities(db_session, collection, index_run):
    adapter = GraphRAGDbAdapter(db_session)
    entities = [
        {"id": "e1", "title": "Entity 1", "type": "Person"},
        {"id": "e2", "title": "Entity 2", "type": "Org"},
    ]

    adapter.insert_entities(collection.id, index_run.id, entities)

    rows = db_session.query(Entity).all()
    assert len(rows) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/services/test_graphrag_db_adapter.py::test_adapter_inserts_entities -v`
Expected: FAIL (adapter/repo missing)

**Step 3: Write minimal implementation**

```python
# backend/app/services/graphrag_db_adapter.py
class GraphRAGDbAdapter:
    def __init__(self, db):
        self.db = db
        self._entities = EntityRepository(db)

    def insert_entities(self, collection_id, index_run_id, entities):
        payloads = [
            {"collection_id": collection_id, "index_run_id": index_run_id, **e}
            for e in entities
        ]
        self._entities.bulk_insert(payloads)
```

```python
# backend/app/repositories/entities.py
class EntityRepository:
    def __init__(self, db):
        self.db = db

    def bulk_insert(self, payloads):
        self.db.bulk_insert_mappings(Entity, payloads)
        self.db.commit()
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/services/test_graphrag_db_adapter.py::test_adapter_inserts_entities -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/graphrag_db_adapter.py backend/app/repositories/entities.py backend/app/repositories/__init__.py backend/tests/unit/services/test_graphrag_db_adapter.py
git commit -m "feat(backend): add db adapter entity inserts"
```

---

### Task 3: Extend adapter for relationships, communities, reports, text units, covariates, embeddings

**Files:**
- Modify: `backend/app/services/graphrag_db_adapter.py:1-200`
- Modify: `backend/app/repositories/*.py`
- Test: `backend/tests/unit/services/test_graphrag_db_adapter.py`

**Step 1: Write failing tests (one per output type)**

```python
# add tests for relationships, communities, community_reports, text_units, covariates, embeddings
```

**Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/unit/services/test_graphrag_db_adapter.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Add insert_relationships, insert_communities, insert_community_reports,
# insert_text_units, insert_covariates, insert_embeddings using bulk_insert_mappings
```

**Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/unit/services/test_graphrag_db_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/graphrag_db_adapter.py backend/app/repositories backend/tests/unit/services/test_graphrag_db_adapter.py
git commit -m "feat(backend): support GraphRAG output inserts"
```

---

### Task 4: Implement worker pipeline to run GraphRAG and persist outputs

**Files:**
- Modify: `backend/app/worker/tasks.py:1-200`
- Modify: `backend/app/services/indexing_service_db.py:1-120`
- Modify: `backend/app/services/document_service_db.py:1-200`
- Test: `backend/tests/integration/test_indexing_worker_db.py` (create)

**Step 1: Write failing integration test**

```python
# backend/tests/integration/test_indexing_worker_db.py

def test_worker_persists_outputs(db_session, collection, document_bytes):
    run = IndexRunRepository(db_session).create(collection_id=collection.id, status=IndexRunStatus.QUEUED)
    # enqueue or call task directly
    run_indexing_job(run.id)

    assert db_session.query(Entity).count() > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/integration/test_indexing_worker_db.py -v`
Expected: FAIL (task stubbed)

**Step 3: Write minimal implementation**

```python
# backend/app/worker/tasks.py
# 1) load collection/documents
# 2) call GraphRAG indexing
# 3) map outputs through GraphRAGDbAdapter
# 4) mark index_run completed/failed
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/integration/test_indexing_worker_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/worker/tasks.py backend/tests/integration/test_indexing_worker_db.py
git commit -m "feat(backend): persist GraphRAG outputs from worker"
```

---

### Task 5: Replace QueryServiceDB with Postgres-backed search

**Files:**
- Modify: `backend/app/services/query_service_db.py:1-200`
- Modify: `backend/app/repositories/query.py:1-200`
- Modify: `backend/app/routers/search.py:1-200`
- Test: `backend/tests/integration/test_search_db.py` (create)

**Step 1: Write failing integration test**

```python
# backend/tests/integration/test_search_db.py

def test_search_returns_results_from_db(client, collection, indexed_run):
    response = client.get(f"/collections/{collection.id}/search?query=foo")
    assert response.status_code == 200
    assert response.json()["answer"]
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/integration/test_search_db.py -v`
Expected: FAIL (empty response)

**Step 3: Write minimal implementation**

```python
# backend/app/repositories/query.py
# Add methods to fetch latest run + load entities/relationships/communities

# backend/app/services/query_service_db.py
# Use query repository to build SearchResponse
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/integration/test_search_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/query_service_db.py backend/app/repositories/query.py backend/app/routers/search.py backend/tests/integration/test_search_db.py
git commit -m "feat(backend): implement postgres-backed search"
```

---

### Task 6: Remove filesystem storage helpers and update config/docs

**Files:**
- Modify: `backend/app/utils/helpers.py:1-200`
- Modify: `backend/app/config.py:1-120`
- Modify: `backend/app/services/__init__.py:1-80`
- Modify: `backend/README.md:1-300`
- Test: `backend/tests/unit/test_helpers.py` (remove if obsolete)

**Step 1: Write failing test (if helper removal needs safety)**

```python
# If tests exist, update them to expect helper removal or no filesystem usage
```

**Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/unit -v`
Expected: FAIL if filesystem helper assumptions exist

**Step 3: Write minimal implementation**

```python
# Remove or stub helpers to prevent filesystem reads
# Remove filesystem settings from config
# Remove storage_service exports
# Update README to describe Postgres-only flow
```

**Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/unit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/utils/helpers.py backend/app/config.py backend/app/services/__init__.py backend/README.md
# Include any removed tests if applicable
git commit -m "refactor(backend): remove filesystem storage helpers"
```

---

### Task 7: Fix direct-run import in main

**Files:**
- Modify: `backend/app/main.py:80-110`
- Test: `backend/tests/unit/test_main.py` (create if not present)

**Step 1: Write the failing test**

```python
# backend/tests/unit/test_main.py

def test_main_imports_settings():
    import importlib
    import backend.app.main as main
    assert hasattr(main, "settings")
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_main.py -v`
Expected: FAIL (settings missing)

**Step 3: Write minimal implementation**

```python
# backend/app/main.py
from .config import settings
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/test_main.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/unit/test_main.py
git commit -m "fix(backend): import settings in main"
```

---

### Task 8: End-to-end verification

**Files:**
- Test: `backend/tests/integration` and `backend/tests/smoke`

**Step 1: Run integration tests**

Run: `pytest backend/tests/integration -v`
Expected: PASS

**Step 2: Run smoke tests**

Run: `pytest backend/tests/smoke -v`
Expected: PASS

**Step 3: Commit (if any changes from fixes)**

```bash
git add backend/tests
# Only if tests were updated/fixed during verification
git commit -m "test(backend): stabilize integration and smoke tests"
```

---

### Notes
- If GraphRAG output formats are not in dict form, insert a thin mapping layer in `GraphRAGDbAdapter` (do not add new dependencies).
- Keep inserts batched with `bulk_insert_mappings` for performance, and use explicit transactions for each output type.
- Ensure Search reads only the latest **completed** run for a collection.
- Update any CI/config or docs that still mention `storage/collections`.
