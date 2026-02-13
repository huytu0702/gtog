# CosmosDB Emulator Phase 4 (Query + Validation Storage Abstraction) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove backend query-path dependence on direct local parquet file reads and migrate indexed-state checks to storage-abstraction logic that works for both file and Cosmos modes.

**Architecture:** Introduce a query data loader abstraction in backend that retrieves required tables by logical keys through GraphRAG storage APIs (`load_table_from_storage` + `create_storage_from_config`) instead of filesystem paths. Refactor validation helpers to detect required indexed artifacts via storage `has()` checks in cosmos mode and file existence checks in file mode.

**Tech Stack:** FastAPI backend services/utils, pandas, GraphRAG storage utilities, pytest.

---

### Task 1: Add failing tests to capture current direct parquet dependency in query service

**Files:**
- Modify/Create: `backend/tests/unit/test_query_service_storage_mode.py`
- Inspect/Modify: `backend/app/services/query_service.py`

**Step 1: Write failing tests**

```python
from unittest.mock import patch
from backend.app.services.query_service import QueryService


@patch("backend.app.services.query_service.pd.read_parquet")
async def test_global_search_does_not_call_read_parquet_in_cosmos_mode(mock_read, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    svc = QueryService()

    # mock loader + graphrag query call to avoid heavy runtime
    with patch("backend.app.services.query_service.load_table_from_storage") as mock_load:
        mock_load.return_value = ...
        with patch("backend.app.services.query_service.api.global_search") as mock_search:
            mock_search.return_value = ("ok", {})
            await svc.global_search("demo", "q")

    mock_read.assert_not_called()
```

Add analogous tests for `local_search`, `drift_search`, `tog_search`.

**Step 2: Run tests and confirm fail**

Run:

```bash
pytest backend/tests/unit/test_query_service_storage_mode.py -v
```

Expected: FAIL because current code still uses `pd.read_parquet` on paths from `get_search_data_paths`.

**Step 3: Commit failing tests (optional if workflow allows red commits)**

```bash
git add backend/tests/unit/test_query_service_storage_mode.py
git commit -m "test: capture parquet path coupling in query service"
```

---

### Task 2: Implement query data loader abstraction in query service

**Files:**
- Modify: `backend/app/services/query_service.py`
- Modify: `backend/app/utils/helpers.py` (if helper additions needed)
- Test: `backend/tests/unit/test_query_service_storage_mode.py`

**Step 1: Implement minimal storage-backed loader helpers**

In `query_service.py`, add helper methods:

```python
def _get_storage(self, config):
    return create_storage_from_config(config.output)

async def _load_required_tables(self, config, method: str) -> dict[str, pd.DataFrame]:
    # always: entities, communities, community_reports
    # method in local/drift/tog: + text_units, relationships
    # local optional: covariates (best-effort)
```

Use `load_table_from_storage("entities", storage)` etc. key names (without `.parquet`, matching GraphRAG helpers).

**Step 2: Replace direct path reads**

For each query method in `QueryService`:
- remove `get_search_data_paths(...)` + `pd.read_parquet(...)`
- call `_load_required_tables(...)`
- pass DataFrames to GraphRAG API calls unchanged.

**Step 3: Run unit tests**

Run:

```bash
pytest backend/tests/unit/test_query_service_storage_mode.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/app/services/query_service.py backend/app/utils/helpers.py backend/tests/unit/test_query_service_storage_mode.py
git commit -m "refactor: load query tables via graphrag storage abstraction"
```

---

### Task 3: Add validation tests for indexed-state checks in cosmos mode

**Files:**
- Modify/Create: `backend/tests/unit/test_helpers_indexed_validation.py`
- Modify: `backend/app/utils/helpers.py`

**Step 1: Write failing tests**

```python
from backend.app.utils.helpers import validate_collection_indexed


def test_validate_collection_indexed_uses_storage_has_in_cosmos_mode(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")

    # mock load config and storage.has responses
    # has entities/communities/community_reports => True
    ok, err = validate_collection_indexed("demo", method="global")

    assert ok is True
    assert err is None


def test_validate_collection_indexed_reports_missing_required_artifacts(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")

    # relationships missing in tog mode
    ok, err = validate_collection_indexed("demo", method="tog")

    assert ok is False
    assert "relationships" in err
```

**Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/unit/test_helpers_indexed_validation.py -v
```

Expected: FAIL (function currently only checks local output directory files).

**Step 3: Implement mode-aware validation**

In `backend/app/utils/helpers.py`:
- Keep existing file-mode path checks unchanged.
- Add cosmos-mode branch:
  - load config via `load_graphrag_config(collection_id)`
  - instantiate storage from `config.output`
  - check required logical keys using `await storage.has("entities.parquet")` or `await storage.has("entities")` depending on storage conventions used by `CosmosDBPipelineStorage.has`.

Recommended consistent key set (matching current Cosmos storage prefix logic):
- `entities.parquet`
- `communities.parquet`
- `community_reports.parquet`
- plus method-specific `text_units.parquet`, `relationships.parquet`

Return same `(bool, error_message)` contract.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_helpers_indexed_validation.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/utils/helpers.py backend/tests/unit/test_helpers_indexed_validation.py
git commit -m "feat: validate indexed artifacts via storage abstraction in cosmos mode"
```

---

### Task 4: Refactor `get_search_data_paths` usage to avoid path-only assumptions

**Files:**
- Modify: `backend/app/utils/helpers.py`
- Modify: `backend/app/routers/search.py` (debug endpoint)
- Test: `backend/tests/unit/test_search_debug_endpoint_mode_aware.py`

**Step 1: Write failing tests for `/tog/debug` in cosmos mode**

```python
async def test_tog_debug_uses_storage_loader_in_cosmos_mode(client, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    # mock query table loader to return entities df
    resp = await client.get("/api/collections/demo/search/tog/debug")
    assert resp.status_code == 200
```

**Step 2: Run tests and confirm fail**

Run:

```bash
pytest backend/tests/unit/test_search_debug_endpoint_mode_aware.py -v
```

Expected: FAIL due to direct `pd.read_parquet(path)` call.

**Step 3: Implement minimal router change**

In `backend/app/routers/search.py`, ToG debug endpoint:
- replace direct path reads with service/helper loader call that works for both modes.
- easiest: add `query_service.load_entities_for_debug(collection_id)` returning DataFrame.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_search_debug_endpoint_mode_aware.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/routers/search.py backend/app/services/query_service.py backend/tests/unit/test_search_debug_endpoint_mode_aware.py

git commit -m "refactor: make tog debug endpoint storage-mode aware"
```

---

### Task 5: Add integration test for query API using storage abstraction in cosmos mode

**Files:**
- Create: `backend/tests/integration/test_query_cosmos_storage_abstraction.py`

**Step 1: Write failing integration test**

Flow:
1. `STORAGE_MODE=cosmos`, cosmos profile active.
2. Setup minimal indexed artifacts in storage (mock or fixture).
3. Call one endpoint (e.g. `/search/global`).
4. Assert response 200 and no filesystem/path errors.

**Step 2: Run integration test and confirm fail**

Run:

```bash
pytest backend/tests/integration/test_query_cosmos_storage_abstraction.py -v
```

Expected: FAIL before abstraction is fully wired.

**Step 3: Fix minimal defects**

Adjust only query loader/validation glue as needed.

**Step 4: Re-run integration test**

Run:

```bash
pytest backend/tests/integration/test_query_cosmos_storage_abstraction.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/integration/test_query_cosmos_storage_abstraction.py backend/app/services/query_service.py backend/app/utils/helpers.py

git commit -m "test: cover cosmos query path through storage abstraction"
```

---

### Task 6: Add cross-mode regression tests to ensure file mode still works

**Files:**
- Modify: `backend/tests/unit/test_query_service_storage_mode.py`
- Modify: `backend/tests/unit/test_helpers_indexed_validation.py`

**Step 1: Add file-mode tests**

Examples:
- query service in file mode still supports `get_search_data_paths` fallback (if retained) or storage abstraction over file output.
- validation in file mode still uses local file checks and error messages unchanged.

**Step 2: Run test bundle and verify pass**

Run:

```bash
pytest backend/tests/unit/test_query_service_storage_mode.py backend/tests/unit/test_helpers_indexed_validation.py -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add backend/tests/unit/test_query_service_storage_mode.py backend/tests/unit/test_helpers_indexed_validation.py
git commit -m "test: add file-mode regression coverage for query and validation abstractions"
```

---

### Task 7: Add verification commands to backend docs

**Files:**
- Modify: `backend/README.md`

**Step 1: Write failing doc check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Phase 4 verification' in t"
```

Expected: FAIL.

**Step 2: Add verification section**

Add command set:

```bash
pytest backend/tests/unit/test_query_service_storage_mode.py -v
pytest backend/tests/unit/test_helpers_indexed_validation.py -v
pytest backend/tests/unit/test_search_debug_endpoint_mode_aware.py -v
pytest backend/tests/integration/test_query_cosmos_storage_abstraction.py -v
```

**Step 3: Re-run doc check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Phase 4 verification' in t"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/README.md
git commit -m "docs: add phase 4 query and validation abstraction verification commands"
```

---

## Phase 4 Completion Checklist

- [ ] Query service no longer depends on direct local parquet paths.
- [ ] Query data tables load via GraphRAG storage abstraction in cosmos mode.
- [ ] Indexed-state validation uses storage checks in cosmos mode.
- [ ] ToG debug endpoint is mode-aware and path-agnostic.
- [ ] Cosmos-mode integration test passes for query endpoint.
- [ ] File-mode behavior remains intact through regression tests.

---

## Notes for Executor

- Keep abstraction local to backend query layer; do not introduce generic framework-wide repository pattern here.
- If `CosmosDBPipelineStorage.keys()` is unavailable, rely on deterministic required key checks via `has()`.
- Ensure async storage calls are awaited correctly to avoid false-positive validation outcomes.
