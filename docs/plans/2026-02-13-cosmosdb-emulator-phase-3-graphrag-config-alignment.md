# CosmosDB Emulator Phase 3 (GraphRAG Config Alignment) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align backend GraphRAG runtime configuration so indexing and query pipelines use Cosmos profile values without hidden file-forcing overrides.

**Architecture:** Make `load_graphrag_config` fully profile-driven in cosmos mode and validate that `api.build_index(...)` receives a `GraphRagConfig` with cosmos output/cache/vector store settings. Keep file mode backward-compatible by preserving explicit file overrides only when `STORAGE_MODE=file`.

**Tech Stack:** GraphRAG config model loading, FastAPI backend utils/services, pytest unit/integration tests.

---

### Task 1: Add regression tests proving current forced-file override behavior is removed in cosmos mode

**Files:**
- Modify/Create: `backend/tests/unit/test_helpers_config_overrides.py`
- Test: `pytest backend/tests/unit/test_helpers_config_overrides.py -v`

**Step 1: Write failing tests**

```python
from backend.app.utils.helpers import load_graphrag_config


def test_cosmos_mode_preserves_profile_output_cache_vector(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    monkeypatch.setenv("GRAPHRAG_SETTINGS_FILE", "settings.cosmos-emulator.yaml")

    cfg = load_graphrag_config("demo")

    assert cfg.output.type == "cosmosdb"
    assert cfg.cache.type == "cosmosdb"
    assert cfg.vector_store["default_vector_store"].type == "cosmosdb"


def test_file_mode_still_uses_file_overrides(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "file")
    monkeypatch.setenv("GRAPHRAG_SETTINGS_FILE", "settings.yaml")

    cfg = load_graphrag_config("demo")

    assert cfg.output.type == "file"
    assert cfg.cache.type == "file"
```

**Step 2: Run tests to confirm fail**

Run:

```bash
pytest backend/tests/unit/test_helpers_config_overrides.py -v
```

Expected: FAIL if cosmos mode still forced to file.

**Step 3: Implement minimal helper fix**

In `backend/app/utils/helpers.py`:
- Ensure cosmos mode path does not inject file overrides at all.
- Keep file-mode `cli_overrides` exactly as needed for existing behavior.

Pseudo-structure:

```python
if settings.is_cosmos_mode:
    config = load_config(root_dir=str(collection_dir), config_filepath=settings.settings_yaml_path)
else:
    config = load_config(..., cli_overrides={...file overrides...})
```

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_helpers_config_overrides.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/utils/helpers.py backend/tests/unit/test_helpers_config_overrides.py
git commit -m "fix: preserve cosmos profile settings in graphrag config loading"
```

---

### Task 2: Validate indexing service consumes cosmos-aligned config

**Files:**
- Modify/Create: `backend/tests/unit/test_indexing_service_config_mode.py`
- Modify (if needed): `backend/app/services/indexing_service.py`

**Step 1: Write failing unit test around config passed to `api.build_index`**

```python
from unittest.mock import AsyncMock, patch
from backend.app.services.indexing_service import IndexingService


@patch("backend.app.services.indexing_service.api.build_index", new_callable=AsyncMock)
@patch("backend.app.services.indexing_service.load_graphrag_config")
async def test_indexing_uses_cosmos_profile_config(mock_load, mock_build, monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")

    fake_cfg = ...
    fake_cfg.output.type = "cosmosdb"
    fake_cfg.cache.type = "cosmosdb"
    mock_load.return_value = fake_cfg

    svc = IndexingService()
    await svc._run_indexing_task("demo")

    mock_build.assert_awaited_once()
    passed_cfg = mock_build.await_args.kwargs["config"]
    assert passed_cfg.output.type == "cosmosdb"
```

**Step 2: Run test and confirm failure (if any)**

Run:

```bash
pytest backend/tests/unit/test_indexing_service_config_mode.py -v
```

Expected: FAIL if indexing path mutates config away from cosmos.

**Step 3: Implement minimal correction**

Only if needed:
- Remove/avoid any local reassignment of output/cache/vector types in `indexing_service.py`.
- Keep logic profile-driven via helper.

**Step 4: Re-run test**

Run:

```bash
pytest backend/tests/unit/test_indexing_service_config_mode.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/unit/test_indexing_service_config_mode.py backend/app/services/indexing_service.py
git commit -m "test: verify indexing pipeline receives cosmos-aligned graphrag config"
```

---

### Task 3: Add config smoke test for `settings.cosmos-emulator.yaml` compatibility

**Files:**
- Create: `backend/tests/unit/test_cosmos_profile_load.py`
- Test: unit pytest command

**Step 1: Write failing smoke test**

```python
from graphrag.config.load_config import load_config


def test_cosmos_profile_loads_with_env(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081")
    monkeypatch.setenv("COSMOS_KEY", "test-key")
    monkeypatch.setenv("COSMOS_DATABASE", "gtog")
    monkeypatch.setenv("COSMOS_CONTAINER", "graphrag")

    cfg = load_config(root_dir=str(tmp_path), config_filepath="backend/settings.cosmos-emulator.yaml")
    assert cfg.output.type == "cosmosdb"
```

**Step 2: Run test and confirm fail**

Run:

```bash
pytest backend/tests/unit/test_cosmos_profile_load.py -v
```

Expected: FAIL initially if profile has schema/key mismatch.

**Step 3: Fix profile schema issues minimally**

Adjust `backend/settings.cosmos-emulator.yaml` to match GraphRAG model fields exactly (examples from existing codebase):
- correct vector store schema key names (e.g. `vector_store_id`, `database_name`, `container_name`, `overwrite`, `text_key`, etc. depending on actual model).
- ensure env substitutions resolve.

**Step 4: Re-run smoke test**

Run:

```bash
pytest backend/tests/unit/test_cosmos_profile_load.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/settings.cosmos-emulator.yaml backend/tests/unit/test_cosmos_profile_load.py
git commit -m "fix: align cosmos emulator profile with graphrag config schema"
```

---

### Task 4: Add startup validation for required cosmos settings in cosmos mode

**Files:**
- Modify: `backend/app/main.py`
- Modify/Create: `backend/tests/unit/test_startup_cosmos_validation.py`

**Step 1: Write failing tests for fail-fast startup behavior**

```python
from backend.app.main import _validate_startup_configuration


def test_cosmos_mode_requires_endpoint_key_database_container(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    monkeypatch.delenv("COSMOS_ENDPOINT", raising=False)
    with pytest.raises(ValueError):
        _validate_startup_configuration()
```

**Step 2: Run tests and verify fail**

Run:

```bash
pytest backend/tests/unit/test_startup_cosmos_validation.py -v
```

Expected: FAIL (`_validate_startup_configuration` missing).

**Step 3: Implement minimal startup validator**

In `backend/app/main.py`:
- add private function `_validate_startup_configuration()`.
- called during lifespan startup before service is ready.
- if `settings.is_cosmos_mode`, assert non-empty values for:
  - `cosmos_endpoint`
  - `cosmos_key`
  - `cosmos_database`
  - `cosmos_container`
  - optionally `settings_yaml_path` exists.

Raise `ValueError` with clear message.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_startup_cosmos_validation.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/unit/test_startup_cosmos_validation.py
git commit -m "feat: add fail-fast startup validation for cosmos mode configuration"
```

---

### Task 5: Add backend API smoke integration test under cosmos profile (index start path)

**Files:**
- Create: `backend/tests/integration/test_indexing_cosmos_profile.py`

**Step 1: Write failing integration test**

Scenario:
1. Configure app in `STORAGE_MODE=cosmos` with profile file.
2. Create collection.
3. Upload one valid `.txt` doc.
4. POST `/api/collections/{id}/index`.
5. Assert `202` and status `running|completed` without config-related exception.

**Step 2: Run test to verify fail**

Run:

```bash
pytest backend/tests/integration/test_indexing_cosmos_profile.py -v
```

Expected: FAIL initially if profile/mode wiring incomplete.

**Step 3: Implement minimal fixes only**

Fix whichever layer fails:
- config wiring,
- startup validation messaging,
- helper config loading path.

Do not modify API schema.

**Step 4: Re-run integration test**

Run:

```bash
pytest backend/tests/integration/test_indexing_cosmos_profile.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/integration/test_indexing_cosmos_profile.py backend/app/utils/helpers.py backend/app/main.py
git commit -m "test: add cosmos profile integration coverage for indexing startup path"
```

---

### Task 6: Add phase-level verification command bundle

**Files:**
- Modify: `backend/README.md` (verification section)

**Step 1: Write failing doc check (section absent)**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Cosmos profile verification' in t"
```

Expected: FAIL.

**Step 2: Add verification section**

Add commands exactly:

```bash
pytest backend/tests/unit/test_helpers_config_overrides.py -v
pytest backend/tests/unit/test_indexing_service_config_mode.py -v
pytest backend/tests/unit/test_cosmos_profile_load.py -v
pytest backend/tests/unit/test_startup_cosmos_validation.py -v
pytest backend/tests/integration/test_indexing_cosmos_profile.py -v
```

**Step 3: Re-run doc check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Cosmos profile verification' in t"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/README.md
git commit -m "docs: add cosmos profile verification checklist"
```

---

## Phase 3 Completion Checklist

- [ ] Cosmos mode no longer receives hidden forced-file overrides.
- [ ] `settings.cosmos-emulator.yaml` loads cleanly against GraphRAG config models.
- [ ] Indexing service uses cosmos-aligned config from helper/profile.
- [ ] Startup fails fast when required cosmos env settings are missing.
- [ ] Integration test validates index-start route under cosmos profile.

---

## Notes for Executor

- Keep changes tightly scoped to config alignment; do not migrate query parquet reads in this phase.
- If a test requires heavy GraphRAG runtime, mock minimal boundaries to stay fast.
- Preserve existing file-mode defaults and behavior.
