# Phase 0 Production Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the backend for production by removing runtime prompt writes, disabling production debug behavior, cleaning duplicate exception logic, preserving temporary blob/parquet fallback, and enforcing GraphRAG config compatibility at startup.

**Architecture:** Phase 0 is backend hardening only: keep existing query/indexing behavior, add strict startup/config validation, and add regression tests before each fix. Behavior changes stay minimal and explicit (debug endpoint off by default, fallback path still active but clearly marked temporary). This creates a safe baseline for later Cosmos DB + Azure AI Search cutover phases.

**Tech Stack:** FastAPI, Pydantic Settings, PyYAML, GraphRAG config enums, pytest, pytest-asyncio, unittest.mock

---

## Execution Discipline (required)

- Process skill: `@superpowers:test-driven-development`
- Verification skill: `@superpowers:verification-before-completion`
- Review skill: `@superpowers:requesting-code-review`
- If unexpected failures appear: `@superpowers:systematic-debugging`
- Run in isolated workspace: `@superpowers:using-git-worktrees`

### Task 1: Add startup configuration compatibility checkpoint

**Files:**
- Create: `backend/app/utils/config_compatibility.py`
- Modify: `backend/app/utils/__init__.py:1-15`
- Modify: `backend/app/main.py:1-44`
- Test: `backend/tests/unit/test_config_compatibility.py`

**Step 1: Write the failing tests**

Create `backend/tests/unit/test_config_compatibility.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.utils.config_compatibility import (
    validate_graphrag_settings_compatibility,
)

VALID_SETTINGS = """
input:
  storage:
    type: blob
output:
  type: blob
cache:
  type: blob
reporting:
  type: blob
vector_store:
  default_vector_store:
    type: azure_ai_search
    url: https://example.search.windows.net
    embeddings_schema:
      entity.description:
        vector_size: 3072
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_rejects_legacy_index_schema(tmp_path: Path):
    legacy = VALID_SETTINGS.replace("embeddings_schema", "index_schema")
    config_path = _write(tmp_path, legacy)

    with pytest.raises(ValueError, match="embeddings_schema"):
        validate_graphrag_settings_compatibility(config_path)


def test_rejects_invalid_input_storage_type(tmp_path: Path):
    invalid = VALID_SETTINGS.replace("type: blob", "type: invalid_type", 1)
    config_path = _write(tmp_path, invalid)

    with pytest.raises(ValueError, match="input.storage.type"):
        validate_graphrag_settings_compatibility(config_path)


def test_startup_calls_compatibility_checkpoint():
    with patch("backend.app.main.validate_graphrag_settings_compatibility") as mock_check:
        with TestClient(app):
            pass

    mock_check.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_config_compatibility.py -v`
Expected: FAIL with missing module/function (`config_compatibility.py` not implemented yet).

**Step 3: Write minimal implementation**

Create `backend/app/utils/config_compatibility.py`:

```python
"""Phase 0 GraphRAG settings compatibility checkpoint."""

from __future__ import annotations

from pathlib import Path

import yaml

from graphrag.config.enums import CacheType, ReportingType, StorageType, VectorStoreType

PHASE0_COMPATIBILITY_CHECKS: tuple[str, ...] = (
    "vector_store.default_vector_store uses embeddings_schema (not index_schema)",
    "input/output/cache/reporting type values match GraphRAG enums",
    "azure_ai_search vector store has required url",
    "cosmosdb vector store has required url + database_name",
)


def _require_enum(value: str | None, allowed: set[str], key_path: str) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{key_path} must be one of [{allowed_values}], got {value!r}")


def validate_graphrag_settings_compatibility(settings_yaml_path: Path) -> None:
    data = yaml.safe_load(settings_yaml_path.read_text(encoding="utf-8")) or {}

    input_type = ((data.get("input") or {}).get("storage") or {}).get("type")
    output_type = (data.get("output") or {}).get("type")
    cache_type = (data.get("cache") or {}).get("type")
    reporting_type = (data.get("reporting") or {}).get("type")

    _require_enum(input_type, {e.value for e in StorageType}, "input.storage.type")
    _require_enum(output_type, {e.value for e in StorageType}, "output.type")
    _require_enum(cache_type, {e.value for e in CacheType}, "cache.type")
    _require_enum(reporting_type, {e.value for e in ReportingType}, "reporting.type")

    default_store = ((data.get("vector_store") or {}).get("default_vector_store") or {})

    if "index_schema" in default_store:
        raise ValueError(
            "vector_store.default_vector_store.index_schema is not supported; "
            "use embeddings_schema."
        )

    store_type = default_store.get("type")
    _require_enum(
        store_type,
        {e.value for e in VectorStoreType},
        "vector_store.default_vector_store.type",
    )

    if store_type == VectorStoreType.AzureAISearch.value and not default_store.get("url"):
        raise ValueError("vector_store.default_vector_store.url is required for azure_ai_search")

    if store_type == VectorStoreType.CosmosDB.value:
        if not default_store.get("url"):
            raise ValueError("vector_store.default_vector_store.url is required for cosmosdb")
        if not default_store.get("database_name"):
            raise ValueError(
                "vector_store.default_vector_store.database_name is required for cosmosdb"
            )
```

Modify `backend/app/utils/__init__.py`:

```python
from .config_compatibility import validate_graphrag_settings_compatibility
...
__all__ = [
    ...,
    "validate_graphrag_settings_compatibility",
]
```

Modify `backend/app/main.py` startup:

```python
from .utils.config_compatibility import validate_graphrag_settings_compatibility
...
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GraphRAG FastAPI backend...")
    validate_graphrag_settings_compatibility(settings.settings_yaml_path)
    ...
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_config_compatibility.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/utils/config_compatibility.py backend/app/utils/__init__.py backend/app/main.py backend/tests/unit/test_config_compatibility.py
git commit -m "feat(backend): add phase0 GraphRAG config compatibility checkpoint"
```

### Task 2: Migrate backend settings to `embeddings_schema` and lock with test

**Files:**
- Modify: `backend/settings.yaml:69-76`
- Modify: `backend/tests/unit/test_config_compatibility.py`

**Step 1: Write the failing test**

Append to `backend/tests/unit/test_config_compatibility.py`:

```python
def test_backend_settings_yaml_passes_phase0_checkpoint():
    repo_root = Path(__file__).resolve().parents[3]
    settings_yaml = repo_root / "backend" / "settings.yaml"
    validate_graphrag_settings_compatibility(settings_yaml)
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_config_compatibility.py::test_backend_settings_yaml_passes_phase0_checkpoint -v`
Expected: FAIL due legacy `index_schema` usage.

**Step 3: Write minimal implementation**

Update `backend/settings.yaml` vector store block:

```yaml
vector_store:
  default_vector_store:
    type: azure_ai_search
    url: ${AZURE_SEARCH_ENDPOINT}
    api_key: ${AZURE_SEARCH_API_KEY}
    embeddings_schema:
      entity.description:
        vector_size: 3072
      community.full_content:
        vector_size: 3072
      text_unit.text:
        vector_size: 3072
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_config_compatibility.py::test_backend_settings_yaml_passes_phase0_checkpoint -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/settings.yaml backend/tests/unit/test_config_compatibility.py
git commit -m "chore(config): migrate backend settings to embeddings_schema"
```

### Task 3: Remove runtime prompt file generation and fail fast on missing prompts

**Files:**
- Modify: `backend/app/utils/helpers.py:61-108,153-161`
- Test: `backend/tests/unit/test_prompt_validation.py`

**Step 1: Write the failing tests**

Create `backend/tests/unit/test_prompt_validation.py`:

```python
from pathlib import Path

import pytest

from backend.app.utils.helpers import _REQUIRED_PROMPT_FILES, _validate_shared_prompt_files


def test_validate_shared_prompt_files_raises_when_missing(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing required prompt files"):
        _validate_shared_prompt_files(prompt_dir)


def test_validate_shared_prompt_files_passes_when_complete(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    for filename in _REQUIRED_PROMPT_FILES:
        (prompt_dir / filename).write_text("placeholder", encoding="utf-8")

    _validate_shared_prompt_files(prompt_dir)
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_prompt_validation.py -v`
Expected: FAIL because `_REQUIRED_PROMPT_FILES` / `_validate_shared_prompt_files` are not implemented.

**Step 3: Write minimal implementation**

In `backend/app/utils/helpers.py`, replace runtime seeding with validation:

```python
_REQUIRED_PROMPT_FILES: tuple[str, ...] = (
    "extract_graph.txt",
    "summarize_descriptions.txt",
    "extract_claims.txt",
    "community_report_graph.txt",
    "community_report_text.txt",
    "local_search_system_prompt.txt",
    "global_search_map_system_prompt.txt",
    "global_search_reduce_system_prompt.txt",
    "global_search_knowledge_system_prompt.txt",
    "drift_search_system_prompt.txt",
    "drift_search_reduce_prompt.txt",
    "basic_search_system_prompt.txt",
    "question_gen_system_prompt.txt",
    "tog_relation_scoring_prompt.txt",
    "tog_entity_scoring_prompt.txt",
    "tog_reasoning_prompt.txt",
)


def _validate_shared_prompt_files(prompt_dir: Path) -> None:
    missing = [
        filename
        for filename in _REQUIRED_PROMPT_FILES
        if not (prompt_dir / filename).exists()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required prompt files in {prompt_dir}: {missing_list}"
        )
```

Update `load_graphrag_config` call site:

```python
_validate_shared_prompt_files(shared_root / "prompts")
```

(Remove runtime `mkdir` + `write_text` prompt generation path.)

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_prompt_validation.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/utils/helpers.py backend/tests/unit/test_prompt_validation.py
git commit -m "fix(backend): fail fast on missing prompt files and remove runtime prompt writes"
```

### Task 4: Disable ToG debug endpoint by default (production-safe)

**Files:**
- Modify: `backend/app/config.py:31-34`
- Modify: `backend/app/routers/search.py:1-126`
- Modify: `backend/tests/unit/test_search_router.py`

**Step 1: Write the failing tests**

Append to `backend/tests/unit/test_search_router.py`:

```python
import pandas as pd
from unittest.mock import patch


class TestToGDebugEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_tog_debug_returns_404_when_disabled(self, client):
        with patch("backend.app.routers.search.settings.enable_tog_debug_endpoint", False):
            response = client.get("/api/collections/test-collection/search/tog/debug")

        assert response.status_code == 404

    def test_tog_debug_returns_data_when_enabled(self, client):
        entities_df = pd.DataFrame(
            [{"title": "Entity A", "description": "Entity A description", "type": "org"}]
        )

        with patch("backend.app.routers.search.settings.enable_tog_debug_endpoint", True):
            with patch("backend.app.utils.get_search_data_paths", return_value={"entities": "entities.parquet"}):
                with patch("pandas.read_parquet", return_value=entities_df):
                    response = client.get("/api/collections/test-collection/search/tog/debug")

        assert response.status_code == 200
        body = response.json()
        assert body["total_entities"] == 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_search_router.py::TestToGDebugEndpoint::test_tog_debug_returns_404_when_disabled -v`
Expected: FAIL because endpoint is currently active without production gate.

**Step 3: Write minimal implementation**

Add setting in `backend/app/config.py`:

```python
enable_tog_debug_endpoint: bool = False
```

Add guard in `backend/app/routers/search.py`:

```python
from ..config import settings
...
@router.get("/tog/debug")
async def get_tog_entities(collection_id: str):
    if not settings.enable_tog_debug_endpoint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    ...
```

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_search_router.py::TestToGDebugEndpoint -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/config.py backend/app/routers/search.py backend/tests/unit/test_search_router.py
git commit -m "fix(api): disable ToG debug endpoint unless explicitly enabled"
```

### Task 5: Remove duplicate/unreachable exception handlers in collections router

**Files:**
- Modify: `backend/app/routers/collections.py:16-64`
- Test: `backend/tests/unit/test_collections_router_phase0.py`

**Step 1: Write the failing tests**

Create `backend/tests/unit/test_collections_router_phase0.py`:

```python
import inspect
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
import backend.app.routers.collections as collections_router


def test_create_collection_has_single_valueerror_and_exception_handler():
    source = inspect.getsource(collections_router.create_collection)
    assert source.count("except ValueError as e:") == 1
    assert source.count("except Exception as e:") == 1


def test_create_collection_value_error_response_shape_is_stable():
    client = TestClient(app)
    with patch(
        "backend.app.routers.collections.storage_service.create_collection",
        side_effect=ValueError("duplicate collection"),
    ):
        response = client.post(
            "/api/collections",
            json={"name": "valid_name", "description": "desc"},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "Validation failed"
    assert detail["field"] == "name"
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_collections_router_phase0.py::test_create_collection_has_single_valueerror_and_exception_handler -v`
Expected: FAIL because duplicate `except` blocks currently exist.

**Step 3: Write minimal implementation**

In `backend/app/routers/collections.py`, keep only one handler set under `create_collection`:

- Keep one `except ValueError as e` (422 structured detail)
- Keep one `except ValidationError as e` (422 structured detail)
- Keep one `except Exception as e` (500 structured detail)
- Remove duplicate unreachable handlers after the first `except Exception`

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_collections_router_phase0.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/routers/collections.py backend/tests/unit/test_collections_router_phase0.py
git commit -m "refactor(api): remove duplicate exception handlers in collections router"
```

### Task 6: Keep blob/parquet query path as explicit temporary fallback (with regression tests)

**Files:**
- Modify: `backend/app/services/query_service.py:36-42`
- Test: `backend/tests/unit/test_query_service_blob_fallback.py`

**Step 1: Write the failing tests**

Create `backend/tests/unit/test_query_service_blob_fallback.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.query_service import QueryService


@pytest.mark.asyncio
async def test_blob_fallback_logs_migration_warning_once():
    service = QueryService()
    frame = pd.DataFrame([{"id": 1, "value": "x"}])
    paths = {
        "entities": Path("entities.parquet"),
        "communities": Path("communities.parquet"),
        "community_reports": Path("community_reports.parquet"),
    }

    with patch("backend.app.services.query_service._BLOB_PARQUET_FALLBACK_WARNING_EMITTED", False):
        with patch("backend.app.services.query_service.validate_collection_indexed", return_value=(True, None)):
            with patch("backend.app.services.query_service.load_graphrag_config", return_value=MagicMock()):
                with patch("backend.app.services.query_service.get_search_data_paths", return_value=paths):
                    with patch("backend.app.services.query_service._blob_parquet", return_value=frame):
                        with patch(
                            "backend.app.services.query_service.api.global_search",
                            new=AsyncMock(return_value=("ok", {})),
                        ):
                            with patch(
                                "backend.app.services.query_service.settings.azure_storage_connection_string",
                                "UseDevelopmentStorage=true",
                            ):
                                with patch("backend.app.services.query_service.logger.warning") as mock_warning:
                                    await service.global_search("c1", "q1")
                                    await service.global_search("c1", "q2")

    assert mock_warning.call_count == 1
    assert "temporary blob/parquet fallback" in mock_warning.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_global_search_uses_local_parquet_when_blob_not_configured():
    service = QueryService()
    frame = pd.DataFrame([{"id": 1, "value": "x"}])
    paths = {
        "entities": Path("entities.parquet"),
        "communities": Path("communities.parquet"),
        "community_reports": Path("community_reports.parquet"),
    }

    with patch("backend.app.services.query_service.validate_collection_indexed", return_value=(True, None)):
        with patch("backend.app.services.query_service.load_graphrag_config", return_value=MagicMock()):
            with patch("backend.app.services.query_service.get_search_data_paths", return_value=paths):
                with patch("backend.app.services.query_service._blob_parquet") as mock_blob:
                    with patch("backend.app.services.query_service.pd.read_parquet", return_value=frame) as mock_read:
                        with patch(
                            "backend.app.services.query_service.api.global_search",
                            new=AsyncMock(return_value=("ok", {})),
                        ):
                            with patch(
                                "backend.app.services.query_service.settings.azure_storage_connection_string",
                                "",
                            ):
                                await service.global_search("c1", "q1")

    mock_blob.assert_not_called()
    assert mock_read.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_query_service_blob_fallback.py::test_blob_fallback_logs_migration_warning_once -v`
Expected: FAIL because one-time warning is not implemented.

**Step 3: Write minimal implementation**

In `backend/app/services/query_service.py`, add one-time warning in `_blob_parquet`:

```python
_BLOB_PARQUET_FALLBACK_WARNING_EMITTED = False


def _blob_parquet(collection_id: str, relative_path: Path) -> pd.DataFrame:
    global _BLOB_PARQUET_FALLBACK_WARNING_EMITTED
    if not _BLOB_PARQUET_FALLBACK_WARNING_EMITTED:
        logger.warning(
            "Using temporary blob/parquet fallback in query hot path; "
            "this remains only until Phase 3 cutover."
        )
        _BLOB_PARQUET_FALLBACK_WARNING_EMITTED = True
    ...
```

(Keep fallback behavior unchanged.)

**Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest backend/tests/unit/test_query_service_blob_fallback.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/services/query_service.py backend/tests/unit/test_query_service_blob_fallback.py
git commit -m "test(query): lock blob parquet fallback behavior and add temporary-use warning"
```

## Final verification (after Task 6)

Run:

```bash
.venv/Scripts/python -m pytest backend/tests/unit/test_config_compatibility.py -v
.venv/Scripts/python -m pytest backend/tests/unit/test_prompt_validation.py -v
.venv/Scripts/python -m pytest backend/tests/unit/test_search_router.py -v
.venv/Scripts/python -m pytest backend/tests/unit/test_collections_router_phase0.py -v
.venv/Scripts/python -m pytest backend/tests/unit/test_query_service_blob_fallback.py -v
.venv/Scripts/python -m pytest backend/tests/integration/test_agent_search.py -v
.venv/Scripts/python -m ruff check backend/app backend/tests
```

Expected:
- All targeted tests PASS
- Ruff check PASS
- Startup fails fast on incompatible settings
- No runtime prompt writes
- ToG debug behavior disabled by default
- Collections router duplicate exception block removed
- Blob/parquet query fallback preserved and explicitly marked temporary
