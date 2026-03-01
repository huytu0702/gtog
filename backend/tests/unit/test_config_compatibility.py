from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from backend.app.main import app, lifespan
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
    async def _run_lifespan():
        async with lifespan(app):
            pass

    with patch("backend.app.main.validate_graphrag_settings_compatibility") as mock_check:
        anyio.run(_run_lifespan)

    mock_check.assert_called_once()


def test_backend_settings_yaml_passes_phase0_checkpoint():
    repo_root = Path(__file__).resolve().parents[3]
    settings_yaml = repo_root / "backend" / "settings.yaml"
    validate_graphrag_settings_compatibility(settings_yaml)
