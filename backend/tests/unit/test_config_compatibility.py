from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from backend.app.azure_runtime import bootstrap_runtime_secrets, get_default_credential
from backend.app.config import settings
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

COSMOS_SETTINGS = """
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
    type: cosmosdb
    url: https://example.documents.azure.com:443/
    database_name: gtog-control
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


def test_rejects_cloud_runtime_local_vector_store(tmp_path: Path):
    local_store = VALID_SETTINGS.replace(
        "type: azure_ai_search", "type: lancedb"
    ).replace("    url: https://example.search.windows.net\n", "")
    config_path = _write(tmp_path, local_store)

    with pytest.raises(ValueError, match="must use azure_ai_search or cosmosdb"):
        validate_graphrag_settings_compatibility(
            config_path,
            cloud_runtime=True,
            effective_store_type="lancedb",
        )


def test_accepts_cloud_runtime_azure_ai_search_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings, "azure_storage_connection_string", "UseDevelopmentStorage=true"
    )
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(
        settings, "azure_search_endpoint", "https://example.search.windows.net"
    )
    monkeypatch.setattr(settings, "azure_search_api_key", "search-key")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)

    config_path = _write(tmp_path, VALID_SETTINGS)

    validate_graphrag_settings_compatibility(
        config_path,
        cloud_runtime=True,
        effective_store_type="azure_ai_search",
    )


def test_rejects_cloud_runtime_missing_search_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings, "azure_storage_connection_string", "UseDevelopmentStorage=true"
    )
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(
        settings, "azure_search_endpoint", "https://example.search.windows.net"
    )
    monkeypatch.setattr(settings, "azure_search_api_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)

    config_path = _write(tmp_path, VALID_SETTINGS)

    with pytest.raises(
        ValueError, match="AZURE_SEARCH_API_KEY or Azure managed identity"
    ):
        validate_graphrag_settings_compatibility(
            config_path,
            cloud_runtime=True,
            effective_store_type="azure_ai_search",
        )


def test_accepts_cloud_runtime_managed_identity_search_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings, "azure_search_endpoint", "https://example.search.windows.net"
    )
    monkeypatch.setattr(settings, "azure_search_api_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", True)

    config_path = _write(tmp_path, VALID_SETTINGS)

    validate_graphrag_settings_compatibility(
        config_path,
        cloud_runtime=True,
        effective_store_type="azure_ai_search",
    )


def test_accepts_cloud_runtime_cosmosdb_with_connection_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings,
        "azure_cosmos_connection_string",
        "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;",
    )
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", "")
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")

    config_path = _write(tmp_path, COSMOS_SETTINGS)

    validate_graphrag_settings_compatibility(
        config_path,
        cloud_runtime=True,
        effective_store_type="cosmosdb",
    )


def test_rejects_cloud_runtime_missing_cosmos_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_cosmos_connection_string", "")
    monkeypatch.setattr(
        settings, "azure_cosmos_endpoint", "https://example.documents.azure.com:443/"
    )
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")

    config_path = _write(tmp_path, COSMOS_SETTINGS)

    with pytest.raises(
        ValueError,
        match="AZURE_COSMOS_CONNECTION_STRING, AZURE_COSMOS_KEY, or Azure managed identity",
    ):
        validate_graphrag_settings_compatibility(
            config_path,
            cloud_runtime=True,
            effective_store_type="cosmosdb",
        )


def test_startup_calls_compatibility_checkpoint():
    async def _run_lifespan():
        async with lifespan(app):
            pass

    with patch(
        "backend.app.main.validate_graphrag_settings_compatibility"
    ) as mock_check:
        anyio.run(_run_lifespan)

    mock_check.assert_called_once()


def test_backend_settings_yaml_passes_phase0_checkpoint():
    repo_root = Path(__file__).resolve().parents[3]
    settings_yaml = repo_root / "backend" / "settings.yaml"
    validate_graphrag_settings_compatibility(settings_yaml)
