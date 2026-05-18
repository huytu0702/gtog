from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.azure_runtime import bootstrap_runtime_secrets, get_default_credential
from backend.app.config import settings
from backend.app.utils.helpers import load_graphrag_config


def _reset_runtime_caches() -> None:
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()


@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_uses_cosmos_pipeline_output_mode(
    mock_load_config, _mock_prompts, monkeypatch
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings,
        "azure_cosmos_connection_string",
        "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;",
    )
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", "")
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1")
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert cli_overrides["output.type"] == "cosmosdb"
    assert cli_overrides["output.base_dir"] == "gtog-control"
    assert str(cli_overrides["output.container_name"]).startswith("pipeline-")
    assert "output.connection_string" in cli_overrides


@patch("backend.app.utils.helpers._validate_shared_prompt_files")
def test_load_graphrag_config_rejects_cosmos_pipeline_without_cosmos_auth(
    _mock_prompts, monkeypatch
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_cosmos_connection_string", "")
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", "")
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)

    with patch("backend.app.utils.helpers.load_config", return_value=MagicMock()):
        with pytest.raises(ValueError, match="AZURE_COSMOS"):
            load_graphrag_config("collection-a", version="v1")


@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_uses_azure_ai_search_for_cloud_runtime(
    mock_load_config,
    _mock_prompts,
    monkeypatch,
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings, "azure_cosmos_connection_string", "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;"
    )
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", "")
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")
    monkeypatch.setattr(
        settings, "azure_search_endpoint", "https://example.search.windows.net"
    )
    monkeypatch.setattr(settings, "azure_search_api_key", "search-key")
    monkeypatch.setattr(settings, "cloud_vector_store_type", "azure_ai_search")
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1", use_cloud_vectors=True)
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert cli_overrides["vector_store.default_vector_store.type"] == "azure_ai_search"
    assert (
        cli_overrides["vector_store.default_vector_store.url"]
        == "https://example.search.windows.net"
    )
    assert cli_overrides["vector_store.default_vector_store.api_key"] == "search-key"


@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_uses_cosmos_vector_store_for_cloud_runtime(
    mock_load_config,
    _mock_prompts,
    monkeypatch,
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_cosmos_connection_string", "")
    monkeypatch.setattr(
        settings, "azure_cosmos_endpoint", "https://example.documents.azure.com:443/"
    )
    monkeypatch.setattr(settings, "azure_cosmos_key", "cosmos-key")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "azure_cosmos_database_name", "gtog-control")
    monkeypatch.setattr(settings, "cloud_vector_store_type", "cosmosdb")
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1", use_cloud_vectors=True)
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert cli_overrides["vector_store.default_vector_store.type"] == "cosmosdb"
    assert (
        cli_overrides["vector_store.default_vector_store.url"]
        == "https://example.documents.azure.com:443/"
    )
    assert (
        cli_overrides["vector_store.default_vector_store.connection_string"]
        == "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=cosmos-key;"
    )
