from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app.azure_runtime import bootstrap_runtime_secrets, get_default_credential
from backend.app.config import settings
from backend.app.utils.helpers import load_graphrag_config


def _reset_runtime_caches() -> None:
    bootstrap_runtime_secrets.cache_clear()
    get_default_credential.cache_clear()


@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_uses_local_lancedb_for_dev(mock_load_config, _mock_prompts, monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_storage_connection_string", "")
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(settings, "azure_search_endpoint", "")
    monkeypatch.setattr(settings, "azure_search_api_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1", query_runtime=False)
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert cli_overrides["vector_store.default_vector_store.container_name"] == "collection-a"
    assert "vector_store.default_vector_store.type" not in cli_overrides
    assert "vector_store.default_vector_store.url" not in cli_overrides


@patch("backend.app.utils.helpers._blob_client", return_value=object())
@patch("backend.app.utils.helpers._ensure_blob_container")
@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_uses_azure_ai_search_for_cloud_runtime(
    mock_load_config,
    _mock_prompts,
    _mock_container,
    _mock_blob_client,
    monkeypatch,
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_storage_connection_string", "UseDevelopmentStorage=true")
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(settings, "azure_search_endpoint", "https://example.search.windows.net")
    monkeypatch.setattr(settings, "azure_search_api_key", "search-key")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "query_context_mode", "cosmos_only")
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1", query_runtime=True)
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert cli_overrides["input.storage.connection_string"] == "UseDevelopmentStorage=true"
    assert cli_overrides["vector_store.default_vector_store.type"] == "azure_ai_search"
    assert cli_overrides["vector_store.default_vector_store.url"] == "https://example.search.windows.net"
    assert cli_overrides["vector_store.default_vector_store.api_key"] == "search-key"
    assert cli_overrides["vector_store.default_vector_store.db_uri"] is None
    assert cli_overrides["vector_store.default_vector_store.container_name"] == "collection-a"


@patch("backend.app.utils.helpers._blob_client", return_value=object())
@patch("backend.app.utils.helpers._ensure_blob_container")
@patch("backend.app.utils.helpers._validate_shared_prompt_files")
@patch("backend.app.utils.helpers.load_config")
def test_load_graphrag_config_omits_api_key_override_for_managed_identity(
    mock_load_config,
    _mock_prompts,
    _mock_container,
    _mock_blob_client,
    monkeypatch,
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_storage_connection_string", "")
    monkeypatch.setattr(settings, "azure_storage_account_url", "https://storage.example.blob.core.windows.net")
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(settings, "azure_search_endpoint", "https://example.search.windows.net")
    monkeypatch.setattr(settings, "azure_search_api_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", True)
    monkeypatch.setattr(settings, "query_context_mode", "cosmos_only")
    mock_load_config.return_value = MagicMock()

    load_graphrag_config("collection-a", version="v1", query_runtime=True)
    cli_overrides = mock_load_config.call_args.kwargs["cli_overrides"]

    assert (
        cli_overrides["input.storage.storage_account_blob_url"]
        == "https://storage.example.blob.core.windows.net"
    )
    assert (
        cli_overrides["output.storage_account_blob_url"]
        == "https://storage.example.blob.core.windows.net"
    )
    assert cli_overrides["vector_store.default_vector_store.type"] == "azure_ai_search"
    assert "vector_store.default_vector_store.api_key" not in cli_overrides


@patch("backend.app.utils.helpers._blob_client", return_value=object())
@patch("backend.app.utils.helpers._ensure_blob_container")
@patch("backend.app.utils.helpers._validate_shared_prompt_files")
def test_load_graphrag_config_rejects_cloud_runtime_without_search_endpoint(
    _mock_prompts,
    _mock_container,
    _mock_blob_client,
    monkeypatch,
):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_storage_connection_string", "UseDevelopmentStorage=true")
    monkeypatch.setattr(settings, "azure_storage_account_name", "")
    monkeypatch.setattr(settings, "azure_storage_account_key", "")
    monkeypatch.setattr(settings, "azure_search_endpoint", "")
    monkeypatch.setattr(settings, "azure_search_api_key", "search-key")
    monkeypatch.setattr(settings, "azure_use_managed_identity", False)
    monkeypatch.setattr(settings, "query_context_mode", "cosmos_only")

    try:
        load_graphrag_config("collection-a", version="v1", query_runtime=True)
    except ValueError as exc:
        assert "AZURE_SEARCH_ENDPOINT" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing Azure Search endpoint")
