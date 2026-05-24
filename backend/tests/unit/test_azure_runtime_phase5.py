from backend.app.config import settings
from backend.app import azure_runtime


def _reset_runtime_caches() -> None:
    azure_runtime.bootstrap_runtime_secrets.cache_clear()
    azure_runtime.get_default_credential.cache_clear()
    azure_runtime._key_vault_client.cache_clear()


def test_is_cosmos_configured_with_managed_identity(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_cosmos_connection_string", "")
    monkeypatch.setattr(
        settings, "azure_cosmos_endpoint", "https://example.documents.azure.com:443/"
    )
    monkeypatch.setattr(settings, "azure_cosmos_key", "")
    monkeypatch.setattr(settings, "azure_use_managed_identity", True)

    assert azure_runtime.is_cosmos_configured() is True


def test_resolve_storage_connection_string_from_account_key(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_storage_connection_string", "")
    monkeypatch.setattr(settings, "azure_storage_account_name", "stexample")
    monkeypatch.setattr(settings, "azure_storage_account_key", "key123")

    conn = azure_runtime.resolve_storage_connection_string()

    assert "AccountName=stexample" in conn
    assert "AccountKey=key123" in conn


def test_resolve_cosmos_connection_string_from_endpoint_key(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(settings, "azure_cosmos_connection_string", "")
    monkeypatch.setattr(
        settings, "azure_cosmos_endpoint", "https://example.documents.azure.com:443/"
    )
    monkeypatch.setattr(settings, "azure_cosmos_key", "key123")

    conn = azure_runtime.resolve_cosmos_connection_string()

    assert (
        conn
        == "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;"
    )


def test_cosmos_account_url_from_connection_string(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_key_vault_url", "")
    monkeypatch.setattr(
        settings,
        "azure_cosmos_connection_string",
        "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;",
    )
    monkeypatch.setattr(settings, "azure_cosmos_endpoint", "")

    url = azure_runtime.cosmos_account_url()

    assert url == "https://example.documents.azure.com:443/"


def test_cosmos_client_kwargs_parses_retry_status_codes(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_cosmos_connection_timeout_seconds", 20)
    monkeypatch.setattr(settings, "azure_cosmos_retry_total", 11)
    monkeypatch.setattr(settings, "azure_cosmos_retry_backoff_max_seconds", 45)
    monkeypatch.setattr(settings, "azure_cosmos_retry_fixed_interval_ms", 500)
    monkeypatch.setattr(settings, "azure_cosmos_retry_connect", 5)
    monkeypatch.setattr(settings, "azure_cosmos_retry_read", 4)
    monkeypatch.setattr(settings, "azure_cosmos_retry_status", 8)
    monkeypatch.setattr(
        settings, "azure_cosmos_retry_on_status_codes", "429, 503, invalid"
    )

    kwargs = azure_runtime.cosmos_client_kwargs()

    assert kwargs["connection_timeout"] == 20
    assert kwargs["retry_total"] == 11
    assert kwargs["retry_backoff_max"] == 45
    assert kwargs["retry_fixed_interval"] == 500
    assert kwargs["retry_connect"] == 5
    assert kwargs["retry_read"] == 4
    assert kwargs["retry_status"] == 8
    assert kwargs["retry_on_status_codes"] == [429, 503]


def test_cosmos_client_kwargs_disables_endpoint_discovery_when_enabled(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_cosmos_disable_endpoint_discovery", True)

    kwargs = azure_runtime.cosmos_client_kwargs()

    assert kwargs["enable_endpoint_discovery"] is False
    assert kwargs["connection_mode"] == "Gateway"


def test_cosmos_client_kwargs_sets_connection_verify(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_cosmos_connection_verify", False)

    kwargs = azure_runtime.cosmos_client_kwargs()

    assert kwargs["connection_verify"] is False


def test_cosmos_client_kwargs_disables_http_logging_by_default(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_sdk_http_logging_enabled", False)

    kwargs = azure_runtime.cosmos_client_kwargs()

    assert kwargs["logging_enable"] is False


def test_cosmos_client_kwargs_enables_http_logging_when_configured(monkeypatch):
    _reset_runtime_caches()
    monkeypatch.setattr(settings, "azure_sdk_http_logging_enabled", True)

    kwargs = azure_runtime.cosmos_client_kwargs()

    assert kwargs["logging_enable"] is True
