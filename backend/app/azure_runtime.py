"""Runtime Azure auth/secret helpers for production profiles."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

_SECRET_BINDINGS: tuple[tuple[str, str, str], ...] = (
    (
        "graphrag_api_key",
        "azure_key_vault_graphrag_api_key_secret_name",
        "GRAPHRAG_API_KEY",
    ),
    ("openai_api_key", "azure_key_vault_openai_api_key_secret_name", "OPENAI_API_KEY"),
    ("google_api_key", "azure_key_vault_google_api_key_secret_name", "GOOGLE_API_KEY"),
    ("tavily_api_key", "azure_key_vault_tavily_api_key_secret_name", "TAVILY_API_KEY"),
    (
        "azure_storage_connection_string",
        "azure_key_vault_storage_connection_string_secret_name",
        "AZURE_STORAGE_CONNECTION_STRING",
    ),
    (
        "azure_storage_account_key",
        "azure_key_vault_storage_account_key_secret_name",
        "AZURE_STORAGE_ACCOUNT_KEY",
    ),
    (
        "azure_search_api_key",
        "azure_key_vault_search_api_key_secret_name",
        "AZURE_SEARCH_API_KEY",
    ),
    (
        "azure_cosmos_connection_string",
        "azure_key_vault_cosmos_connection_string_secret_name",
        "AZURE_COSMOS_CONNECTION_STRING",
    ),
    ("azure_cosmos_key", "azure_key_vault_cosmos_key_secret_name", "AZURE_COSMOS_KEY"),
)


_QUEUE_ENDPOINT_SUFFIX = ".queue.core.windows.net"


def _set_setting(setting_attr: str, env_name: str, value: str) -> None:
    setattr(settings, setting_attr, value)
    os.environ[env_name] = value


@lru_cache(maxsize=1)
def get_default_credential():
    """Return a reusable DefaultAzureCredential for MI/CLI auth flows."""
    from azure.identity import DefaultAzureCredential

    kwargs: dict[str, Any] = {}
    if settings.azure_managed_identity_client_id:
        kwargs["managed_identity_client_id"] = settings.azure_managed_identity_client_id
    return DefaultAzureCredential(**kwargs)


@lru_cache(maxsize=1)
def _key_vault_client():
    """Return cached Key Vault secret client when a vault URL is configured."""
    if not settings.azure_key_vault_url:
        return None
    try:
        from azure.keyvault.secrets import SecretClient
    except ImportError as exc:
        raise RuntimeError(
            "Key Vault integration requires azure-keyvault-secrets. "
            "Install dependencies with `uv sync`."
        ) from exc
    return SecretClient(
        vault_url=settings.azure_key_vault_url, credential=get_default_credential()
    )


def _fetch_secret(secret_name: str) -> str:
    client = _key_vault_client()
    if client is None:
        raise RuntimeError("AZURE_KEY_VAULT_URL is not configured.")
    secret = client.get_secret(secret_name)
    value = secret.value or ""
    if not value:
        raise RuntimeError(f"Key Vault secret '{secret_name}' is empty.")
    return value


@lru_cache(maxsize=1)
def bootstrap_runtime_secrets() -> None:
    """
    Hydrate runtime env/settings from Key Vault when secret names are configured.

    Existing explicit env values always win over Key Vault.
    """
    if not settings.azure_key_vault_url:
        return

    for setting_attr, secret_attr, env_name in _SECRET_BINDINGS:
        if getattr(settings, setting_attr):
            continue
        secret_name = str(getattr(settings, secret_attr, "") or "").strip()
        if not secret_name:
            continue
        value = _fetch_secret(secret_name)
        _set_setting(setting_attr, env_name, value)
        logger.info("Loaded %s from Key Vault secret '%s'", env_name, secret_name)


def is_managed_identity_enabled() -> bool:
    return bool(settings.azure_use_managed_identity)


def is_cosmos_configured() -> bool:
    """Return True when Cosmos can authenticate via key/connection string/managed identity."""
    bootstrap_runtime_secrets()
    if settings.azure_cosmos_connection_string:
        return True
    if settings.azure_cosmos_endpoint and settings.azure_cosmos_key:
        return True
    return bool(settings.azure_cosmos_endpoint and is_managed_identity_enabled())


def cosmos_endpoint_credential():
    """Return credential object to use with Cosmos endpoint auth."""
    if settings.azure_cosmos_key:
        return settings.azure_cosmos_key
    if settings.azure_cosmos_endpoint and is_managed_identity_enabled():
        return get_default_credential()
    return None


def _parse_status_code_csv(raw_value: str) -> list[int]:
    values = []
    for part in raw_value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError:
            logger.warning("Ignoring invalid Cosmos retry status code value: %r", text)
    return values


def cosmos_client_kwargs() -> dict[str, Any]:
    """Build CosmosClient retry/timeout kwargs from runtime settings."""
    kwargs: dict[str, Any] = {
        "connection_timeout": settings.azure_cosmos_connection_timeout_seconds,
        "retry_total": settings.azure_cosmos_retry_total,
        "retry_backoff_max": settings.azure_cosmos_retry_backoff_max_seconds,
        "retry_fixed_interval": settings.azure_cosmos_retry_fixed_interval_ms,
        "retry_connect": settings.azure_cosmos_retry_connect,
        "retry_read": settings.azure_cosmos_retry_read,
        "retry_status": settings.azure_cosmos_retry_status,
    }
    status_codes = _parse_status_code_csv(settings.azure_cosmos_retry_on_status_codes)
    if status_codes:
        kwargs["retry_on_status_codes"] = status_codes
    return kwargs


def resolve_storage_connection_string() -> str:
    """Resolve storage connection string from direct value or account key fallback."""
    bootstrap_runtime_secrets()
    if settings.azure_storage_connection_string:
        return settings.azure_storage_connection_string
    if settings.azure_storage_account_name and settings.azure_storage_account_key:
        return (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={settings.azure_storage_account_name};"
            f"AccountKey={settings.azure_storage_account_key};"
            "EndpointSuffix=core.windows.net"
        )
    return ""


def resolve_cosmos_connection_string() -> str:
    """Resolve Cosmos connection string from direct value or endpoint/key fallback."""
    bootstrap_runtime_secrets()
    if settings.azure_cosmos_connection_string:
        return settings.azure_cosmos_connection_string
    if settings.azure_cosmos_endpoint and settings.azure_cosmos_key:
        endpoint = settings.azure_cosmos_endpoint.rstrip("/") + "/"
        return f"AccountEndpoint={endpoint};AccountKey={settings.azure_cosmos_key};"
    return ""


def cosmos_account_url() -> str:
    """Resolve Cosmos account URL from direct endpoint or connection string."""
    bootstrap_runtime_secrets()
    if settings.azure_cosmos_endpoint:
        return settings.azure_cosmos_endpoint.rstrip("/") + "/"

    connection_string = settings.azure_cosmos_connection_string
    if not connection_string:
        return ""

    for segment in connection_string.split(";"):
        key, _, value = segment.partition("=")
        if key.strip().lower() == "accountendpoint" and value.strip():
            return value.strip().rstrip("/") + "/"
    return ""


def blob_account_url() -> str:
    """Resolve blob account URL for managed identity auth."""
    if settings.azure_storage_account_url:
        return settings.azure_storage_account_url.rstrip("/")
    if settings.azure_storage_account_name:
        return f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
    return ""


def queue_account_url() -> str:
    """Resolve queue account URL for managed identity auth."""
    if settings.azure_storage_account_url:
        base_url = settings.azure_storage_account_url.rstrip("/")
        if base_url.endswith(".blob.core.windows.net"):
            account_name = base_url.removeprefix("https://").removesuffix(
                ".blob.core.windows.net"
            )
            return f"https://{account_name}{_QUEUE_ENDPOINT_SUFFIX}"
        return base_url
    if settings.azure_storage_account_name:
        return f"https://{settings.azure_storage_account_name}{_QUEUE_ENDPOINT_SUFFIX}"
    return ""


def create_blob_service_client():
    """Create BlobServiceClient using connection string or managed identity."""
    from azure.storage.blob import BlobServiceClient

    connection_string = resolve_storage_connection_string()
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    if is_managed_identity_enabled():
        account_url = blob_account_url()
        if account_url:
            return BlobServiceClient(
                account_url=account_url, credential=get_default_credential()
            )
    return None


def create_queue_service_client():
    """Create QueueServiceClient using connection string or managed identity."""
    from azure.storage.queue import QueueServiceClient

    connection_string = resolve_storage_connection_string()
    if connection_string:
        return QueueServiceClient.from_connection_string(connection_string)

    if is_managed_identity_enabled():
        account_url = queue_account_url()
        if account_url:
            return QueueServiceClient(
                account_url=account_url, credential=get_default_credential()
            )
    return None
