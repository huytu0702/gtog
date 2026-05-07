"""Application configuration using Pydantic Settings."""

import logging
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE_PATH = _BACKEND_DIR / ".env"

_config_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    graphrag_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # Storage Configuration
    storage_root_dir: str = "./storage"
    azure_storage_connection_string: str = ""
    azure_storage_account_name: str = ""
    azure_storage_account_key: str = ""
    azure_storage_account_url: str = ""
    azure_storage_queue_name: str = "indexing-jobs"
    azure_storage_queue_visibility_timeout_seconds: int = 300
    azure_storage_queue_poll_interval_seconds: int = 5
    azure_storage_queue_dequeue_batch_size: int = 4
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_use_managed_identity: bool = False
    azure_managed_identity_client_id: str = ""
    azure_key_vault_url: str = ""
    azure_key_vault_graphrag_api_key_secret_name: str = ""
    azure_key_vault_openai_api_key_secret_name: str = ""
    azure_key_vault_google_api_key_secret_name: str = ""
    azure_key_vault_tavily_api_key_secret_name: str = ""
    azure_key_vault_storage_connection_string_secret_name: str = ""
    azure_key_vault_storage_account_key_secret_name: str = ""
    azure_key_vault_search_api_key_secret_name: str = ""
    azure_key_vault_cosmos_connection_string_secret_name: str = ""
    azure_key_vault_cosmos_key_secret_name: str = ""
    azure_cosmos_connection_string: str = ""
    azure_cosmos_endpoint: str = ""
    azure_cosmos_key: str = ""
    azure_cosmos_connection_timeout_seconds: int = 15
    azure_cosmos_retry_total: int = 9
    azure_cosmos_retry_backoff_max_seconds: int = 30
    azure_cosmos_retry_fixed_interval_ms: int = 0
    azure_cosmos_retry_connect: int = 3
    azure_cosmos_retry_read: int = 3
    azure_cosmos_retry_status: int = 9
    azure_cosmos_retry_on_status_codes: str = "429,503,408"
    azure_cosmos_database_name: str = "gtog-control"
    azure_cosmos_collections_container: str = "collections"
    azure_cosmos_documents_container: str = "documents"
    azure_cosmos_indexing_jobs_container: str = "indexingJobs"
    azure_cosmos_job_events_container: str = "jobEvents"
    azure_cosmos_artifact_manifest_container: str = "artifactManifest"
    azure_cosmos_entities_container: str = "entities"
    azure_cosmos_relationships_container: str = "relationships"
    azure_cosmos_text_units_container: str = "textUnits"
    azure_cosmos_communities_container: str = "communities"
    azure_cosmos_community_reports_container: str = "communityReports"
    azure_cosmos_covariates_container: str = "covariates"
    azure_cosmos_conversation_sessions_container: str = "conversationSessions"
    azure_cosmos_conversation_turns_container: str = "conversationTurns"

    # Indexing worker configuration
    indexing_job_max_attempts: int = 3
    indexing_worker_lease_duration_seconds: int = 300
    indexing_worker_heartbeat_interval_seconds: int = 30
    indexing_worker_recovery_interval_seconds: int = 30

    # Query serving mode
    query_context_mode: str = "cosmos_only"
    cloud_vector_store_type: str = "cosmosdb"
    serving_dataset_cache_max_entries: int = 96
    serving_cache_warm_on_index_complete: bool = True

    # Conversation memory
    conversation_legacy_payload_enabled: bool = True
    conversation_turn_ttl_days: int = 30
    conversation_session_ttl_days: int = 90
    conversation_summarize_user_turn_threshold: int = 8
    conversation_recent_user_turns: int = 3
    conversation_turn_max_chars: int = 4000
    conversation_summary_max_chars: int = 2000

    # Model Configuration
    default_chat_model: str = ""
    default_embedding_model: str = ""
    insufficiency_judge_enabled: bool = True
    insufficiency_judge_model: str = ""
    insufficiency_judge_timeout_seconds: int = 4
    insufficiency_judge_max_tokens: int = 250
    insufficiency_judge_temperature: float = 0.0
    insufficiency_judge_min_confidence: float = 0.5
    insufficiency_judge_max_response_chars: int = 6000
    web_fallback_enabled: bool = True

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    enable_tog_debug_endpoint: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    edge_origin_secret: str = Field(
        default="",
        validation_alias=AliasChoices("EDGE_ORIGIN_SECRET", "AFD_ORIGIN_SECRET"),
    )
    require_edge_auth: bool = False
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120
    rate_limiter_backend: str = Field(
        default="memory",
        description="Rate limiter backend: 'memory' (process-local) or 'redis' (distributed).",
    )

    # LRU cache tunables
    cache_ttl_seconds: int = Field(
        default=1800,
        description="TTL for LRU cache entries in seconds.",
    )
    cache_max_size: int = Field(
        default=50,
        description="Maximum number of LRU cache entries.",
    )

    @model_validator(mode="after")
    def _warn_missing_recommended_settings(self) -> "Settings":
        """Log warnings for any missing recommended production settings."""
        if not self.azure_cosmos_endpoint and not self.azure_cosmos_connection_string:
            _config_logger.warning(
                "AZURE_COSMOS_ENDPOINT (or AZURE_COSMOS_CONNECTION_STRING) is not set. "
                "Cosmos DB features will be unavailable."
            )
        if not self.azure_storage_connection_string and not self.azure_storage_account_name:
            _config_logger.warning(
                "AZURE_STORAGE_CONNECTION_STRING (or AZURE_STORAGE_ACCOUNT_NAME) is not set. "
                "Azure Blob Storage will be unavailable; falling back to local filesystem."
            )
        if not self.edge_origin_secret and self.require_edge_auth:
            _config_logger.warning(
                "EDGE_ORIGIN_SECRET is not set but REQUIRE_EDGE_AUTH=true. "
                "The application will fail to start."
            )
        if self.rate_limiter_backend == "memory":
            _config_logger.warning(
                "RATE_LIMITER_BACKEND=memory (default). "
                "Rate limiting is process-local and will NOT be enforced across multiple "
                "container instances. Set RATE_LIMITER_BACKEND=redis for distributed limiting."
            )
        return self

    @property
    def collections_dir(self) -> Path:
        """Get the collections directory path, resolved relative to backend dir."""
        backend_dir = Path(__file__).parent.parent
        p = Path(self.storage_root_dir)
        if not p.is_absolute():
            p = backend_dir / p
        return p / "collections"

    @property
    def settings_yaml_path(self) -> Path:
        """Get the shared settings.yaml path."""
        return Path(__file__).parent.parent / "settings.yaml"

    @cached_property
    def _models_config(self) -> dict:
        """Load models section from backend settings.yaml."""
        try:
            data = yaml.safe_load(self.settings_yaml_path.read_text(encoding="utf-8")) or {}
            return (data.get("models") or {})
        except Exception as exc:
            _config_logger.warning("Failed to read models config from settings.yaml: %s", exc)
            return {}

    def _resolve_model_config(self, model_id: str, fallback_model: str, fallback_provider: str) -> tuple[str, str]:
        model_config = (self._models_config.get(model_id) or {})
        model = str(model_config.get("model") or "").strip()
        provider = str(model_config.get("model_provider") or "").strip().lower()
        if model and provider:
            return model, provider
        _config_logger.warning(
            "models.%s is missing model/provider in settings.yaml; using defaults.",
            model_id,
        )
        return fallback_model, fallback_provider

    @cached_property
    def _query_chat_model_config(self) -> tuple[str, str]:
        """Resolve query chat model/provider from backend settings.yaml."""
        return self._resolve_model_config(
            "query_chat_model",
            fallback_model="gpt-5.4-mini",
            fallback_provider="openai",
        )

    @cached_property
    def _default_chat_model_config(self) -> tuple[str, str]:
        """Resolve default chat model/provider from backend settings.yaml."""
        return self._resolve_model_config(
            "default_chat_model",
            fallback_model="gpt-5.2",
            fallback_provider="openai",
        )

    @property
    def query_chat_model(self) -> str:
        """Return model name configured for query-time chat calls."""
        return self._query_chat_model_config[0]

    @property
    def query_chat_model_provider(self) -> str:
        """Return provider configured for query-time chat model."""
        return self._query_chat_model_config[1]

    @property
    def query_chat_model_litellm(self) -> str:
        """Return LiteLLM model identifier for query-time chat calls."""
        model = self.query_chat_model
        provider = self.query_chat_model_provider
        if "/" in model:
            return model
        return f"{provider}/{model}" if provider else model

    def api_key_for_provider(self, provider: str) -> str:
        """Return API key for known providers; empty string for unknown providers."""
        normalized = provider.strip().lower()
        if normalized in {"openai", "azure_openai"}:
            return self.openai_api_key
        if normalized in {"gemini", "google"}:
            return self.google_api_key or self.graphrag_api_key
        _config_logger.warning("No API key mapping for provider: %s", provider)
        return ""

    @property
    def query_chat_model_api_key(self) -> str:
        """Return API key mapped to query-time chat model provider."""
        return self.api_key_for_provider(self.query_chat_model_provider)

    def provider_from_model(self, model_name: str, fallback_provider: str) -> str:
        """Infer provider from prefixed model string, else return fallback provider."""
        if "/" in model_name:
            return model_name.split("/", 1)[0].strip().lower()
        return fallback_provider

    @property
    def default_chat_model_provider(self) -> str:
        """Return provider for the effective default chat model."""
        fallback_provider = self._default_chat_model_config[1]
        model = self.default_chat_model or self._default_chat_model_config[0]
        return self.provider_from_model(model, fallback_provider)

    @property
    def default_chat_model_litellm(self) -> str:
        """Return LiteLLM model identifier for default chat calls."""
        model = self.default_chat_model or self._default_chat_model_config[0]
        provider = self.default_chat_model_provider
        if "/" in model:
            return model
        return f"{provider}/{model}" if provider else model

    @property
    def default_chat_model_api_key(self) -> str:
        """Return API key mapped to default chat model provider."""
        return self.api_key_for_provider(self.default_chat_model_provider)

    @cached_property
    def _default_embedding_model_config(self) -> tuple[str, str]:
        """Resolve default embedding model/provider from backend settings.yaml."""
        try:
            data = yaml.safe_load(self.settings_yaml_path.read_text(encoding="utf-8")) or {}
            embedding_model = (
                ((data.get("models") or {}).get("default_embedding_model") or {})
            )
            model = str(embedding_model.get("model") or "").strip()
            provider = str(embedding_model.get("model_provider") or "").strip().lower()
            if model and provider:
                return model, provider
            _config_logger.warning(
                "models.default_embedding_model is missing model/provider in settings.yaml; using defaults."
            )
        except Exception as exc:
            _config_logger.warning(
                "Failed to read default_embedding_model from settings.yaml: %s",
                exc,
            )
        return "gemini-embedding-001", "gemini"

    @property
    def default_embedding_model_provider(self) -> str:
        """Return provider configured for default embedding model."""
        return self._default_embedding_model_config[1]

    @property
    def default_embedding_model_litellm(self) -> str:
        """Return LiteLLM model identifier for embedding calls."""
        model = self.default_embedding_model or self._default_embedding_model_config[0]
        provider = self.default_embedding_model_provider
        if "/" in model:
            return model
        return f"{provider}/{model}" if provider else model


# Global settings instance
settings = Settings()
