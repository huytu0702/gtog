"""Application configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    azure_cosmos_retry_on_status_codes: str = "429,503"
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

    # Query serving mode
    query_context_mode: str = "cosmos_only"
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
    default_chat_model: str = "gemini/gemini-2.5-flash-lite"
    default_embedding_model: str = "gemini/gemini-embedding-001"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    enable_tog_debug_endpoint: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    afd_origin_secret: str = ""
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120

    @property
    def collections_dir(self) -> Path:
        """Get the collections directory path."""
        return Path(self.storage_root_dir) / "collections"

    @property
    def settings_yaml_path(self) -> Path:
        """Get the shared settings.yaml path."""
        return Path(__file__).parent.parent / "settings.yaml"


# Global settings instance
settings = Settings()
