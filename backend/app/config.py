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
    azure_cosmos_connection_string: str = ""
    azure_cosmos_endpoint: str = ""
    azure_cosmos_key: str = ""
    azure_cosmos_database_name: str = "gtog-control"
    azure_cosmos_collections_container: str = "collections"
    azure_cosmos_documents_container: str = "documents"
    azure_cosmos_indexing_jobs_container: str = "indexingJobs"
    azure_cosmos_job_events_container: str = "jobEvents"
    azure_cosmos_artifact_manifest_container: str = "artifactManifest"

    # Model Configuration
    default_chat_model: str = "gemini/gemini-2.5-flash-lite"
    default_embedding_model: str = "gemini/gemini-embedding-001"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    enable_tog_debug_endpoint: bool = False

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
