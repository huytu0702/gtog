"""Application configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Configuration
    graphrag_api_key: str = ""
    openai_api_key: str = ""
    
    # Storage Configuration
    storage_root_dir: str = "./storage"
    
    # Model Configuration
    default_chat_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-3-small"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
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
