"""Application configuration using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _load_graphrag_settings_file(cls, values):
        graphrag_settings = os.getenv("GRAPHRAG_SETTINGS_FILE")
        if graphrag_settings:
            values["settings_file"] = graphrag_settings
        return values

    # API Configuration
    graphrag_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # Storage Configuration
    storage_root_dir: str = "./storage"
    storage_mode: str = "file"

    # Cosmos DB Configuration
    cosmos_endpoint: str = ""
    cosmos_key: str = ""
    cosmos_database: str = ""
    cosmos_container: str = ""

    # GraphRAG Settings
    settings_file: str = "settings.yaml"

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
        """Get the settings.yaml path based on settings_file."""
        path = Path(self.settings_file)
        if path.is_absolute():
            return path
        return Path(__file__).parent.parent / path

    @property
    def is_cosmos_mode(self) -> bool:
        """Check if storage mode is set to cosmos."""
        return self.storage_mode.strip().lower() == "cosmos"


# Global settings instance
settings = Settings()
