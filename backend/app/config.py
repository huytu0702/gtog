"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=[".env", ".env.test"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    graphrag_api_key: str = ""
    openai_api_key: str = ""


    # Model Configuration
    default_chat_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-3-small"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "graphrag"
    postgres_password: str = "graphrag"
    postgres_db: str = "graphrag"
    database_url: Optional[str] = None

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"

    @property
    def settings_yaml_path(self) -> Path:
        """Get the shared settings.yaml path."""
        return Path(__file__).parent.parent / "settings.yaml"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        if not self.redis_url:
            self.redis_url = "redis://localhost:6379/0"
        else:
            parsed = urlsplit(self.redis_url)
            if parsed.scheme and parsed.netloc and parsed.path in ("", "/"):
                self.redis_url = f"{parsed.scheme}://{parsed.netloc}/0"


# Global settings instance
settings = Settings()
