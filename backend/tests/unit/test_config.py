"""Tests for database configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def clear_env_vars(monkeypatch):
    """Clear environment variables that might interfere with tests."""
    for var in ["DATABASE_URL", "REDIS_URL", "POSTGRES_HOST", "POSTGRES_PORT",
                "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]:
        monkeypatch.delenv(var, raising=False)


def test_database_url_from_components(monkeypatch):
    """Test DATABASE_URL is built from components when not provided."""
    # Ensure env vars are cleared before importing Settings fresh
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Import fresh to avoid cached settings
    from importlib import reload
    import app.config as config_module
    reload(config_module)

    from app.config import Settings
    settings = Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="graphrag",
        postgres_password="secret",
        postgres_db="graphrag_test",
    )
    assert "postgresql+asyncpg://graphrag:secret@localhost:5432/graphrag_test" in settings.database_url


def test_database_url_override():
    """Test DATABASE_URL can be directly overridden."""
    from app.config import Settings
    settings = Settings(
        database_url="postgresql+asyncpg://custom:pass@db:5432/mydb"
    )
    assert settings.database_url == "postgresql+asyncpg://custom:pass@db:5432/mydb"


def test_redis_url_default(monkeypatch):
    """Test Redis URL default value."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    from importlib import reload
    import app.config as config_module
    reload(config_module)

    from app.config import Settings
    settings = Settings()
    assert settings.redis_url == "redis://localhost:6379/0"
