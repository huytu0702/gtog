"""Tests for database configuration."""

import os
import pytest
from app.config import Settings


def test_database_url_from_components():
    """Test DATABASE_URL is built from components when not provided."""
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
    settings = Settings(
        database_url="postgresql+asyncpg://custom:pass@db:5432/mydb"
    )
    assert settings.database_url == "postgresql+asyncpg://custom:pass@db:5432/mydb"


def test_redis_url_default():
    """Test Redis URL default value."""
    settings = Settings()
    assert settings.redis_url == "redis://localhost:6379/0"
