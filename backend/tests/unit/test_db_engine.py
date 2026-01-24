"""Tests for database engine."""

import pytest
from app.db.engine import get_engine
from sqlalchemy.ext.asyncio import AsyncEngine


def test_get_engine_returns_async_engine():
    """Test get_engine returns an AsyncEngine."""
    engine = get_engine("postgresql+asyncpg://user:pass@localhost:5432/test")
    assert isinstance(engine, AsyncEngine)


def test_get_engine_caches_instance():
    """Test get_engine returns cached instance for same URL."""
    url = "postgresql+asyncpg://user:pass@localhost:5432/test"
    engine1 = get_engine(url)
    engine2 = get_engine(url)
    assert engine1 is engine2
