"""Database engine configuration."""

from functools import lru_cache
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine


@lru_cache(maxsize=1)
def get_engine(database_url: str) -> AsyncEngine:
    """
    Create and cache async database engine.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        AsyncEngine instance
    """
    return create_async_engine(
        database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
