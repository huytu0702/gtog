"""Pytest fixtures for integration testing."""

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = (
        "postgresql+asyncpg://graphrag:graphrag@localhost:5432/graphrag"
    )

from app.db.models import Base
from app.main import app
from app.api.deps import get_db_session


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container with pgvector."""
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def redis_container():
    """Start Redis container."""
    with RedisContainer("redis:7-alpine") as redis:
        host = redis.get_container_host_ip()
        port = redis.get_exposed_port(6379)
        redis_url = f"redis://{host}:{port}"

        if not os.environ.get("REDIS_URL"):
            os.environ["REDIS_URL"] = redis_url

        from app.config import settings
        from app.worker.queue import get_queue, get_redis_connection

        settings.redis_url = redis_url
        get_redis_connection.cache_clear()
        get_queue.cache_clear()

        yield redis


@pytest_asyncio.fixture
async def db_engine(postgres_container):
    """Create database engine for tests."""
    url = make_url(postgres_container.get_connection_url())
    url = url.set(drivername="postgresql+asyncpg")
    engine = create_async_engine(url, echo=False)

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for tests."""
    session_factory = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(
    db_session: AsyncSession, redis_container
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with overridden dependencies."""

    async def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
