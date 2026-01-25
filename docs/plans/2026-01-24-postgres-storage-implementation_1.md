# PostgreSQL Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate backend from filesystem storage to PostgreSQL with pgvector, Redis worker queue, and Docker Compose orchestration.

**Architecture:** Replace filesystem-based StorageService with SQLAlchemy ORM models backed by PostgreSQL. Documents stored as bytea, GraphRAG outputs in normalized tables, embeddings in pgvector. Indexing jobs run via Redis-backed worker queue (RQ). Docker Compose orchestrates postgres, redis, backend, and worker services.

**Tech Stack:** PostgreSQL 16 + pgvector, SQLAlchemy 2.0 (async), Alembic, Redis, RQ (Redis Queue), Docker Compose, pytest + testcontainers

---

## Phase 1: Database Foundation

### Task 1: Add Database Dependencies

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Write the updated requirements file**

Add these dependencies to `backend/requirements.txt`:

```txt
fastapi==0.100.0
uvicorn[standard]==0.23.2
python-multipart==0.0.6
pydantic==2.10.0
pydantic-settings==2.1.0
aiofiles==23.2.1
pandas==2.1.4
pyyaml==6.0.1

# Database
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
alembic==1.13.1
pgvector==0.2.4

# Worker Queue
redis==5.0.1
rq==1.16.0

# Testing
pytest==8.0.0
pytest-asyncio==0.23.3
testcontainers[postgres,redis]==3.7.1
httpx==0.26.0
```

**Step 2: Commit**

```bash
git add backend/requirements.txt
git commit -m "$(cat <<'EOF'
feat(backend): add database and worker queue dependencies

Add SQLAlchemy, asyncpg, Alembic for PostgreSQL ORM.
Add pgvector for embedding storage.
Add redis and rq for worker queue.
Add testing dependencies.

EOF
)"
```

---

### Task 2: Create Database Configuration

**Files:**
- Modify: `backend/app/config.py`

**Step 1: Write the failing test**

Create test file `backend/tests/unit/test_config.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_config.py -v`
Expected: FAIL with "no attribute 'database_url'"

**Step 3: Write minimal implementation**

Update `backend/app/config.py`:

```python
"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
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

    # Storage Configuration (legacy, kept for migration)
    storage_root_dir: str = "./storage"

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
    def collections_dir(self) -> Path:
        """Get the collections directory path (legacy)."""
        return Path(self.storage_root_dir) / "collections"

    @property
    def settings_yaml_path(self) -> Path:
        """Get the shared settings.yaml path."""
        return Path(__file__).parent.parent / "settings.yaml"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.database_url is None:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )


# Global settings instance
settings = Settings()
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/unit/test_config.py
git commit -m "$(cat <<'EOF'
feat(backend): add database and redis configuration

Add postgres connection settings with URL builder.
Add redis URL configuration.
Keep legacy storage settings for migration period.

EOF
)"
```

---

### Task 3: Create Database Engine and Session

**Files:**
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/engine.py`
- Create: `backend/app/db/session.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_db_engine.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_db_engine.py -v`
Expected: FAIL with "No module named 'app.db'"

**Step 3: Write minimal implementation**

Create `backend/app/db/__init__.py`:

```python
"""Database module."""

from .engine import get_engine
from .session import get_session, AsyncSessionLocal

__all__ = ["get_engine", "get_session", "AsyncSessionLocal"]
```

Create `backend/app/db/engine.py`:

```python
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
```

Create `backend/app/db/session.py`:

```python
"""Database session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..config import settings
from .engine import get_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get async session factory."""
    engine = get_engine(settings.database_url)
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


AsyncSessionLocal = get_session_factory()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Yields:
        AsyncSession instance
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_db_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/db/
git add backend/tests/unit/test_db_engine.py
git commit -m "$(cat <<'EOF'
feat(backend): add async database engine and session management

Add cached AsyncEngine factory.
Add async session context manager with auto-commit/rollback.

EOF
)"
```

---

## Phase 2: SQLAlchemy Models

### Task 4: Create Base Model with Common Fields

**Files:**
- Create: `backend/app/db/models/__init__.py`
- Create: `backend/app/db/models/base.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_models_base.py`:

```python
"""Tests for base model."""

import pytest
from uuid import UUID
from app.db.models.base import Base, GraphRAGBase


def test_graphrag_base_has_id_field():
    """Test GraphRAGBase has UUID id field."""
    assert hasattr(GraphRAGBase, "id")


def test_graphrag_base_has_human_readable_id():
    """Test GraphRAGBase has human_readable_id field."""
    assert hasattr(GraphRAGBase, "human_readable_id")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_models_base.py -v`
Expected: FAIL with "No module named 'app.db.models'"

**Step 3: Write minimal implementation**

Create `backend/app/db/models/__init__.py`:

```python
"""SQLAlchemy models."""

from .base import Base, GraphRAGBase

__all__ = ["Base", "GraphRAGBase"]
```

Create `backend/app/db/models/base.py`:

```python
"""Base model definitions."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class GraphRAGBase:
    """
    Mixin with shared fields for GraphRAG output tables.

    All GraphRAG output tables have:
    - id: UUID primary key
    - human_readable_id: int, incremented per-run for easy citation
    """

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    human_readable_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_models_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/db/models/
git add backend/tests/unit/test_models_base.py
git commit -m "$(cat <<'EOF'
feat(backend): add SQLAlchemy base models

Add DeclarativeBase for all models.
Add GraphRAGBase mixin with id and human_readable_id fields.

EOF
)"
```

---

### Task 5: Create Collection and IndexRun Models

**Files:**
- Create: `backend/app/db/models/operational.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_models_operational.py`:

```python
"""Tests for operational models."""

import pytest
from datetime import datetime
from app.db.models.operational import Collection, IndexRun, IndexRunStatus


def test_collection_has_required_fields():
    """Test Collection model has required fields."""
    assert hasattr(Collection, "id")
    assert hasattr(Collection, "name")
    assert hasattr(Collection, "description")
    assert hasattr(Collection, "created_at")


def test_index_run_has_required_fields():
    """Test IndexRun model has required fields."""
    assert hasattr(IndexRun, "id")
    assert hasattr(IndexRun, "collection_id")
    assert hasattr(IndexRun, "status")
    assert hasattr(IndexRun, "started_at")
    assert hasattr(IndexRun, "finished_at")
    assert hasattr(IndexRun, "error")


def test_index_run_status_enum():
    """Test IndexRunStatus enum values."""
    assert IndexRunStatus.QUEUED.value == "queued"
    assert IndexRunStatus.RUNNING.value == "running"
    assert IndexRunStatus.COMPLETED.value == "completed"
    assert IndexRunStatus.FAILED.value == "failed"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_models_operational.py -v`
Expected: FAIL with "No module named 'app.db.models.operational'"

**Step 3: Write minimal implementation**

Create `backend/app/db/models/operational.py`:

```python
"""Operational models for collections and index runs."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .graphrag import Document


class IndexRunStatus(str, Enum):
    """Status of an index run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Collection(Base):
    """Collection of documents for indexing."""

    __tablename__ = "collections"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    index_runs: Mapped[list["IndexRun"]] = relationship(
        "IndexRun",
        back_populates="collection",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="collection",
        cascade="all, delete-orphan",
    )


class IndexRun(Base):
    """Record of an indexing run."""

    __tablename__ = "index_runs"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[IndexRunStatus] = mapped_column(
        String(20),
        default=IndexRunStatus.QUEUED,
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="index_runs",
    )
```

Update `backend/app/db/models/__init__.py`:

```python
"""SQLAlchemy models."""

from .base import Base, GraphRAGBase
from .operational import Collection, IndexRun, IndexRunStatus

__all__ = [
    "Base",
    "GraphRAGBase",
    "Collection",
    "IndexRun",
    "IndexRunStatus",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_models_operational.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/db/models/
git add backend/tests/unit/test_models_operational.py
git commit -m "$(cat <<'EOF'
feat(backend): add Collection and IndexRun models

Add Collection model with name, description, created_at.
Add IndexRun model with status tracking and error field.
Add IndexRunStatus enum for run lifecycle.

EOF
)"
```

---

### Task 6: Create GraphRAG Output Models (Documents, Entities, Relationships)

**Files:**
- Create: `backend/app/db/models/graphrag.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_models_graphrag.py`:

```python
"""Tests for GraphRAG output models."""

import pytest
from app.db.models.graphrag import (
    Document,
    Entity,
    Relationship,
    Community,
    CommunityReport,
    TextUnit,
    Covariate,
)


def test_document_has_required_fields():
    """Test Document model has required fields."""
    assert hasattr(Document, "id")
    assert hasattr(Document, "human_readable_id")
    assert hasattr(Document, "collection_id")
    assert hasattr(Document, "index_run_id")
    assert hasattr(Document, "title")
    assert hasattr(Document, "text")
    assert hasattr(Document, "metadata")
    assert hasattr(Document, "filename")
    assert hasattr(Document, "content_type")
    assert hasattr(Document, "bytes_content")


def test_entity_has_required_fields():
    """Test Entity model has required fields."""
    assert hasattr(Entity, "id")
    assert hasattr(Entity, "title")
    assert hasattr(Entity, "type")
    assert hasattr(Entity, "description")
    assert hasattr(Entity, "frequency")
    assert hasattr(Entity, "degree")
    assert hasattr(Entity, "x")
    assert hasattr(Entity, "y")


def test_relationship_has_required_fields():
    """Test Relationship model has required fields."""
    assert hasattr(Relationship, "id")
    assert hasattr(Relationship, "source")
    assert hasattr(Relationship, "target")
    assert hasattr(Relationship, "description")
    assert hasattr(Relationship, "weight")
    assert hasattr(Relationship, "combined_degree")


def test_community_has_required_fields():
    """Test Community model has required fields."""
    assert hasattr(Community, "id")
    assert hasattr(Community, "community")
    assert hasattr(Community, "parent")
    assert hasattr(Community, "level")
    assert hasattr(Community, "title")
    assert hasattr(Community, "size")


def test_community_report_has_required_fields():
    """Test CommunityReport model has required fields."""
    assert hasattr(CommunityReport, "id")
    assert hasattr(CommunityReport, "community")
    assert hasattr(CommunityReport, "title")
    assert hasattr(CommunityReport, "summary")
    assert hasattr(CommunityReport, "full_content")
    assert hasattr(CommunityReport, "rank")
    assert hasattr(CommunityReport, "findings")


def test_text_unit_has_required_fields():
    """Test TextUnit model has required fields."""
    assert hasattr(TextUnit, "id")
    assert hasattr(TextUnit, "text")
    assert hasattr(TextUnit, "n_tokens")


def test_covariate_has_required_fields():
    """Test Covariate model has required fields."""
    assert hasattr(Covariate, "id")
    assert hasattr(Covariate, "covariate_type")
    assert hasattr(Covariate, "type")
    assert hasattr(Covariate, "description")
    assert hasattr(Covariate, "subject_id")
    assert hasattr(Covariate, "status")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_models_graphrag.py -v`
Expected: FAIL with "No module named 'app.db.models.graphrag'"

**Step 3: Write minimal implementation**

Create `backend/app/db/models/graphrag.py`:

```python
"""GraphRAG output models aligned with docs/index/outputs.md."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, GraphRAGBase

if TYPE_CHECKING:
    from .operational import Collection, IndexRun


class Document(GraphRAGBase, Base):
    """
    Document content after import.

    Combines GraphRAG document output with file storage.
    """

    __tablename__ = "documents"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # GraphRAG document fields
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # File storage fields
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    bytes_content: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(back_populates="documents")


class Entity(GraphRAGBase, Base):
    """Entity found in the data by the LLM."""

    __tablename__ = "entities"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    frequency: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    degree: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Relationship(GraphRAGBase, Base):
    """Entity-to-entity relationship (graph edge)."""

    __tablename__ = "relationships"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Entity references (by title for GraphRAG compatibility)
    source: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    target: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    # Optional foreign keys to entity records
    source_entity_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_entity_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="SET NULL"),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    combined_degree: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Community(GraphRAGBase, Base):
    """Leiden-generated community cluster."""

    __tablename__ = "communities"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    community: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    parent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    period: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class CommunityReport(GraphRAGBase, Base):
    """Summarized report for each community."""

    __tablename__ = "community_reports"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    community: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    parent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rank: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rating_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    findings: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    full_content_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    period: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class TextUnit(GraphRAGBase, Base):
    """Text chunk parsed from input documents."""

    __tablename__ = "text_units"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    text: Mapped[str] = mapped_column(Text, nullable=False)
    n_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Covariate(GraphRAGBase, Base):
    """Extracted covariate (claim) if claim extraction is enabled."""

    __tablename__ = "covariates"

    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    covariate_type: Mapped[str] = mapped_column(String(50), nullable=False)
    type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subject_id: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    object_id: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    start_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    end_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    source_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_unit_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("text_units.id", ondelete="SET NULL"),
        nullable=True,
    )
```

Update `backend/app/db/models/__init__.py`:

```python
"""SQLAlchemy models."""

from .base import Base, GraphRAGBase
from .operational import Collection, IndexRun, IndexRunStatus
from .graphrag import (
    Document,
    Entity,
    Relationship,
    Community,
    CommunityReport,
    TextUnit,
    Covariate,
)

__all__ = [
    "Base",
    "GraphRAGBase",
    "Collection",
    "IndexRun",
    "IndexRunStatus",
    "Document",
    "Entity",
    "Relationship",
    "Community",
    "CommunityReport",
    "TextUnit",
    "Covariate",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_models_graphrag.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/db/models/
git add backend/tests/unit/test_models_graphrag.py
git commit -m "$(cat <<'EOF'
feat(backend): add GraphRAG output models

Add Document, Entity, Relationship models.
Add Community, CommunityReport models.
Add TextUnit, Covariate models.
All aligned with docs/index/outputs.md schema.

EOF
)"
```

---

### Task 7: Create Join Tables and Embeddings Model

**Files:**
- Create: `backend/app/db/models/associations.py`
- Create: `backend/app/db/models/embeddings.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_models_associations.py`:

```python
"""Tests for association tables and embeddings."""

import pytest
from app.db.models.associations import (
    document_text_units,
    text_unit_entities,
    text_unit_relationships,
    community_entities,
    community_relationships,
    community_text_units,
    community_hierarchy,
)
from app.db.models.embeddings import Embedding, EmbeddingType


def test_association_tables_exist():
    """Test all association tables are defined."""
    assert document_text_units is not None
    assert text_unit_entities is not None
    assert text_unit_relationships is not None
    assert community_entities is not None
    assert community_relationships is not None
    assert community_text_units is not None
    assert community_hierarchy is not None


def test_embedding_has_required_fields():
    """Test Embedding model has required fields."""
    assert hasattr(Embedding, "id")
    assert hasattr(Embedding, "collection_id")
    assert hasattr(Embedding, "index_run_id")
    assert hasattr(Embedding, "embedding_type")
    assert hasattr(Embedding, "ref_id")
    assert hasattr(Embedding, "vector")


def test_embedding_type_enum():
    """Test EmbeddingType enum values."""
    assert EmbeddingType.TEXT_UNIT.value == "text_unit"
    assert EmbeddingType.ENTITY.value == "entity"
    assert EmbeddingType.COMMUNITY_REPORT.value == "community_report"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_models_associations.py -v`
Expected: FAIL with "No module named 'app.db.models.associations'"

**Step 3: Write minimal implementation**

Create `backend/app/db/models/associations.py`:

```python
"""Association (join) tables for many-to-many relationships."""

from sqlalchemy import Column, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


# documents.text_unit_ids <-> text_units.document_ids
document_text_units = Table(
    "document_text_units",
    Base.metadata,
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "text_unit_id",
        UUID(as_uuid=True),
        ForeignKey("text_units.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# text_units.entity_ids <-> entities.text_unit_ids
text_unit_entities = Table(
    "text_unit_entities",
    Base.metadata,
    Column(
        "text_unit_id",
        UUID(as_uuid=True),
        ForeignKey("text_units.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "entity_id",
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# text_units.relationship_ids <-> relationships.text_unit_ids
text_unit_relationships = Table(
    "text_unit_relationships",
    Base.metadata,
    Column(
        "text_unit_id",
        UUID(as_uuid=True),
        ForeignKey("text_units.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "relationship_id",
        UUID(as_uuid=True),
        ForeignKey("relationships.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# communities.entity_ids
community_entities = Table(
    "community_entities",
    Base.metadata,
    Column(
        "community_id",
        UUID(as_uuid=True),
        ForeignKey("communities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "entity_id",
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# communities.relationship_ids
community_relationships = Table(
    "community_relationships",
    Base.metadata,
    Column(
        "community_id",
        UUID(as_uuid=True),
        ForeignKey("communities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "relationship_id",
        UUID(as_uuid=True),
        ForeignKey("relationships.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# communities.text_unit_ids
community_text_units = Table(
    "community_text_units",
    Base.metadata,
    Column(
        "community_id",
        UUID(as_uuid=True),
        ForeignKey("communities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "text_unit_id",
        UUID(as_uuid=True),
        ForeignKey("text_units.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# communities.children (self-referential hierarchy)
community_hierarchy = Table(
    "community_hierarchy",
    Base.metadata,
    Column(
        "parent_id",
        UUID(as_uuid=True),
        ForeignKey("communities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "child_id",
        UUID(as_uuid=True),
        ForeignKey("communities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)
```

Create `backend/app/db/models/embeddings.py`:

```python
"""Embedding storage with pgvector."""

from enum import Enum
from uuid import uuid4

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

# Note: pgvector Vector type will be registered at runtime
# For now, we use a placeholder that will work with pgvector extension
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for testing without pgvector installed
    from sqlalchemy import LargeBinary as Vector


class EmbeddingType(str, Enum):
    """Type of content being embedded."""

    TEXT_UNIT = "text_unit"
    ENTITY = "entity"
    COMMUNITY_REPORT = "community_report"


class Embedding(Base):
    """Vector embedding storage using pgvector."""

    __tablename__ = "embeddings"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    collection_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding_type: Mapped[EmbeddingType] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )
    ref_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    # Vector dimension will be set based on embedding model (1536 for text-embedding-3-small)
    vector = mapped_column(Vector(1536), nullable=False)
```

Update `backend/app/db/models/__init__.py`:

```python
"""SQLAlchemy models."""

from .base import Base, GraphRAGBase
from .operational import Collection, IndexRun, IndexRunStatus
from .graphrag import (
    Document,
    Entity,
    Relationship,
    Community,
    CommunityReport,
    TextUnit,
    Covariate,
)
from .associations import (
    document_text_units,
    text_unit_entities,
    text_unit_relationships,
    community_entities,
    community_relationships,
    community_text_units,
    community_hierarchy,
)
from .embeddings import Embedding, EmbeddingType

__all__ = [
    # Base
    "Base",
    "GraphRAGBase",
    # Operational
    "Collection",
    "IndexRun",
    "IndexRunStatus",
    # GraphRAG outputs
    "Document",
    "Entity",
    "Relationship",
    "Community",
    "CommunityReport",
    "TextUnit",
    "Covariate",
    # Associations
    "document_text_units",
    "text_unit_entities",
    "text_unit_relationships",
    "community_entities",
    "community_relationships",
    "community_text_units",
    "community_hierarchy",
    # Embeddings
    "Embedding",
    "EmbeddingType",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_models_associations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/db/models/
git add backend/tests/unit/test_models_associations.py
git commit -m "$(cat <<'EOF'
feat(backend): add association tables and embeddings model

Add join tables for document-textunit, textunit-entity, etc.
Add community hierarchy self-referential table.
Add Embedding model with pgvector support.

EOF
)"
```

---

## Phase 3: Alembic Migrations

### Task 8: Initialize Alembic

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/versions/` (directory)

**Step 1: Create alembic configuration**

Create `backend/alembic.ini`:

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Create `backend/alembic/env.py`:

```python
"""Alembic environment configuration."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.config import settings
from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Get database URL from settings."""
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `backend/alembic/script.py.mako`:

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

Create empty `backend/alembic/versions/.gitkeep`.

**Step 2: Commit**

```bash
git add backend/alembic.ini backend/alembic/
git commit -m "$(cat <<'EOF'
feat(backend): initialize Alembic for migrations

Add alembic.ini with async PostgreSQL support.
Add env.py configured for async SQLAlchemy.
Add script.py.mako template.

EOF
)"
```

---

### Task 9: Create Initial Migration

**Files:**
- Create: `backend/alembic/versions/001_initial_schema.py`

**Step 1: Create initial migration**

Create `backend/alembic/versions/001_initial_schema.py`:

```python
"""Initial schema with all GraphRAG tables.

Revision ID: 001
Revises:
Create Date: 2026-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Collections table
    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Index runs table
    op.create_table(
        "index_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, default="queued"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("text", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=True),
        sa.Column("bytes_content", sa.LargeBinary, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Entities table
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=False, index=True),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("frequency", sa.Integer, nullable=True),
        sa.Column("degree", sa.Integer, nullable=True),
        sa.Column("x", sa.Float, nullable=True),
        sa.Column("y", sa.Float, nullable=True),
    )

    # Relationships table
    op.create_table(
        "relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source", sa.String(500), nullable=False, index=True),
        sa.Column("target", sa.String(500), nullable=False, index=True),
        sa.Column("source_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("target_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("weight", sa.Float, nullable=True),
        sa.Column("combined_degree", sa.Integer, nullable=True),
    )

    # Text units table
    op.create_table(
        "text_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("n_tokens", sa.Integer, nullable=True),
    )

    # Communities table
    op.create_table(
        "communities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("community", sa.Integer, nullable=False, index=True),
        sa.Column("parent", sa.Integer, nullable=True),
        sa.Column("level", sa.Integer, nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("period", sa.String(50), nullable=True),
        sa.Column("size", sa.Integer, nullable=True),
    )

    # Community reports table
    op.create_table(
        "community_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("community", sa.Integer, nullable=False, index=True),
        sa.Column("parent", sa.Integer, nullable=True),
        sa.Column("level", sa.Integer, nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("full_content", sa.Text, nullable=True),
        sa.Column("rank", sa.Float, nullable=True),
        sa.Column("rating_explanation", sa.Text, nullable=True),
        sa.Column("findings", postgresql.JSONB, nullable=True),
        sa.Column("full_content_json", postgresql.JSONB, nullable=True),
        sa.Column("period", sa.String(50), nullable=True),
        sa.Column("size", sa.Integer, nullable=True),
    )

    # Covariates table
    op.create_table(
        "covariates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("covariate_type", sa.String(50), nullable=False),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("subject_id", sa.String(500), nullable=True),
        sa.Column("object_id", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), nullable=True),
        sa.Column("start_date", sa.String(50), nullable=True),
        sa.Column("end_date", sa.String(50), nullable=True),
        sa.Column("source_text", sa.Text, nullable=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="SET NULL"), nullable=True),
    )

    # Embeddings table with pgvector
    op.create_table(
        "embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("embedding_type", sa.String(30), nullable=False, index=True),
        sa.Column("ref_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
    )
    # Add vector column with pgvector
    op.execute("ALTER TABLE embeddings ADD COLUMN vector vector(1536)")

    # Create HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX embeddings_vector_idx ON embeddings
        USING hnsw (vector vector_cosine_ops)
    """)

    # Association tables
    op.create_table(
        "document_text_units",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "text_unit_entities",
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "text_unit_relationships",
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("relationships.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_entities",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_relationships",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("relationships.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_text_units",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_hierarchy",
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("child_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade() -> None:
    # Drop association tables
    op.drop_table("community_hierarchy")
    op.drop_table("community_text_units")
    op.drop_table("community_relationships")
    op.drop_table("community_entities")
    op.drop_table("text_unit_relationships")
    op.drop_table("text_unit_entities")
    op.drop_table("document_text_units")

    # Drop main tables
    op.drop_table("embeddings")
    op.drop_table("covariates")
    op.drop_table("community_reports")
    op.drop_table("communities")
    op.drop_table("text_units")
    op.drop_table("relationships")
    op.drop_table("entities")
    op.drop_table("documents")
    op.drop_table("index_runs")
    op.drop_table("collections")

    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector")
```

**Step 2: Commit**

```bash
git add backend/alembic/versions/001_initial_schema.py
git commit -m "$(cat <<'EOF'
feat(backend): add initial database migration

Create all GraphRAG tables with proper indexes.
Enable pgvector extension for embeddings.
Add HNSW index for vector similarity search.

EOF
)"
```

---

## Phase 4: Repository Layer

### Task 10: Create Base Repository

**Files:**
- Create: `backend/app/repositories/__init__.py`
- Create: `backend/app/repositories/base.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_repository_base.py`:

```python
"""Tests for base repository."""

import pytest
from app.repositories.base import BaseRepository


def test_base_repository_has_crud_methods():
    """Test BaseRepository defines CRUD method signatures."""
    assert hasattr(BaseRepository, "get_by_id")
    assert hasattr(BaseRepository, "get_all")
    assert hasattr(BaseRepository, "create")
    assert hasattr(BaseRepository, "update")
    assert hasattr(BaseRepository, "delete")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_repository_base.py -v`
Expected: FAIL with "No module named 'app.repositories'"

**Step 3: Write minimal implementation**

Create `backend/app/repositories/__init__.py`:

```python
"""Repository layer for database operations."""

from .base import BaseRepository

__all__ = ["BaseRepository"]
```

Create `backend/app/repositories/base.py`:

```python
"""Base repository with common CRUD operations."""

from typing import Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with CRUD operations."""

    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        """
        Initialize repository.

        Args:
            session: Async database session
            model: SQLAlchemy model class
        """
        self.session = session
        self.model = model

    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """
        Get a record by ID.

        Args:
            id: Record UUID

        Returns:
            Model instance or None
        """
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[ModelType]:
        """
        Get all records with pagination.

        Args:
            limit: Max records to return
            offset: Records to skip

        Returns:
            List of model instances
        """
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def create(self, obj: ModelType) -> ModelType:
        """
        Create a new record.

        Args:
            obj: Model instance to create

        Returns:
            Created model instance
        """
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def update(self, obj: ModelType) -> ModelType:
        """
        Update an existing record.

        Args:
            obj: Model instance to update

        Returns:
            Updated model instance
        """
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def delete(self, obj: ModelType) -> bool:
        """
        Delete a record.

        Args:
            obj: Model instance to delete

        Returns:
            True if deleted
        """
        await self.session.delete(obj)
        await self.session.flush()
        return True
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_repository_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/repositories/
git add backend/tests/unit/test_repository_base.py
git commit -m "$(cat <<'EOF'
feat(backend): add base repository with CRUD operations

Add generic BaseRepository with get_by_id, get_all, create, update, delete.
Use async SQLAlchemy session for all operations.

EOF
)"
```

---

### Task 11: Create Collection Repository

**Files:**
- Create: `backend/app/repositories/collection.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_repository_collection.py`:

```python
"""Tests for collection repository."""

import pytest
from app.repositories.collection import CollectionRepository


def test_collection_repository_has_methods():
    """Test CollectionRepository has required methods."""
    assert hasattr(CollectionRepository, "get_by_name")
    assert hasattr(CollectionRepository, "get_with_document_count")
    assert hasattr(CollectionRepository, "get_latest_completed_run")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_repository_collection.py -v`
Expected: FAIL with "No module named 'app.repositories.collection'"

**Step 3: Write minimal implementation**

Create `backend/app/repositories/collection.py`:

```python
"""Collection repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Collection, Document, IndexRun, IndexRunStatus
from .base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    """Repository for collection operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Collection)

    async def get_by_name(self, name: str) -> Optional[Collection]:
        """
        Get collection by name.

        Args:
            name: Collection name

        Returns:
            Collection or None
        """
        result = await self.session.execute(
            select(Collection).where(Collection.name == name)
        )
        return result.scalar_one_or_none()

    async def get_with_document_count(self, collection_id: UUID) -> Optional[tuple]:
        """
        Get collection with document count.

        Args:
            collection_id: Collection UUID

        Returns:
            Tuple of (Collection, document_count) or None
        """
        result = await self.session.execute(
            select(
                Collection,
                func.count(Document.id).label("document_count")
            )
            .outerjoin(Document, Collection.id == Document.collection_id)
            .where(Collection.id == collection_id)
            .group_by(Collection.id)
        )
        row = result.first()
        if row:
            return row[0], row[1]
        return None

    async def get_latest_completed_run(self, collection_id: UUID) -> Optional[IndexRun]:
        """
        Get the latest completed index run for a collection.

        Args:
            collection_id: Collection UUID

        Returns:
            IndexRun or None
        """
        result = await self.session.execute(
            select(IndexRun)
            .where(IndexRun.collection_id == collection_id)
            .where(IndexRun.status == IndexRunStatus.COMPLETED)
            .order_by(IndexRun.finished_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def is_indexed(self, collection_id: UUID) -> bool:
        """
        Check if collection has a completed index run.

        Args:
            collection_id: Collection UUID

        Returns:
            True if indexed
        """
        run = await self.get_latest_completed_run(collection_id)
        return run is not None
```

Update `backend/app/repositories/__init__.py`:

```python
"""Repository layer for database operations."""

from .base import BaseRepository
from .collection import CollectionRepository

__all__ = ["BaseRepository", "CollectionRepository"]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_repository_collection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/repositories/
git add backend/tests/unit/test_repository_collection.py
git commit -m "$(cat <<'EOF'
feat(backend): add collection repository

Add get_by_name for unique constraint checks.
Add get_with_document_count for API responses.
Add get_latest_completed_run for query operations.

EOF
)"
```

---

## Phase 5: Docker Compose Infrastructure

### Task 12: Create Docker Compose Configuration

**Files:**
- Create: `backend/docker-compose.yml`
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`

**Step 1: Create Docker configuration files**

Create `backend/docker-compose.yml`:

```yaml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: graphrag-postgres
    environment:
      POSTGRES_USER: graphrag
      POSTGRES_PASSWORD: graphrag
      POSTGRES_DB: graphrag
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U graphrag"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: graphrag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: graphrag-backend
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=graphrag
      - POSTGRES_PASSWORD=graphrag
      - POSTGRES_DB=graphrag
      - REDIS_URL=redis://redis:6379/0
      - GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: graphrag-worker
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=graphrag
      - POSTGRES_PASSWORD=graphrag
      - POSTGRES_DB=graphrag
      - REDIS_URL=redis://redis:6379/0
      - GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./:/app
    command: rq worker --url redis://redis:6379/0 graphrag-indexing

volumes:
  postgres_data:
  redis_data:
```

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install graphrag from parent directory (editable mode)
COPY --from=context .. /graphrag
RUN pip install -e /graphrag

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `backend/.dockerignore`:

```
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
.venv
venv
venv_backend
*.egg-info
.pytest_cache
.coverage
htmlcov
.mypy_cache
.ruff_cache
```

**Step 2: Commit**

```bash
git add backend/docker-compose.yml backend/Dockerfile backend/.dockerignore
git commit -m "$(cat <<'EOF'
feat(backend): add Docker Compose infrastructure

Add postgres service with pgvector extension.
Add redis service for worker queue.
Add backend and worker services.
Configure health checks and volume persistence.

EOF
)"
```

---

## Phase 6: Worker Queue

### Task 13: Create Worker Queue Infrastructure

**Files:**
- Create: `backend/app/worker/__init__.py`
- Create: `backend/app/worker/queue.py`
- Create: `backend/app/worker/tasks.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_worker_queue.py`:

```python
"""Tests for worker queue."""

import pytest
from app.worker.queue import get_queue, enqueue_indexing_job


def test_get_queue_returns_queue():
    """Test get_queue returns an RQ Queue."""
    from rq import Queue
    queue = get_queue()
    assert isinstance(queue, Queue)


def test_enqueue_indexing_job_signature():
    """Test enqueue_indexing_job has correct signature."""
    import inspect
    sig = inspect.signature(enqueue_indexing_job)
    params = list(sig.parameters.keys())
    assert "collection_id" in params
    assert "index_run_id" in params
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_worker_queue.py -v`
Expected: FAIL with "No module named 'app.worker'"

**Step 3: Write minimal implementation**

Create `backend/app/worker/__init__.py`:

```python
"""Worker queue module."""

from .queue import get_queue, enqueue_indexing_job

__all__ = ["get_queue", "enqueue_indexing_job"]
```

Create `backend/app/worker/queue.py`:

```python
"""Redis queue configuration."""

from functools import lru_cache
from uuid import UUID

from redis import Redis
from rq import Queue

from app.config import settings


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    """Get cached Redis connection."""
    return Redis.from_url(settings.redis_url)


@lru_cache(maxsize=1)
def get_queue() -> Queue:
    """Get the indexing job queue."""
    return Queue("graphrag-indexing", connection=get_redis_connection())


def enqueue_indexing_job(collection_id: UUID, index_run_id: UUID) -> str:
    """
    Enqueue an indexing job.

    Args:
        collection_id: Collection to index
        index_run_id: Index run record ID

    Returns:
        Job ID
    """
    from app.worker.tasks import run_indexing_task

    queue = get_queue()
    job = queue.enqueue(
        run_indexing_task,
        str(collection_id),
        str(index_run_id),
        job_timeout="2h",
        result_ttl=86400,  # 24 hours
    )
    return job.id
```

Create `backend/app/worker/tasks.py`:

```python
"""Worker tasks for background processing."""

import asyncio
import logging
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


def run_indexing_task(collection_id: str, index_run_id: str) -> dict:
    """
    Run indexing pipeline as a background task.

    This is called by the RQ worker.

    Args:
        collection_id: Collection UUID as string
        index_run_id: Index run UUID as string

    Returns:
        Result dict with status
    """
    return asyncio.run(_run_indexing_async(
        UUID(collection_id),
        UUID(index_run_id)
    ))


async def _run_indexing_async(collection_id: UUID, index_run_id: UUID) -> dict:
    """
    Async implementation of indexing task.

    Args:
        collection_id: Collection UUID
        index_run_id: Index run UUID

    Returns:
        Result dict with status
    """
    from app.db.session import get_session
    from app.db.models import IndexRun, IndexRunStatus

    logger.info(f"Starting indexing for collection {collection_id}, run {index_run_id}")

    async with get_session() as session:
        # Get index run
        from sqlalchemy import select
        result = await session.execute(
            select(IndexRun).where(IndexRun.id == index_run_id)
        )
        index_run = result.scalar_one_or_none()

        if not index_run:
            logger.error(f"Index run {index_run_id} not found")
            return {"status": "error", "message": "Index run not found"}

        try:
            # Update status to running
            index_run.status = IndexRunStatus.RUNNING
            index_run.started_at = datetime.now()
            await session.commit()

            # TODO: Run actual GraphRAG indexing pipeline
            # This will be implemented in Phase 7

            # For now, mark as completed
            index_run.status = IndexRunStatus.COMPLETED
            index_run.finished_at = datetime.now()
            await session.commit()

            logger.info(f"Indexing completed for collection {collection_id}")
            return {"status": "completed", "index_run_id": str(index_run_id)}

        except Exception as e:
            logger.exception(f"Indexing failed for collection {collection_id}")
            index_run.status = IndexRunStatus.FAILED
            index_run.finished_at = datetime.now()
            index_run.error = str(e)
            await session.commit()
            return {"status": "failed", "error": str(e)}
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_worker_queue.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/worker/
git add backend/tests/unit/test_worker_queue.py
git commit -m "$(cat <<'EOF'
feat(backend): add Redis worker queue for indexing jobs

Add RQ queue configuration with Redis connection.
Add enqueue_indexing_job function.
Add run_indexing_task with status updates.

EOF
)"
```

---

## Phase 7: Service Layer Refactoring

### Task 14: Create Database-Backed Collection Service

**Files:**
- Create: `backend/app/services/collection_service_db.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_collection_service_db.py`:

```python
"""Tests for database-backed collection service."""

import pytest
from app.services.collection_service_db import CollectionServiceDB


def test_collection_service_has_methods():
    """Test CollectionServiceDB has required methods."""
    assert hasattr(CollectionServiceDB, "create_collection")
    assert hasattr(CollectionServiceDB, "get_collection")
    assert hasattr(CollectionServiceDB, "list_collections")
    assert hasattr(CollectionServiceDB, "delete_collection")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_collection_service_db.py -v`
Expected: FAIL with "No module named 'app.services.collection_service_db'"

**Step 3: Write minimal implementation**

Create `backend/app/services/collection_service_db.py`:

```python
"""Database-backed collection service."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Document, IndexRun, IndexRunStatus
from app.models import CollectionResponse
from app.repositories import CollectionRepository


class CollectionServiceDB:
    """Service for collection operations using database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = CollectionRepository(session)

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None
    ) -> CollectionResponse:
        """
        Create a new collection.

        Args:
            name: Collection name (unique)
            description: Optional description

        Returns:
            CollectionResponse

        Raises:
            ValueError: If collection already exists
        """
        # Check if exists
        existing = await self.repo.get_by_name(name)
        if existing:
            raise ValueError(f"Collection '{name}' already exists")

        # Create collection
        collection = Collection(
            name=name,
            description=description,
        )
        await self.repo.create(collection)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=0,
            indexed=False,
        )

    async def get_collection(self, collection_id: UUID) -> Optional[CollectionResponse]:
        """
        Get collection by ID.

        Args:
            collection_id: Collection UUID

        Returns:
            CollectionResponse or None
        """
        result = await self.repo.get_with_document_count(collection_id)
        if not result:
            return None

        collection, doc_count = result
        is_indexed = await self.repo.is_indexed(collection_id)

        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            document_count=doc_count,
            indexed=is_indexed,
        )

    async def get_collection_by_name(self, name: str) -> Optional[CollectionResponse]:
        """
        Get collection by name.

        Args:
            name: Collection name

        Returns:
            CollectionResponse or None
        """
        collection = await self.repo.get_by_name(name)
        if not collection:
            return None
        return await self.get_collection(collection.id)

    async def list_collections(self) -> List[CollectionResponse]:
        """
        List all collections.

        Returns:
            List of CollectionResponse
        """
        collections = await self.repo.get_all()
        result = []
        for collection in collections:
            resp = await self.get_collection(collection.id)
            if resp:
                result.append(resp)
        return result

    async def delete_collection(self, collection_id: UUID) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection UUID

        Returns:
            True if deleted

        Raises:
            ValueError: If collection not found
        """
        collection = await self.repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection not found")

        await self.repo.delete(collection)
        return True
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_collection_service_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/collection_service_db.py
git add backend/tests/unit/test_collection_service_db.py
git commit -m "$(cat <<'EOF'
feat(backend): add database-backed collection service

Add CollectionServiceDB with CRUD operations.
Use repository pattern for database access.
Return same response models as filesystem service.

EOF
)"
```

---

## Phase 8: API Router Updates

### Task 15: Update Collection Router for Database

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/deps.py`
- Modify: `backend/app/routers/collections.py`

**Step 1: Create API dependencies**

Create `backend/app/api/__init__.py`:

```python
"""API module."""

from .deps import get_db_session, get_collection_service

__all__ = ["get_db_session", "get_collection_service"]
```

Create `backend/app/api/deps.py`:

```python
"""FastAPI dependencies for dependency injection."""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.services.collection_service_db import CollectionServiceDB


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Yields:
        AsyncSession
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_collection_service(
    session: AsyncSession = Depends(get_db_session)
) -> CollectionServiceDB:
    """
    FastAPI dependency for collection service.

    Args:
        session: Database session from dependency

    Returns:
        CollectionServiceDB instance
    """
    return CollectionServiceDB(session)
```

**Step 2: Update collections router**

The router at `backend/app/routers/collections.py` should be updated to use the new database service. This is a larger change that maintains API compatibility:

```python
"""Collection management endpoints."""

import logging
from typing import Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.api.deps import get_collection_service
from app.config import settings
from app.models import CollectionCreate, CollectionResponse, CollectionList
from app.services.collection_service_db import CollectionServiceDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Create a new collection."""
    try:
        logger.info(f"Creating collection: {collection.name}")
        result = await service.create_collection(
            name=collection.name,
            description=collection.description,
        )
        logger.info(f"Created collection: {collection.name}")
        return result
    except ValueError as e:
        logger.warning(f"Conflict creating collection {collection.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "Collection already exists", "message": str(e)},
        )
    except Exception as e:
        logger.exception(f"Error creating collection {collection.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.get("", response_model=CollectionList)
async def list_collections(
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """List all collections."""
    try:
        collections = await service.list_collections()
        return CollectionList(collections=collections, total=len(collections))
    except Exception as e:
        logger.exception("Error listing collections")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Get details about a specific collection."""
    try:
        # Try to parse as UUID first, fall back to name lookup
        try:
            uuid_id = UUID(collection_id)
            collection = await service.get_collection(uuid_id)
        except ValueError:
            collection = await service.get_collection_by_name(collection_id)

        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_id}' not found",
            )
        return collection
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting collection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: str,
    service: CollectionServiceDB = Depends(get_collection_service),
):
    """Delete a collection and all its contents."""
    try:
        try:
            uuid_id = UUID(collection_id)
        except ValueError:
            # Look up by name
            coll = await service.get_collection_by_name(collection_id)
            if not coll:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection '{collection_id}' not found",
                )
            uuid_id = UUID(coll.id)

        await service.delete_collection(uuid_id)
        logger.info(f"Deleted collection: {collection_id}")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Error deleting collection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
```

**Step 3: Commit**

```bash
git add backend/app/api/
git add backend/app/routers/collections.py
git commit -m "$(cat <<'EOF'
feat(backend): update collection router to use database

Add FastAPI dependencies for session and service injection.
Update collection endpoints to use CollectionServiceDB.
Return 409 on collection name conflicts.

EOF
)"
```

---

## Phase 9: Integration Testing

### Task 16: Add Integration Tests with Testcontainers

**Files:**
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/integration/test_collections_api.py`

**Step 1: Create test fixtures**

Create `backend/tests/conftest.py`:

```python
"""Pytest fixtures for integration testing."""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

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
        yield redis


@pytest_asyncio.fixture
async def db_engine(postgres_container):
    """Create database engine for tests."""
    url = postgres_container.get_connection_url().replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(url, echo=False)

    async with engine.begin() as conn:
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
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
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
```

**Step 2: Create integration tests**

Create `backend/tests/integration/test_collections_api.py`:

```python
"""Integration tests for collections API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_collection(client: AsyncClient):
    """Test creating a new collection."""
    response = await client.post(
        "/api/collections",
        json={"name": "test-collection", "description": "Test description"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-collection"
    assert data["description"] == "Test description"
    assert data["document_count"] == 0
    assert data["indexed"] is False


@pytest.mark.asyncio
async def test_create_duplicate_collection_returns_409(client: AsyncClient):
    """Test creating duplicate collection returns 409."""
    # Create first
    await client.post("/api/collections", json={"name": "duplicate-test"})

    # Try to create duplicate
    response = await client.post("/api/collections", json={"name": "duplicate-test"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_collections(client: AsyncClient):
    """Test listing collections."""
    # Create some collections
    await client.post("/api/collections", json={"name": "list-test-1"})
    await client.post("/api/collections", json={"name": "list-test-2"})

    response = await client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert data["total"] >= 2


@pytest.mark.asyncio
async def test_get_collection(client: AsyncClient):
    """Test getting a specific collection."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "get-test"})
    collection_id = create_resp.json()["id"]

    # Get by ID
    response = await client.get(f"/api/collections/{collection_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "get-test"


@pytest.mark.asyncio
async def test_get_collection_by_name(client: AsyncClient):
    """Test getting collection by name."""
    await client.post("/api/collections", json={"name": "name-lookup-test"})

    response = await client.get("/api/collections/name-lookup-test")
    assert response.status_code == 200
    assert response.json()["name"] == "name-lookup-test"


@pytest.mark.asyncio
async def test_get_nonexistent_collection_returns_404(client: AsyncClient):
    """Test getting nonexistent collection returns 404."""
    response = await client.get("/api/collections/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection(client: AsyncClient):
    """Test deleting a collection."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "delete-test"})
    collection_id = create_resp.json()["id"]

    # Delete
    response = await client.delete(f"/api/collections/{collection_id}")
    assert response.status_code == 204

    # Verify deleted
    get_resp = await client.get(f"/api/collections/{collection_id}")
    assert get_resp.status_code == 404
```

**Step 3: Run integration tests**

Run: `cd backend && pytest tests/integration/ -v`
Expected: PASS (requires Docker for testcontainers)

**Step 4: Commit**

```bash
git add backend/tests/conftest.py
git add backend/tests/integration/
git commit -m "$(cat <<'EOF'
test(backend): add integration tests with testcontainers

Add pytest fixtures for PostgreSQL and Redis containers.
Add collection API integration tests.
Test CRUD operations and error cases (409, 404).

EOF
)"
```

---

## Remaining Phases (Summary)

The following phases continue the implementation:

### Phase 10: Document Service (Tasks 17-18)
- Create DocumentRepository
- Create DocumentServiceDB with upload/list/delete
- Update document router

### Phase 11: Indexing Service (Tasks 19-21)
- Create IndexRunRepository
- Create IndexingServiceDB that enqueues jobs
- Implement GraphRAG adapter to write outputs to DB
- Update indexing router

### Phase 12: Query Service (Tasks 22-24)
- Create query repositories for entities, relationships, etc.
- Create QueryServiceDB that reads from latest index run
- Update search routers

### Phase 13: Lifespan and Migration Runner (Tasks 25-26)
- Update main.py lifespan to run migrations
- Add startup database health check
- Add graceful shutdown

### Phase 14: End-to-End Testing (Tasks 27-28)
- Full indexing + search integration test
- Performance tests for batch inserts

---

## Execution Commands

**Start development environment:**
```bash
cd backend
docker-compose up -d postgres redis
alembic upgrade head
uvicorn app.main:app --reload
```

**Run worker:**
```bash
cd backend
rq worker --url redis://localhost:6379/0 graphrag-indexing
```

**Run all tests:**
```bash
cd backend
pytest tests/ -v
```

**Run migrations:**
```bash
cd backend
alembic upgrade head
```
