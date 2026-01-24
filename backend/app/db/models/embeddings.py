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
