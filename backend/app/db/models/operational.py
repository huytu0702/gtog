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
