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
    doc_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

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
