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
