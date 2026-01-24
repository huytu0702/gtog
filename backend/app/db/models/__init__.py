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
