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
