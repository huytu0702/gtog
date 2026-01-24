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
