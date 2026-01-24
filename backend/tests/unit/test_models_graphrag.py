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
    assert hasattr(Document, "doc_metadata")
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
