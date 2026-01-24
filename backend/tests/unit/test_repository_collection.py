"""Tests for collection repository."""

import pytest
from app.repositories.collection import CollectionRepository


def test_collection_repository_has_methods():
    """Test CollectionRepository has required methods."""
    assert hasattr(CollectionRepository, "get_by_name")
    assert hasattr(CollectionRepository, "get_with_document_count")
    assert hasattr(CollectionRepository, "get_latest_completed_run")
