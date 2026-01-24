"""Tests for database-backed collection service."""

import pytest
from app.services.collection_service_db import CollectionServiceDB


def test_collection_service_has_methods():
    """Test CollectionServiceDB has required methods."""
    assert hasattr(CollectionServiceDB, "create_collection")
    assert hasattr(CollectionServiceDB, "get_collection")
    assert hasattr(CollectionServiceDB, "list_collections")
    assert hasattr(CollectionServiceDB, "delete_collection")
