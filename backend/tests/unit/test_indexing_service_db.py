"""Tests for database-backed indexing service."""

import pytest
from app.services.indexing_service_db import IndexingServiceDB


def test_indexing_service_has_methods():
    """Test IndexingServiceDB has required methods."""
    assert hasattr(IndexingServiceDB, "start_indexing")
    assert hasattr(IndexingServiceDB, "get_index_status")
