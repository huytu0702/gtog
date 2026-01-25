"""Tests for query repository."""

import pytest
from app.repositories.query import QueryRepository


def test_query_repository_has_methods():
    """Test QueryRepository has required methods."""
    assert hasattr(QueryRepository, "get_latest_run_data")
