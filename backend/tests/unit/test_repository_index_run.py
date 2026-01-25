"""Tests for index run repository."""

import pytest
from app.repositories.index_run import IndexRunRepository


def test_index_run_repository_has_methods():
    """Test IndexRunRepository has required methods."""
    assert hasattr(IndexRunRepository, "get_latest_for_collection")
    assert hasattr(IndexRunRepository, "create_run")
