"""Tests for base repository."""

import pytest
from app.repositories.base import BaseRepository


def test_base_repository_has_crud_methods():
    """Test BaseRepository defines CRUD method signatures."""
    assert hasattr(BaseRepository, "get_by_id")
    assert hasattr(BaseRepository, "get_all")
    assert hasattr(BaseRepository, "create")
    assert hasattr(BaseRepository, "update")
    assert hasattr(BaseRepository, "delete")
