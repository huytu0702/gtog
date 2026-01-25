"""Tests for document repository."""

import pytest
from app.repositories.document import DocumentRepository


def test_document_repository_has_methods():
    """Test DocumentRepository has required methods."""
    assert hasattr(DocumentRepository, "get_by_collection")
    assert hasattr(DocumentRepository, "get_by_name")
    assert hasattr(DocumentRepository, "delete_by_name")
