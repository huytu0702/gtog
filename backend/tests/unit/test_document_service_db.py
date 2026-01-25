"""Tests for database-backed document service."""

import pytest
from app.services.document_service_db import DocumentServiceDB


def test_document_service_has_methods():
    """Test DocumentServiceDB has required methods."""
    assert hasattr(DocumentServiceDB, "upload_document")
    assert hasattr(DocumentServiceDB, "list_documents")
    assert hasattr(DocumentServiceDB, "delete_document")
