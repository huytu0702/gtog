"""Tests for GraphRAG DB adapter."""

import pytest
from app.services.graphrag_db_adapter import GraphRAGDbAdapter


def test_adapter_has_methods():
    """Test adapter exposes ingestion methods."""
    assert hasattr(GraphRAGDbAdapter, "ingest_outputs")
