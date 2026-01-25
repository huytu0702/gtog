"""Tests for database-backed query service."""

import pytest
from app.services.query_service_db import QueryServiceDB


def test_query_service_has_methods():
    """Test QueryServiceDB has required methods."""
    assert hasattr(QueryServiceDB, "global_search")
    assert hasattr(QueryServiceDB, "local_search")
    assert hasattr(QueryServiceDB, "tog_search")
    assert hasattr(QueryServiceDB, "drift_search")
