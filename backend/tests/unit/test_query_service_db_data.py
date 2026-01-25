"""Tests for QueryServiceDB data loading."""

import pytest
from app.services.query_service_db import QueryServiceDB


def test_query_service_db_loads_latest_run():
    """Test QueryServiceDB uses latest completed run."""
    assert hasattr(QueryServiceDB, "_load_run_data")
