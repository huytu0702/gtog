"""Tests for operational models."""

import pytest
from datetime import datetime
from app.db.models.operational import Collection, IndexRun, IndexRunStatus


def test_collection_has_required_fields():
    """Test Collection model has required fields."""
    assert hasattr(Collection, "id")
    assert hasattr(Collection, "name")
    assert hasattr(Collection, "description")
    assert hasattr(Collection, "created_at")


def test_index_run_has_required_fields():
    """Test IndexRun model has required fields."""
    assert hasattr(IndexRun, "id")
    assert hasattr(IndexRun, "collection_id")
    assert hasattr(IndexRun, "status")
    assert hasattr(IndexRun, "started_at")
    assert hasattr(IndexRun, "finished_at")
    assert hasattr(IndexRun, "error")


def test_index_run_status_enum():
    """Test IndexRunStatus enum values."""
    assert IndexRunStatus.QUEUED.value == "queued"
    assert IndexRunStatus.RUNNING.value == "running"
    assert IndexRunStatus.COMPLETED.value == "completed"
    assert IndexRunStatus.FAILED.value == "failed"
