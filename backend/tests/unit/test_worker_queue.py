"""Tests for worker queue."""

import pytest
from app.worker.queue import get_queue, enqueue_indexing_job


def test_get_queue_returns_queue():
    """Test get_queue returns an RQ Queue."""
    from rq import Queue
    queue = get_queue()
    assert isinstance(queue, Queue)


def test_enqueue_indexing_job_signature():
    """Test enqueue_indexing_job has correct signature."""
    import inspect
    sig = inspect.signature(enqueue_indexing_job)
    params = list(sig.parameters.keys())
    assert "collection_id" in params
    assert "index_run_id" in params
