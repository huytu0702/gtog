"""Unit tests for control-plane indexing job lifecycle helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.repositories.control_plane_repository import (
    INDEX_JOB_COMPLETED,
    INDEX_JOB_QUEUED,
    INDEX_JOB_RETRYING,
    INDEX_JOB_RUNNING,
    CosmosControlPlaneRepository,
)


@pytest.fixture
def repository() -> CosmosControlPlaneRepository:
    repo = object.__new__(CosmosControlPlaneRepository)
    repo._containers = {"jobs": MagicMock(), "events": MagicMock()}
    repo._container_names = {
        "indexing_jobs": "jobs",
        "job_events": "events",
    }
    return repo


def test_enqueue_indexing_job_returns_existing_active_job(repository: CosmosControlPlaneRepository):
    existing = {"id": "job-1", "status": INDEX_JOB_QUEUED}
    repository._containers["jobs"].query_items.return_value = [existing]

    job, created = repository.enqueue_indexing_job("c1", max_attempts=3)

    assert job == existing
    assert created is False


def test_transition_running_increments_attempt_and_clears_error(repository: CosmosControlPlaneRepository):
    repository._containers["jobs"].read_item.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": INDEX_JOB_QUEUED,
        "attempt": 0,
        "maxAttempts": 3,
        "_etag": "etag-1",
    }
    repository._containers["jobs"].replace_item.side_effect = lambda item, body, **_: body
    repository.record_job_event = MagicMock()

    updated = repository.transition_indexing_job(
        collection_id="c1",
        job_id="job-1",
        to_status=INDEX_JOB_RUNNING,
    )

    assert updated["status"] == INDEX_JOB_RUNNING
    assert updated["attempt"] == 1
    assert updated["error"] is None


def test_transition_retrying_clears_lease_and_sets_next_attempt(repository: CosmosControlPlaneRepository):
    repository._containers["jobs"].read_item.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": INDEX_JOB_RUNNING,
        "attempt": 1,
        "maxAttempts": 3,
        "leaseOwnerId": "worker-a",
        "leaseAcquiredAt": "2026-03-01T10:00:00",
        "leaseExpiresAt": "2026-03-01T10:05:00",
        "heartbeatAt": "2026-03-01T10:01:00",
        "_etag": "etag-1",
    }
    repository._containers["jobs"].replace_item.side_effect = lambda item, body, **_: body
    repository.record_job_event = MagicMock()

    updated = repository.transition_indexing_job(
        collection_id="c1",
        job_id="job-1",
        to_status=INDEX_JOB_RETRYING,
        error="temporary failure",
        next_attempt_at="2026-03-01T10:05:00",
        expected_lease_owner="worker-a",
    )

    assert updated["status"] == INDEX_JOB_RETRYING
    assert updated["leaseOwnerId"] is None
    assert updated["nextAttemptAt"] == "2026-03-01T10:05:00"
    assert updated["error"] == "temporary failure"


def test_transition_completed_sets_finished_at(repository: CosmosControlPlaneRepository):
    repository._containers["jobs"].read_item.return_value = {
        "id": "job-1",
        "collectionId": "c1",
        "status": INDEX_JOB_RUNNING,
        "attempt": 1,
        "maxAttempts": 3,
        "leaseOwnerId": "worker-a",
        "_etag": "etag-1",
    }
    repository._containers["jobs"].replace_item.side_effect = lambda item, body, **_: body
    repository.record_job_event = MagicMock()

    updated = repository.transition_indexing_job(
        collection_id="c1",
        job_id="job-1",
        to_status=INDEX_JOB_COMPLETED,
        expected_lease_owner="worker-a",
    )

    assert updated["status"] == INDEX_JOB_COMPLETED
    assert updated["finishedAt"] is not None
    assert updated["leaseOwnerId"] is None
