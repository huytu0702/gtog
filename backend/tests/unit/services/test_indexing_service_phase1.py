"""Unit tests for durable Phase 2 indexing dispatch behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.models import IndexStatus
from backend.app.repositories import INDEX_JOB_QUEUED, INDEX_JOB_RETRYING, INDEX_JOB_RUNNING
from backend.app.services.indexing_service import IndexingService


@pytest.mark.asyncio
async def test_start_indexing_enqueues_job_and_dispatches_queue_message():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.queue_service = MagicMock()
    service.queue_service.is_configured.return_value = True
    service.control_plane.enqueue_indexing_job.return_value = (
        {
            "id": "job-1",
            "collectionId": "collection-a",
            "status": INDEX_JOB_QUEUED,
            "attempt": 0,
            "maxAttempts": 3,
            "progress": 0.0,
            "message": "Indexing job queued",
            "startedAt": None,
            "finishedAt": None,
            "nextAttemptAt": None,
            "heartbeatAt": None,
            "leaseOwnerId": None,
            "error": None,
        },
        True,
    )

    response = await service.start_indexing("collection-a")

    assert response.status == IndexStatus.PENDING
    assert response.job_id == "job-1"
    service.control_plane.enqueue_indexing_job.assert_called_once_with(
        "collection-a",
        max_attempts=3,
    )
    service.queue_service.send_indexing_job_message.assert_called_once_with(
        job_id="job-1",
        attempt=0,
    )


@pytest.mark.asyncio
async def test_start_indexing_does_not_redispatch_existing_active_job():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.queue_service = MagicMock()
    service.queue_service.is_configured.return_value = True
    service.control_plane.enqueue_indexing_job.return_value = (
        {
            "id": "job-2",
            "collectionId": "collection-a",
            "status": INDEX_JOB_QUEUED,
            "attempt": 1,
            "maxAttempts": 3,
            "progress": 0.0,
            "message": "Indexing job queued",
            "startedAt": None,
            "finishedAt": None,
            "nextAttemptAt": None,
            "heartbeatAt": None,
            "leaseOwnerId": None,
            "error": None,
        },
        False,
    )

    response = await service.start_indexing("collection-a")

    assert response.status == IndexStatus.PENDING
    service.queue_service.send_indexing_job_message.assert_not_called()


def test_get_index_status_reads_latest_job_from_cosmos_repository():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.control_plane.get_latest_indexing_job.return_value = {
        "id": "job-3",
        "collectionId": "collection-a",
        "status": INDEX_JOB_RUNNING,
        "attempt": 1,
        "maxAttempts": 3,
        "progress": 42.5,
        "message": "Running indexing pipeline...",
        "startedAt": "2026-03-01T10:00:00",
        "finishedAt": None,
        "nextAttemptAt": None,
        "heartbeatAt": "2026-03-01T10:01:00",
        "leaseOwnerId": "worker-a",
        "error": None,
    }

    response = service.get_index_status("collection-a")

    assert response is not None
    assert response.status == IndexStatus.RUNNING
    assert response.progress == 42.5
    assert response.lease_owner_id == "worker-a"


def test_get_index_status_maps_retrying_state():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.control_plane.get_latest_indexing_job.return_value = {
        "id": "job-4",
        "collectionId": "collection-a",
        "status": INDEX_JOB_RETRYING,
        "attempt": 1,
        "maxAttempts": 3,
        "progress": 0.0,
        "message": "Retry scheduled",
        "startedAt": "2026-03-01T10:00:00",
        "finishedAt": None,
        "nextAttemptAt": "2026-03-01T10:05:00",
        "heartbeatAt": None,
        "leaseOwnerId": None,
        "error": "temporary failure",
    }

    response = service.get_index_status("collection-a")

    assert response is not None
    assert response.status == IndexStatus.RETRYING
    assert response.retry_at is not None
    assert response.error == "temporary failure"


def test_create_vector_cosmos_client_strips_internal_scope_keys_from_connection_string_kwargs() -> None:
    service = IndexingService()
    client_kwargs = {
        "connection_timeout": 30,
        "retry_total": 9,
        "__collection_id": "collection-a",
        "__version": "v1",
        "__collection_version": "collection-a:v1",
    }
    default_store = MagicMock(
        connection_string="AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key;",
        client_kwargs=client_kwargs,
    )
    config = MagicMock(vector_store={"default_vector_store": default_store})

    with patch(
        "backend.app.services.indexing_service.CosmosClient.from_connection_string"
    ) as from_connection_string:
        service._create_vector_cosmos_client(config)

    from_connection_string.assert_called_once_with(
        "AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key;",
        connection_timeout=30,
        retry_total=9,
    )
    assert default_store.client_kwargs == client_kwargs
