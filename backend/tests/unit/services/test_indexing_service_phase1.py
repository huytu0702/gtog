"""Unit tests for Phase 1 Cosmos-backed indexing service behavior."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from backend.app.models import IndexStatus
from backend.app.repositories import INDEX_JOB_QUEUED, INDEX_JOB_RUNNING
from backend.app.services.indexing_service import IndexingService

indexing_service_module = importlib.import_module("backend.app.services.indexing_service")


class _FakeTask:
    def __init__(self, done: bool) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done


@pytest.mark.asyncio
async def test_start_indexing_enqueues_job_and_schedules_background_task():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.control_plane.enqueue_indexing_job.return_value = (
        {
            "id": "job-1",
            "status": INDEX_JOB_QUEUED,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
        },
        True,
    )

    fake_task = _FakeTask(done=False)
    def _fake_create_task(coro):
        coro.close()
        return fake_task

    with patch.object(indexing_service_module.asyncio, "create_task", side_effect=_fake_create_task) as create_task:
        response = await service.start_indexing("collection-a")

    assert response.status == IndexStatus.PENDING
    service.control_plane.enqueue_indexing_job.assert_called_once_with(
        "collection-a",
        max_attempts=3,
    )
    create_task.assert_called_once()
    assert service.running_tasks["collection-a"] is fake_task


@pytest.mark.asyncio
async def test_start_indexing_is_idempotent_when_running_task_exists():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.control_plane.enqueue_indexing_job.return_value = (
        {
            "id": "job-2",
            "status": INDEX_JOB_QUEUED,
            "startedAt": None,
            "finishedAt": None,
            "error": None,
        },
        False,
    )
    service.running_tasks["collection-a"] = _FakeTask(done=False)

    with patch.object(indexing_service_module.asyncio, "create_task") as create_task:
        response = await service.start_indexing("collection-a")

    assert response.status == IndexStatus.PENDING
    create_task.assert_not_called()


def test_get_index_status_reads_latest_job_from_cosmos_repository():
    service = IndexingService()
    service.control_plane = MagicMock()
    service.control_plane.get_latest_indexing_job.return_value = {
        "id": "job-3",
        "status": INDEX_JOB_RUNNING,
        "startedAt": "2026-03-01T10:00:00",
        "finishedAt": None,
        "error": None,
    }
    service.runtime_progress["job-3"] = {
        "progress": 42.5,
        "message": "Running indexing pipeline...",
    }

    response = service.get_index_status("collection-a")

    assert response is not None
    assert response.status == IndexStatus.RUNNING
    assert response.progress == 42.5
