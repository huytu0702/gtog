"""Unit tests for queue dispatch and worker behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.worker import _run_worker_loop


@pytest.mark.asyncio
async def test_worker_deletes_terminal_job_dispatch_message():
    message = SimpleNamespace(content='{"job_id": "job-1", "job_type": "indexing", "attempt": 1}')

    with patch("backend.app.worker.indexing_service") as mock_indexing:
        with patch("backend.app.worker.queue_service") as mock_queue:
            mock_indexing._ensure_dispatch_enabled.return_value = None
            mock_indexing._require_control_plane.return_value = mock_indexing.control_plane
            mock_indexing.requeue_recoverable_jobs.return_value = 0
            mock_queue.ensure_queue.return_value = None
            mock_queue.receive_messages.side_effect = [[message], asyncio.CancelledError()]
            mock_queue.decode_message.return_value = {"job_id": "job-1", "job_type": "indexing", "attempt": 1}
            mock_indexing.control_plane.get_indexing_job_by_id.return_value = {
                "id": "job-1",
                "collectionId": "c1",
                "status": "completed",
            }

            with pytest.raises(asyncio.CancelledError):
                await _run_worker_loop()

    mock_queue.delete_message.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_worker_polls_after_successful_queue_initialization():
    with patch("backend.app.worker.indexing_service") as mock_indexing:
        with patch("backend.app.worker.queue_service") as mock_queue:
            mock_indexing._ensure_dispatch_enabled.return_value = None
            mock_indexing._require_control_plane.return_value = mock_indexing.control_plane
            mock_indexing.requeue_recoverable_jobs.return_value = 0
            mock_queue.ensure_queue.return_value = None
            mock_queue.receive_messages.side_effect = [[], asyncio.CancelledError()]

            with pytest.raises(asyncio.CancelledError):
                await _run_worker_loop()

    mock_queue.ensure_queue.assert_called_once_with()
    assert mock_queue.receive_messages.call_count >= 1


@pytest.mark.asyncio
async def test_worker_executes_job_after_acquiring_lease():
    message = SimpleNamespace(content='{"job_id": "job-1", "job_type": "indexing", "attempt": 0}')

    with patch("backend.app.worker.indexing_service") as mock_indexing:
        with patch("backend.app.worker.queue_service") as mock_queue:
            mock_indexing._ensure_dispatch_enabled.return_value = None
            mock_indexing._require_control_plane.return_value = mock_indexing.control_plane
            mock_indexing.requeue_recoverable_jobs.return_value = 0
            mock_queue.ensure_queue.return_value = None
            mock_queue.receive_messages.side_effect = [[message], asyncio.CancelledError()]
            mock_queue.decode_message.return_value = {"job_id": "job-1", "job_type": "indexing", "attempt": 0}
            mock_indexing.control_plane.get_indexing_job_by_id.return_value = {
                "id": "job-1",
                "collectionId": "c1",
                "status": "queued",
            }
            mock_indexing.control_plane.acquire_indexing_job_lease.return_value = {
                "id": "job-1",
                "collectionId": "c1",
                "status": "queued",
            }
            mock_indexing.execute_indexing_job = AsyncMock(return_value={"id": "job-1"})

            with pytest.raises(asyncio.CancelledError):
                await _run_worker_loop()

    mock_indexing.execute_indexing_job.assert_awaited_once()
    mock_queue.delete_message.assert_called_once_with(message)
