"""Unit tests for queue service behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceExistsError

from backend.app.services.queue_service import QueueService


def test_ensure_queue_ignores_existing_queue_error():
    service = QueueService()
    queue_client = MagicMock()
    queue_client.create_queue.side_effect = ResourceExistsError("Queue already exists")

    with patch.object(service, "_ensure_client", return_value=queue_client):
        service.ensure_queue()

    queue_client.create_queue.assert_called_once_with()


def test_ensure_queue_raises_unexpected_errors():
    service = QueueService()
    queue_client = MagicMock()
    queue_client.create_queue.side_effect = RuntimeError("boom")

    with patch.object(service, "_ensure_client", return_value=queue_client):
        with pytest.raises(RuntimeError, match="boom"):
            service.ensure_queue()

    queue_client.create_queue.assert_called_once_with()
