"""Azure Storage Queue helpers for indexing job dispatch."""

from __future__ import annotations

import json
import logging
from typing import Any

from azure.core.exceptions import ResourceExistsError

from ..azure_runtime import create_queue_service_client
from ..config import settings

logger = logging.getLogger(__name__)


class QueueService:
    """Thin wrapper around Azure Storage Queue for job dispatch."""

    def __init__(self) -> None:
        self._queue_service_client = None
        self._queue_client = None

    def _ensure_client(self):
        if self._queue_client is not None:
            return self._queue_client

        queue_service_client = create_queue_service_client()
        if queue_service_client is None:
            raise RuntimeError(
                "Azure Storage Queue is not configured. Configure storage connection settings "
                "or enable managed identity with an accessible queue endpoint."
            )

        queue_client = queue_service_client.get_queue_client(settings.azure_storage_queue_name)
        self._queue_service_client = queue_service_client
        self._queue_client = queue_client
        return queue_client

    def ensure_queue(self) -> None:
        """Create the dispatch queue if it does not exist."""
        try:
            self._ensure_client().create_queue()
        except ResourceExistsError:
            logger.info("Queue %s already exists", settings.azure_storage_queue_name)

    def is_configured(self) -> bool:
        """Return True when queue auth configuration is available."""
        try:
            self._ensure_client()
        except Exception:
            return False
        return True

    def get_queue_properties(self) -> dict[str, Any]:
        """Return queue metadata for readiness checks."""
        properties = self._ensure_client().get_queue_properties()
        return dict(properties or {})

    def send_indexing_job_message(
        self,
        *,
        job_id: str,
        attempt: int,
        visibility_timeout: int = 0,
        job_type: str = "indexing",
    ) -> None:
        """Send a minimal indexing dispatch message."""
        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "attempt": attempt,
        }
        self._ensure_client().send_message(
            json.dumps(payload),
            visibility_timeout=max(0, visibility_timeout),
        )

    def receive_messages(self) -> list[Any]:
        """Receive a batch of queue messages."""
        messages = self._ensure_client().receive_messages(
            messages_per_page=settings.azure_storage_queue_dequeue_batch_size,
            visibility_timeout=settings.azure_storage_queue_visibility_timeout_seconds,
        )
        return list(messages)

    def delete_message(self, message: Any) -> None:
        """Delete a previously received queue message."""
        self._ensure_client().delete_message(message)

    @staticmethod
    def decode_message(message: Any) -> dict[str, Any]:
        """Decode queue message content into a dispatch payload."""
        content = getattr(message, "content", "")
        if not content:
            raise ValueError("Queue message content is empty")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Queue message payload must be an object")
        return payload


queue_service = QueueService()
