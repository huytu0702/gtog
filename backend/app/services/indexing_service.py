"""Indexing service for GraphRAG operations."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

from ..models import IndexStatus, IndexStatusResponse
from ..repositories import (
    INDEX_JOB_COMPLETED,
    INDEX_JOB_FAILED,
    INDEX_JOB_QUEUED,
    INDEX_JOB_RUNNING,
    get_control_plane_repository,
)
from ..utils import load_graphrag_config
from ..utils.arrow_fix import apply_arrow_fix, remove_arrow_fix
from .serving_materialization_service import serving_materialization_service

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for managing indexing operations."""

    def __init__(self):
        """Initialize the indexing service."""
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.runtime_progress: Dict[str, Dict[str, Any]] = {}
        self.control_plane = get_control_plane_repository()

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def _ensure_control_plane_enabled(self) -> None:
        if self.control_plane is None:
            raise RuntimeError(
                "Azure Cosmos DB is required for control-plane metadata in Phase 1. "
                "Configure AZURE_COSMOS_CONNECTION_STRING or "
                "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY."
            )

    def _status_to_response(self, collection_id: str, job: dict[str, Any]) -> IndexStatusResponse:
        status_map = {
            INDEX_JOB_QUEUED: IndexStatus.PENDING,
            INDEX_JOB_RUNNING: IndexStatus.RUNNING,
            INDEX_JOB_COMPLETED: IndexStatus.COMPLETED,
            INDEX_JOB_FAILED: IndexStatus.FAILED,
        }
        response_status = status_map.get(str(job["status"]), IndexStatus.PENDING)

        progress = 0.0
        message = "Indexing job queued"
        if response_status == IndexStatus.RUNNING:
            runtime = self.runtime_progress.get(job["id"], {})
            progress = float(runtime.get("progress", 10.0))
            message = str(runtime.get("message", "Running indexing pipeline..."))
        elif response_status == IndexStatus.COMPLETED:
            progress = 100.0
            message = "Indexing completed successfully"
        elif response_status == IndexStatus.FAILED:
            progress = 100.0
            message = "Indexing failed"

        return IndexStatusResponse(
            collection_id=collection_id,
            status=response_status,
            progress=progress,
            message=message,
            started_at=self._parse_time(job.get("startedAt")),
            completed_at=self._parse_time(job.get("finishedAt")),
            error=job.get("error"),
        )

    def _set_runtime_progress(self, job_id: str, progress: float, message: str) -> None:
        self.runtime_progress[job_id] = {
            "progress": progress,
            "message": message,
        }

    def _schedule_retry_if_possible(self, collection_id: str, failed_job: dict[str, Any]) -> None:
        attempt = int(failed_job.get("attempt", 0))
        max_attempts = int(failed_job.get("maxAttempts", 0))
        if attempt >= max_attempts:
            return

        job_id = str(failed_job["id"])
        self.control_plane.transition_indexing_job(
            collection_id=collection_id,
            job_id=job_id,
            to_status=INDEX_JOB_QUEUED,
            metadata={"reason": "auto-retry", "attempt": attempt},
        )
        self.running_tasks[collection_id] = asyncio.create_task(
            self._run_indexing_task(collection_id=collection_id, job_id=job_id)
        )

    async def start_indexing(self, collection_id: str) -> IndexStatusResponse:
        """
        Start indexing a collection in the background.

        Args:
            collection_id: The collection identifier

        Returns:
            IndexStatusResponse with initial status
        """
        self._ensure_control_plane_enabled()

        job, created = self.control_plane.enqueue_indexing_job(collection_id, max_attempts=3)
        current_task = self.running_tasks.get(collection_id)
        task_running = current_task is not None and not current_task.done()

        if created or (str(job["status"]) == INDEX_JOB_QUEUED and not task_running):
            self.running_tasks[collection_id] = asyncio.create_task(
                self._run_indexing_task(collection_id=collection_id, job_id=str(job["id"]))
            )

        return self._status_to_response(collection_id, job)

    async def _run_indexing_task(self, collection_id: str, job_id: str) -> None:
        """
        Internal task for running the indexing process.

        Args:
            collection_id: The collection identifier
            job_id: Cosmos job identifier
        """
        self._ensure_control_plane_enabled()

        try:
            running_job = self.control_plane.transition_indexing_job(
                collection_id=collection_id,
                job_id=job_id,
                to_status=INDEX_JOB_RUNNING,
                metadata={"source": "api"},
            )
            target_version = str(running_job.get("targetVersion") or "")
            if not target_version:
                target_version = f"v{uuid4().hex[:12]}"
            self._set_runtime_progress(job_id, 5.0, "Starting indexing...")

            # Apply ArrowStringArray fix before indexing.
            apply_arrow_fix()
            logger.info(f"Starting indexing for collection: {collection_id}")

            self._set_runtime_progress(job_id, 10.0, "Loading configuration...")
            config = load_graphrag_config(
                collection_id,
                version=target_version if target_version else None,
            )
            logger.info(f"Configuration loaded for {collection_id}")

            self._set_runtime_progress(job_id, 20.0, "Running indexing pipeline...")
            outputs = await api.build_index(
                config=config,
                verbose=True,
                callbacks=[NoopWorkflowCallbacks()],
            )

            has_errors = any(output.errors and len(output.errors) > 0 for output in outputs)
            if has_errors:
                error_messages = []
                for output in outputs:
                    if output.errors:
                        error_messages.extend([str(err) for err in output.errors])

                joined_errors = "; ".join(error_messages[:3])
                failed_job = self.control_plane.transition_indexing_job(
                    collection_id=collection_id,
                    job_id=job_id,
                    to_status=INDEX_JOB_FAILED,
                    error=joined_errors,
                    metadata={"stage": "build_index"},
                )
                self._set_runtime_progress(job_id, 100.0, "Indexing failed")
                logger.error(f"Indexing failed for {collection_id}: {joined_errors}")
                self._schedule_retry_if_possible(collection_id, failed_job)
            else:
                self._set_runtime_progress(job_id, 70.0, "Materializing serving context...")
                materialized_counts = serving_materialization_service.materialize_collection_version(
                    collection_id=collection_id,
                    version=target_version,
                )
                self.control_plane.set_active_version(collection_id, target_version)
                self.control_plane.transition_indexing_job(
                    collection_id=collection_id,
                    job_id=job_id,
                    to_status=INDEX_JOB_COMPLETED,
                    metadata={
                        "stage": "build_index",
                        "version": target_version,
                        "materializedCounts": materialized_counts,
                    },
                )
                self._set_runtime_progress(job_id, 100.0, "Indexing completed successfully")
                logger.info(f"Indexing completed successfully for {collection_id}")

        except Exception as err:
            logger.exception(f"Error during indexing for {collection_id}")
            try:
                failed_job = self.control_plane.transition_indexing_job(
                    collection_id=collection_id,
                    job_id=job_id,
                    to_status=INDEX_JOB_FAILED,
                    error=str(err),
                    metadata={"stage": "exception"},
                )
                self._schedule_retry_if_possible(collection_id, failed_job)
            except Exception:
                logger.exception("Failed to transition indexing job to failed state")
            self._set_runtime_progress(job_id, 100.0, "Indexing failed with error")
        finally:
            # Always remove the patch after indexing.
            remove_arrow_fix()
            task = self.running_tasks.get(collection_id)
            if task is asyncio.current_task():
                self.running_tasks.pop(collection_id, None)

    def get_index_status(self, collection_id: str) -> Optional[IndexStatusResponse]:
        """
        Get the current indexing status for a collection.

        Args:
            collection_id: The collection identifier

        Returns:
            IndexStatusResponse or None if never indexed
        """
        self._ensure_control_plane_enabled()
        latest_job = self.control_plane.get_latest_indexing_job(collection_id)
        if latest_job is None:
            return None
        return self._status_to_response(collection_id, latest_job)


# Global indexing service instance
indexing_service = IndexingService()
