"""Indexing service for GraphRAG operations."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

from ..config import settings
from ..models import IndexJobResponse, IndexStatus, IndexStatusResponse
from ..repositories import (
    INDEX_JOB_CANCELLED,
    INDEX_JOB_COMPLETED,
    INDEX_JOB_FAILED,
    INDEX_JOB_QUEUED,
    INDEX_JOB_RETRYING,
    INDEX_JOB_RUNNING,
    CosmosControlPlaneRepository,
    get_control_plane_repository,
    get_pipeline_output_repository,
)
from ..utils import load_graphrag_config
from ..utils.arrow_fix import apply_arrow_fix, remove_arrow_fix
from .queue_service import queue_service

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for managing indexing operations."""

    def __init__(self):
        self.control_plane = get_control_plane_repository()
        self.pipeline_repo = get_pipeline_output_repository()
        self.queue_service = queue_service
        # Local mode: in-memory job state keyed by collection_id
        self._local_jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _sanitize_error(error: str | None) -> str | None:
        if not error:
            return None
        normalized = " ".join(str(error).split())
        return normalized[:500]

    @staticmethod
    def _next_attempt_at() -> str:
        return (
            (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=settings.azure_storage_queue_visibility_timeout_seconds
                )
            )
            .replace(tzinfo=None)
            .isoformat()
        )

    def _require_control_plane(self) -> CosmosControlPlaneRepository:
        if self.control_plane is None:
            raise RuntimeError(
                "Azure Cosmos DB is required for control-plane metadata in Phase 2. "
                "Configure AZURE_COSMOS_CONNECTION_STRING or "
                "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )
        return self.control_plane

    def _ensure_control_plane_enabled(self) -> None:
        self._require_control_plane()

    def _ensure_dispatch_enabled(self) -> None:
        self._require_control_plane()
        if not self.queue_service.is_configured():
            raise RuntimeError(
                "Azure Storage Queue is required for durable indexing dispatch in Phase 2. "
                "Configure storage connection settings or enable managed identity."
            )

    def _status_to_response(
        self, collection_id: str, job: dict[str, Any]
    ) -> IndexStatusResponse:
        status_map = {
            INDEX_JOB_QUEUED: IndexStatus.PENDING,
            INDEX_JOB_RUNNING: IndexStatus.RUNNING,
            INDEX_JOB_RETRYING: IndexStatus.RETRYING,
            INDEX_JOB_COMPLETED: IndexStatus.COMPLETED,
            INDEX_JOB_FAILED: IndexStatus.FAILED,
            INDEX_JOB_CANCELLED: IndexStatus.CANCELLED,
        }
        response_status = status_map.get(
            str(job.get("status", "")), IndexStatus.PENDING
        )
        progress = float(job.get("progress") or 0.0)
        message = str(job.get("message") or "Indexing job queued")

        if (
            response_status == IndexStatus.COMPLETED
            or response_status == IndexStatus.FAILED
        ):
            progress = 100.0
        elif response_status == IndexStatus.CANCELLED:
            progress = progress or 100.0

        return IndexStatusResponse(
            collection_id=collection_id,
            job_id=str(job["id"]),
            status=response_status,
            progress=progress,
            message=message,
            attempt=int(job.get("attempt", 0)),
            max_attempts=int(
                job.get("maxAttempts", settings.indexing_job_max_attempts)
            ),
            started_at=self._parse_time(job.get("startedAt")),
            completed_at=self._parse_time(job.get("finishedAt")),
            retry_at=self._parse_time(job.get("nextAttemptAt")),
            lease_owner_id=job.get("leaseOwnerId"),
            heartbeat_at=self._parse_time(job.get("heartbeatAt")),
            error=job.get("error"),
        )

    def _job_to_response(self, job: dict[str, Any]) -> IndexJobResponse:
        return IndexJobResponse(
            job_id=str(job["id"]),
            collection_id=str(job["collectionId"]),
            status=str(job.get("status", INDEX_JOB_QUEUED)),
            attempt=int(job.get("attempt", 0)),
            max_attempts=int(
                job.get("maxAttempts", settings.indexing_job_max_attempts)
            ),
            target_version=str(job.get("targetVersion") or ""),
            requested_at=self._parse_time(job.get("requestedAt")),
            started_at=self._parse_time(job.get("startedAt")),
            completed_at=self._parse_time(job.get("finishedAt")),
            retry_at=self._parse_time(job.get("nextAttemptAt")),
            lease_owner_id=job.get("leaseOwnerId"),
            lease_acquired_at=self._parse_time(job.get("leaseAcquiredAt")),
            lease_expires_at=self._parse_time(job.get("leaseExpiresAt")),
            heartbeat_at=self._parse_time(job.get("heartbeatAt")),
            progress=float(job.get("progress") or 0.0),
            message=str(job.get("message") or "Indexing job queued"),
            error=job.get("error"),
        )

    def _local_status_response(self, collection_id: str) -> IndexStatusResponse:
        job = self._local_jobs.get(collection_id, {})
        return IndexStatusResponse(
            collection_id=collection_id,
            job_id=str(job.get("id", "")),
            status=job.get("status", IndexStatus.PENDING),
            progress=float(job.get("progress", 0.0)),
            message=str(job.get("message", "Indexing queued")),
            attempt=0,
            max_attempts=1,
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            error=job.get("error"),
        )

    def _verify_pipeline_output(self, *, collection_id: str, version: str) -> dict[str, int]:
        if self.pipeline_repo is None:
            raise RuntimeError("Pipeline output repository is not configured")

        required_datasets = [
            "entities",
            "relationships",
            "text_units",
            "communities",
            "community_reports",
        ]
        row_counts: dict[str, int] = {}
        for dataset in required_datasets:
            if not self.pipeline_repo.dataset_exists(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            ):
                raise RuntimeError(f"Missing pipeline dataset: {dataset}")
            row_counts[dataset] = self.pipeline_repo.count_rows(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )
        if self.pipeline_repo.dataset_exists(
            collection_id=collection_id,
            version=version,
            dataset="covariates",
        ):
            row_counts["covariates"] = self.pipeline_repo.count_rows(
                collection_id=collection_id,
                version=version,
                dataset="covariates",
            )
        return row_counts

    async def _run_local_indexing(self, collection_id: str, job_id: str) -> None:
        """Execute indexing directly in the background for local mode."""
        job = self._local_jobs[collection_id]
        job.update({"status": IndexStatus.RUNNING, "progress": 10.0, "message": "Loading configuration...", "started_at": datetime.now(timezone.utc).replace(tzinfo=None)})
        try:
            apply_arrow_fix()
            config = load_graphrag_config(collection_id, use_cloud_vectors=False)

            job.update({"progress": 20.0, "message": "Running indexing pipeline..."})
            outputs = await api.build_index(
                config=config,
                verbose=True,
                callbacks=[NoopWorkflowCallbacks()],
            )

            error_messages: list[str] = []
            for output in outputs:
                if output.errors:
                    error_messages.extend([str(err) for err in output.errors])

            if error_messages:
                job.update({
                    "status": IndexStatus.FAILED,
                    "progress": 100.0,
                    "message": "Indexing failed",
                    "error": self._sanitize_error("; ".join(error_messages[:3])),
                    "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
                return

            job.update({"progress": 90.0, "message": "Indexing completed successfully"})
            try:
                from .query_service import query_service
                query_service.invalidate_collection_cache(collection_id)
            except Exception:
                logger.exception("Failed to invalidate cache for collection %s", collection_id)

            job.update({
                "status": IndexStatus.COMPLETED,
                "progress": 100.0,
                "message": "Indexing completed successfully",
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })
        except Exception as err:
            logger.exception("Error during local indexing for %s", collection_id)
            job.update({
                "status": IndexStatus.FAILED,
                "progress": 100.0,
                "message": "Indexing failed",
                "error": self._sanitize_error(str(err)),
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })
        finally:
            remove_arrow_fix()

    async def _run_cosmos_indexing_direct(self, collection_id: str, job_id: str) -> None:
        """Execute indexing directly in cosmos_pipeline mode (no queue required)."""
        control_plane = self._require_control_plane()
        job = self._local_jobs[collection_id]
        job.update({"status": IndexStatus.RUNNING, "progress": 10.0, "message": "Loading configuration...", "started_at": datetime.now(timezone.utc).replace(tzinfo=None)})
        try:
            apply_arrow_fix()
            version = job_id[:16]
            config = load_graphrag_config(collection_id, version=version, use_cloud_vectors=True)

            job.update({"progress": 20.0, "message": "Running indexing pipeline..."})
            outputs = await api.build_index(
                config=config,
                verbose=True,
                callbacks=[NoopWorkflowCallbacks()],
            )

            error_messages: list[str] = []
            for output in outputs:
                if output.errors:
                    error_messages.extend([str(err) for err in output.errors])

            if error_messages:
                job.update({
                    "status": IndexStatus.FAILED,
                    "progress": 100.0,
                    "message": "Indexing failed",
                    "error": self._sanitize_error("; ".join(error_messages[:3])),
                    "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
                return

            job.update({"progress": 80.0, "message": "Verifying pipeline output..."})
            if self.pipeline_repo is not None:
                try:
                    row_counts = await asyncio.to_thread(
                        self._verify_pipeline_output,
                        collection_id=collection_id,
                        version=version,
                    )
                    logger.info("Pipeline output verified: %s", row_counts)
                except Exception as verify_err:
                    logger.warning("Pipeline output verification failed: %s", verify_err)

            job.update({"progress": 90.0, "message": "Publishing active version..."})
            control_plane.set_active_version(collection_id, version)

            try:
                from .query_service import query_service
                query_service.invalidate_collection_cache(collection_id)
            except Exception:
                logger.exception("Failed to invalidate cache for collection %s", collection_id)

            job.update({
                "status": IndexStatus.COMPLETED,
                "progress": 100.0,
                "message": "Indexing completed successfully",
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })
        except Exception as err:
            logger.exception("Error during cosmos direct indexing for %s", collection_id)
            job.update({
                "status": IndexStatus.FAILED,
                "progress": 100.0,
                "message": "Indexing failed",
                "error": self._sanitize_error(str(err)),
                "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })
        finally:
            remove_arrow_fix()

    async def start_indexing(self, collection_id: str) -> IndexStatusResponse:
        """Persist a durable job, dispatch it, and return current status immediately."""
        if self._is_local:
            existing = self._local_jobs.get(collection_id, {})
            if existing.get("status") == IndexStatus.RUNNING:
                return self._local_status_response(collection_id)
            job_id = uuid.uuid4().hex
            self._local_jobs[collection_id] = {
                "id": job_id,
                "status": IndexStatus.PENDING,
                "progress": 0.0,
                "message": "Indexing queued",
            }
            asyncio.create_task(self._run_local_indexing(collection_id, job_id))
            return self._local_status_response(collection_id)

        # cosmos_pipeline without queue: run directly (emulator / local dev)
        if not self.queue_service.is_configured():
            existing = self._local_jobs.get(collection_id, {})
            if existing.get("status") == IndexStatus.RUNNING:
                return self._local_status_response(collection_id)
            job_id = uuid.uuid4().hex
            self._local_jobs[collection_id] = {
                "id": job_id,
                "status": IndexStatus.PENDING,
                "progress": 0.0,
                "message": "Indexing queued (direct cosmos mode)",
            }
            asyncio.create_task(self._run_cosmos_indexing_direct(collection_id, job_id))
            return self._local_status_response(collection_id)

        self._ensure_dispatch_enabled()
        control_plane = self._require_control_plane()

        job, created = control_plane.enqueue_indexing_job(
            collection_id,
            max_attempts=settings.indexing_job_max_attempts,
        )
        if created:
            self.queue_service.send_indexing_job_message(
                job_id=str(job["id"]),
                attempt=int(job.get("attempt", 0)),
            )

        return self._status_to_response(collection_id, job)

    async def execute_indexing_job(
        self, *, collection_id: str, job_id: str, worker_id: str
    ) -> dict[str, Any]:
        """Run one indexing job as the active lease holder."""
        self._ensure_dispatch_enabled()
        control_plane = self._require_control_plane()

        running_job = control_plane.transition_indexing_job(
            collection_id=collection_id,
            job_id=job_id,
            to_status=INDEX_JOB_RUNNING,
            expected_lease_owner=worker_id,
            metadata={"source": "worker", "leaseOwnerId": worker_id},
            progress=5.0,
            message="Starting indexing...",
        )

        target_version = str(running_job.get("targetVersion") or "")

        renew_task = asyncio.create_task(
            self._heartbeat_loop(
                collection_id=collection_id,
                job_id=job_id,
                worker_id=worker_id,
            )
        )

        try:
            apply_arrow_fix()
            control_plane.renew_indexing_job_lease(
                collection_id=collection_id,
                job_id=job_id,
                lease_owner_id=worker_id,
                lease_duration_seconds=settings.indexing_worker_lease_duration_seconds,
                progress=10.0,
                message="Loading configuration...",
            )
            config = load_graphrag_config(
                collection_id,
                version=target_version or None,
                use_cloud_vectors=True,
            )

            control_plane.renew_indexing_job_lease(
                collection_id=collection_id,
                job_id=job_id,
                lease_owner_id=worker_id,
                lease_duration_seconds=settings.indexing_worker_lease_duration_seconds,
                progress=20.0,
                message="Running indexing pipeline...",
            )
            outputs = await api.build_index(
                config=config,
                verbose=True,
                callbacks=[NoopWorkflowCallbacks()],
            )

            error_messages: list[str] = []
            for output in outputs:
                if output.errors:
                    error_messages.extend([str(err) for err in output.errors])

            if error_messages:
                sanitized_error = self._sanitize_error("; ".join(error_messages[:3]))
                return self._handle_retryable_failure(
                    collection_id=collection_id,
                    job_id=job_id,
                    worker_id=worker_id,
                    error=sanitized_error or "Indexing pipeline failed",
                    metadata={"stage": "build_index"},
                )

            control_plane.renew_indexing_job_lease(
                collection_id=collection_id,
                job_id=job_id,
                lease_owner_id=worker_id,
                lease_duration_seconds=settings.indexing_worker_lease_duration_seconds,
                progress=70.0,
                message="Verifying pipeline output...",
            )
            pipeline_counts = await asyncio.to_thread(
                self._verify_pipeline_output,
                collection_id=collection_id,
                version=target_version,
            )
            control_plane.upsert_artifact_manifest(
                collection_id=collection_id,
                version=target_version,
                artifact_name="pipeline-datasets",
                counts=pipeline_counts,
                checksum="storageMode=cosmos_pipeline",
            )
            control_plane.set_active_version(collection_id, target_version)

            try:
                from .query_service import query_service

                query_service.invalidate_collection_cache(collection_id)
                if settings.serving_cache_warm_on_index_complete:
                    await query_service._load_context_from_pipeline(
                        collection_id, "global"
                    )
            except Exception:
                logger.exception(
                    "Failed to invalidate pipeline context cache for collection %s",
                    collection_id,
                )

            completed_job = control_plane.transition_indexing_job(
                collection_id=collection_id,
                job_id=job_id,
                to_status=INDEX_JOB_COMPLETED,
                expected_lease_owner=worker_id,
                metadata={
                    "stage": "build_index",
                    "version": target_version,
                    "pipelineCounts": pipeline_counts,
                    "storageMode": "cosmos_pipeline",
                    "leaseOwnerId": worker_id,
                },
                progress=100.0,
                message="Indexing completed successfully",
            )
            return completed_job
        except Exception as err:
            logger.exception("Error during indexing for %s", collection_id)
            return self._handle_retryable_failure(
                collection_id=collection_id,
                job_id=job_id,
                worker_id=worker_id,
                error=self._sanitize_error(str(err)) or "Indexing failed",
                metadata={"stage": "exception"},
            )
        finally:
            renew_task.cancel()
            try:
                await renew_task
            except asyncio.CancelledError:
                pass
            remove_arrow_fix()

    def _handle_retryable_failure(
        self,
        *,
        collection_id: str,
        job_id: str,
        worker_id: str,
        error: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        control_plane = self._require_control_plane()
        current_job = control_plane.get_indexing_job(collection_id, job_id)
        if current_job is None:
            raise ValueError(f"Indexing job '{job_id}' not found")

        attempt = int(current_job.get("attempt", 0))
        max_attempts = int(
            current_job.get("maxAttempts", settings.indexing_job_max_attempts)
        )
        if attempt < max_attempts:
            next_attempt_at = self._next_attempt_at()
            retrying_job = control_plane.transition_indexing_job(
                collection_id=collection_id,
                job_id=job_id,
                to_status=INDEX_JOB_RETRYING,
                error=error,
                expected_lease_owner=worker_id,
                next_attempt_at=next_attempt_at,
                metadata={
                    **metadata,
                    "reason": "auto-retry",
                    "attempt": attempt,
                    "leaseOwnerId": worker_id,
                },
                progress=0.0,
                message="Retry scheduled",
            )
            self.queue_service.send_indexing_job_message(
                job_id=job_id,
                attempt=attempt,
                visibility_timeout=settings.azure_storage_queue_visibility_timeout_seconds,
            )
            return retrying_job

        return control_plane.transition_indexing_job(
            collection_id=collection_id,
            job_id=job_id,
            to_status=INDEX_JOB_FAILED,
            error=error,
            expected_lease_owner=worker_id,
            metadata={
                **metadata,
                "reason": "attempts-exhausted",
                "leaseOwnerId": worker_id,
            },
            progress=100.0,
            message="Indexing failed",
        )

    async def _heartbeat_loop(
        self, *, collection_id: str, job_id: str, worker_id: str
    ) -> None:
        """Renew lease ownership while a worker is actively processing a job."""
        control_plane = self._require_control_plane()
        while True:
            await asyncio.sleep(settings.indexing_worker_heartbeat_interval_seconds)
            renewed = control_plane.renew_indexing_job_lease(
                collection_id=collection_id,
                job_id=job_id,
                lease_owner_id=worker_id,
                lease_duration_seconds=settings.indexing_worker_lease_duration_seconds,
            )
            if renewed is None:
                logger.warning(
                    "Stopped heartbeating indexing job %s because the lease was lost",
                    job_id,
                )
                return

    def requeue_recoverable_jobs(self) -> int:
        """Re-dispatch queued, retrying, or expired-lease jobs."""
        self._ensure_dispatch_enabled()
        control_plane = self._require_control_plane()
        recoverable_jobs = control_plane.list_recoverable_indexing_jobs()
        for job in recoverable_jobs:
            self.queue_service.send_indexing_job_message(
                job_id=str(job["id"]),
                attempt=int(job.get("attempt", 0)),
            )
        return len(recoverable_jobs)

    def get_index_status(self, collection_id: str) -> IndexStatusResponse | None:
        """Get the latest indexing status for a collection."""
        if self._is_local or not self.queue_service.is_configured():
            if collection_id not in self._local_jobs:
                return None
            return self._local_status_response(collection_id)
        control_plane = self._require_control_plane()
        latest_job = control_plane.get_latest_indexing_job(collection_id)
        if latest_job is None:
            return None
        return self._status_to_response(collection_id, latest_job)

    def get_job_status(self, job_id: str) -> IndexJobResponse | None:
        """Return canonical status for one job id from Cosmos."""
        control_plane = self._require_control_plane()
        job = control_plane.get_indexing_job_by_id(job_id)
        if job is None:
            return None
        return self._job_to_response(job)


indexing_service = IndexingService()
