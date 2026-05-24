"""Dedicated indexing worker runtime."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from .config import settings
from .logging_config import configure_logging
from .services.indexing_service import indexing_service
from .services.queue_service import queue_service
from .vector_stores import register_backend_vector_stores

logger = logging.getLogger(__name__)


async def _run_worker_loop() -> None:
    worker_id = uuid4().hex
    logger.info("Starting indexing worker with id=%s", worker_id)
    indexing_service._ensure_dispatch_enabled()
    control_plane = indexing_service._require_control_plane()
    queue_service.ensure_queue()

    while True:
        recovered_jobs = indexing_service.requeue_recoverable_jobs()
        if recovered_jobs:
            logger.info("Re-dispatched %s recoverable indexing jobs", recovered_jobs)

        messages = queue_service.receive_messages()
        if not messages:
            await asyncio.sleep(settings.azure_storage_queue_poll_interval_seconds)
            continue

        for message in messages:
            try:
                payload = queue_service.decode_message(message)
            except Exception:
                logger.exception(
                    "Failed to decode queue message; deleting invalid payload"
                )
                queue_service.delete_message(message)
                continue

            job_id = str(payload.get("job_id") or "")
            if not job_id:
                logger.warning("Discarding queue message without job_id")
                queue_service.delete_message(message)
                continue

            job = control_plane.get_indexing_job_by_id(job_id)
            if job is None:
                logger.info("Deleting dispatch for missing job %s", job_id)
                queue_service.delete_message(message)
                continue

            if str(job.get("status")) in {"completed", "failed", "cancelled"}:
                logger.info("Deleting stale dispatch for terminal job %s", job_id)
                queue_service.delete_message(message)
                continue

            leased_job = control_plane.acquire_indexing_job_lease(
                collection_id=str(job["collectionId"]),
                job_id=job_id,
                lease_owner_id=worker_id,
                lease_duration_seconds=settings.indexing_worker_lease_duration_seconds,
            )
            if leased_job is None:
                logger.info(
                    "Skipping dispatch for job %s because another worker owns the lease",
                    job_id,
                )
                continue

            await indexing_service.execute_indexing_job(
                collection_id=str(job["collectionId"]),
                job_id=job_id,
                worker_id=worker_id,
            )
            queue_service.delete_message(message)


def main() -> None:
    configure_logging()
    register_backend_vector_stores()
    asyncio.run(_run_worker_loop())


if __name__ == "__main__":
    main()
