"""Worker tasks for background processing."""

import asyncio
import logging
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


def run_indexing_task(collection_id: str, index_run_id: str) -> dict:
    """
    Run indexing pipeline as a background task.

    This is called by the RQ worker.

    Args:
        collection_id: Collection UUID as string
        index_run_id: Index run UUID as string

    Returns:
        Result dict with status
    """
    return asyncio.run(_run_indexing_async(
        UUID(collection_id),
        UUID(index_run_id)
    ))


async def _run_indexing_async(collection_id: UUID, index_run_id: UUID) -> dict:
    """
    Async implementation of indexing task.

    Args:
        collection_id: Collection UUID
        index_run_id: Index run UUID

    Returns:
        Result dict with status
    """
    from app.db.session import get_session
    from app.db.models import IndexRun, IndexRunStatus

    logger.info(f"Starting indexing for collection {collection_id}, run {index_run_id}")

    async with get_session() as session:
        # Get index run
        from sqlalchemy import select
        result = await session.execute(
            select(IndexRun).where(IndexRun.id == index_run_id)
        )
        index_run = result.scalar_one_or_none()

        if not index_run:
            logger.error(f"Index run {index_run_id} not found")
            return {"status": "error", "message": "Index run not found"}

        try:
            # Update status to running
            index_run.status = IndexRunStatus.RUNNING
            index_run.started_at = datetime.now()
            await session.commit()

            # TODO: Run actual GraphRAG indexing pipeline
            # This will be implemented in Phase 7

            # For now, mark as completed
            index_run.status = IndexRunStatus.COMPLETED
            index_run.finished_at = datetime.now()
            await session.commit()

            logger.info(f"Indexing completed for collection {collection_id}")
            return {"status": "completed", "index_run_id": str(index_run_id)}

        except Exception as e:
            logger.exception(f"Indexing failed for collection {collection_id}")
            index_run.status = IndexRunStatus.FAILED
            index_run.finished_at = datetime.now()
            index_run.error = str(e)
            await session.commit()
            return {"status": "failed", "error": str(e)}
