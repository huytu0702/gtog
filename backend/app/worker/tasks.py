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

    Returns
    -------
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

    Returns
    -------
        Result dict with status
    """
    from app.db.models import IndexRun, IndexRunStatus
    from app.db.session import get_session

    logger.info("Starting indexing for collection %s, run %s", collection_id, index_run_id)

    async with get_session() as session:
        # Get index run
        from sqlalchemy import select
        result = await session.execute(
            select(IndexRun).where(IndexRun.id == index_run_id)
        )
        index_run = result.scalar_one_or_none()

        if not index_run:
            logger.error("Index run %s not found", index_run_id)
            return {"status": "error", "message": "Index run not found"}

        try:
            # Update status to running
            index_run.status = IndexRunStatus.RUNNING
            index_run.started_at = datetime.now()
            await session.commit()

            # Run GraphRAG indexing pipeline
            from app.services.graphrag_db_adapter import GraphRAGDbAdapter
            adapter = GraphRAGDbAdapter(session)
            outputs = await _run_graphrag_pipeline(session, collection_id, index_run_id)
            await adapter.ingest_outputs(collection_id, index_run_id, outputs)

            # Mark as completed
            index_run.status = IndexRunStatus.COMPLETED
            index_run.finished_at = datetime.now()
            await session.commit()

            logger.info("Indexing completed for collection %s", collection_id)
            return {"status": "completed", "index_run_id": str(index_run_id)}

        except Exception as e:
            logger.exception("Indexing failed for collection %s", collection_id)
            index_run.status = IndexRunStatus.FAILED
            index_run.finished_at = datetime.now()
            index_run.error = str(e)
            await session.commit()
            return {"status": "failed", "error": str(e)}


async def _run_graphrag_pipeline(
    session,
    collection_id: UUID,
    index_run_id: UUID,
):
    if "_fake_build_index" in globals():
        return await globals()["_fake_build_index"]()

    import graphrag.api as api
    from app.db.models import Document
    from sqlalchemy import select

    result = await session.execute(
        select(Document).where(Document.collection_id == collection_id)
    )
    documents = result.scalars().all()

    # Write temp files for GraphRAG input
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            filename = doc.filename or f"{doc.id}.txt"
            path = input_dir / filename
            path.write_bytes(doc.bytes_content or b"")

        return await api.build_index(str(input_dir))
