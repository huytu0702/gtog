"""Database-backed indexing service."""

from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRunStatus
from app.models import IndexStatus, IndexStatusResponse
from app.repositories import CollectionRepository, IndexRunRepository
from app.worker.queue import enqueue_indexing_job


class IndexingServiceDB:
    """Service for managing indexing operations using database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.collections = CollectionRepository(session)
        self.runs = IndexRunRepository(session)

    async def start_indexing(self, collection_id: UUID) -> IndexStatusResponse:
        collection = await self.collections.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection '{collection_id}' not found")

        run = await self.runs.create_run(collection_id)
        job_id = enqueue_indexing_job(collection_id, run.id)

        return IndexStatusResponse(
            collection_id=str(collection_id),
            status=IndexStatus.RUNNING,
            progress=0.0,
            message=f"Queued job {job_id}",
            started_at=datetime.now(),
        )

    async def get_index_status(self, collection_id: UUID) -> IndexStatusResponse | None:
        run = await self.runs.get_latest_for_collection(collection_id)
        if not run:
            return None

        status_map = {
            IndexRunStatus.QUEUED: IndexStatus.RUNNING,
            IndexRunStatus.RUNNING: IndexStatus.RUNNING,
            IndexRunStatus.COMPLETED: IndexStatus.COMPLETED,
            IndexRunStatus.FAILED: IndexStatus.FAILED,
        }
        return IndexStatusResponse(
            collection_id=str(collection_id),
            status=status_map[run.status],
            progress=100.0 if run.status == IndexRunStatus.COMPLETED else 0.0,
            message=run.error or "",
            started_at=run.started_at,
            completed_at=run.finished_at,
        )
