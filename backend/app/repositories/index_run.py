"""Index run repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRun, IndexRunStatus
from .base import BaseRepository


class IndexRunRepository(BaseRepository[IndexRun]):
    """Repository for index run operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, IndexRun)

    async def get_latest_for_collection(self, collection_id: UUID) -> Optional[IndexRun]:
        """Get latest index run for a collection."""
        result = await self.session.execute(
            select(IndexRun)
            .where(IndexRun.collection_id == collection_id)
            .order_by(IndexRun.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_run(self, collection_id: UUID) -> IndexRun:
        """Create a new queued index run for a collection."""
        run = IndexRun(collection_id=collection_id, status=IndexRunStatus.QUEUED)
        await self.create(run)
        return run
