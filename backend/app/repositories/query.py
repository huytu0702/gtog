"""Query repository for GraphRAG data."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRun, IndexRunStatus


class QueryRepository:
    """Repository for query-time data access."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_run_id(self, collection_id: UUID) -> UUID | None:
        result = await self.session.execute(
            select(IndexRun.id)
            .where(IndexRun.collection_id == collection_id)
            .where(IndexRun.status == IndexRunStatus.COMPLETED)
            .order_by(IndexRun.finished_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_latest_run_data(self, collection_id: UUID):
        """Get latest completed run data for a collection."""
        run_id = await self.get_latest_run_id(collection_id)
        if not run_id:
            return None
        return run_id
