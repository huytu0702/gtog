"""Collection repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Document, IndexRun, IndexRunStatus
from .base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    """Repository for collection operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Collection)

    async def get_by_name(self, name: str) -> Optional[Collection]:
        """Get collection by name."""
        result = await self.session.execute(
            select(Collection).where(Collection.name == name)
        )
        return result.scalar_one_or_none()

    async def get_with_document_count(self, collection_id: UUID) -> Optional[tuple]:
        """Get collection with document count."""
        result = await self.session.execute(
            select(
                Collection,
                func.count(Document.id).label("document_count")
            )
            .outerjoin(Document, Collection.id == Document.collection_id)
            .where(Collection.id == collection_id)
            .group_by(Collection.id)
        )
        row = result.first()
        if row:
            return row[0], row[1]
        return None

    async def get_latest_completed_run(self, collection_id: UUID) -> Optional[IndexRun]:
        """Get the latest completed index run for a collection."""
        result = await self.session.execute(
            select(IndexRun)
            .where(IndexRun.collection_id == collection_id)
            .where(IndexRun.status == IndexRunStatus.COMPLETED)
            .order_by(IndexRun.finished_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def is_indexed(self, collection_id: UUID) -> bool:
        """Check if collection has a completed index run."""
        run = await self.get_latest_completed_run(collection_id)
        return run is not None
