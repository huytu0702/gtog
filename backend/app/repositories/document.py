"""Document repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from .base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for document operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Document)

    async def get_by_collection(self, collection_id: UUID) -> list[Document]:
        """Get documents for a collection."""
        result = await self.session.execute(
            select(Document).where(Document.collection_id == collection_id)
        )
        return list(result.scalars().all())

    async def get_by_name(self, collection_id: UUID, name: str) -> Optional[Document]:
        """Get document by filename within a collection."""
        result = await self.session.execute(
            select(Document)
            .where(Document.collection_id == collection_id)
            .where(Document.filename == name)
        )
        return result.scalar_one_or_none()

    async def delete_by_name(self, collection_id: UUID, name: str) -> bool:
        """Delete document by filename within a collection."""
        doc = await self.get_by_name(collection_id, name)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.flush()
        return True
