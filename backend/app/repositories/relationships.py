"""Repository for GraphRAG relationships."""

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Relationship
from .base import BaseRepository


class RelationshipRepository(BaseRepository[Relationship]):
    """Entity relationship repository."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Relationship)

    async def bulk_insert(self, payloads: Sequence[Mapping[str, Any]]) -> None:
        """Bulk insert relationship rows."""
        rows = [dict(payload) for payload in payloads]
        if not rows:
            return

        def _bulk_insert(sync_session):
            sync_session.bulk_insert_mappings(Relationship, rows)

        await self.session.run_sync(_bulk_insert)
        await self.session.commit()
