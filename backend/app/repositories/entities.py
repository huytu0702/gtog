"""Repository for GraphRAG entities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Entity
from .base import BaseRepository


class EntityRepository(BaseRepository[Entity]):
    """Persistence helpers for Entity records."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Entity)

    async def bulk_insert(self, payloads: Sequence[Mapping[str, Any]]) -> None:
        """Insert many entity mappings in a single batch."""
        rows = [dict(payload) for payload in payloads]
        if not rows:
            return

        def _bulk_insert(sync_session):
            sync_session.bulk_insert_mappings(Entity, rows)

        await self.session.run_sync(_bulk_insert)
        await self.session.commit()
