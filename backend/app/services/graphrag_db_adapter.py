"""Adapter to persist GraphRAG outputs to database."""

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


class GraphRAGDbAdapter:
    """Persist GraphRAG output artifacts into SQL tables."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def ingest_outputs(self, collection_id: UUID, index_run_id: UUID, outputs: Iterable[object]) -> None:
        """Persist GraphRAG outputs to database (placeholder)."""
        # Implementation will map outputs to models (entities, relationships, etc.)
        return
