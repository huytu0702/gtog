"""Adapter to persist GraphRAG outputs to database."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Mapping
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import (
    CommunityReportRepository,
    CommunityRepository,
    CovariateRepository,
    EmbeddingRepository,
    EntityRepository,
    RelationshipRepository,
    TextUnitRepository,
)


class GraphRAGDbAdapter:
    """Persist GraphRAG output artifacts into SQL tables."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._entities = EntityRepository(session)
        self._relationships = RelationshipRepository(session)
        self._communities = CommunityRepository(session)
        self._community_reports = CommunityReportRepository(session)
        self._text_units = TextUnitRepository(session)
        self._covariates = CovariateRepository(session)
        self._embeddings = EmbeddingRepository(session)

    async def ingest_outputs(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        outputs: Iterable[object],
    ) -> None:
        """Persist GraphRAG outputs to database (placeholder)."""
        return

    async def insert_entities(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        entities: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert entity payloads via repository."""
        rows = [
            {
                "collection_id": collection_id,
                "index_run_id": index_run_id,
                **{key: value for key, value in entity.items() if key != "id"},
            }
            for entity in entities
        ]
        if rows:
            await self._entities.bulk_insert(rows)
