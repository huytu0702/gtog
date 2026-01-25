"""Adapter to persist GraphRAG outputs to database."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Mapping, Any
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

    def _build_rows(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        items: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach collection/index run columns to payloads."""
        rows: list[dict[str, Any]] = []
        for item in items:
            payload = {key: value for key, value in item.items() if key != "id"}
            payload["collection_id"] = collection_id
            payload["index_run_id"] = index_run_id
            rows.append(payload)
        return rows

    async def insert_entities(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        entities: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert entity payloads via repository."""
        rows = self._build_rows(collection_id, index_run_id, entities)
        if rows:
            await self._entities.bulk_insert(rows)

    async def insert_relationships(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        relationships: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert relationship payloads."""
        rows = self._build_rows(collection_id, index_run_id, relationships)
        if rows:
            await self._relationships.bulk_insert(rows)

    async def insert_communities(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        communities: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert community payloads."""
        rows = self._build_rows(collection_id, index_run_id, communities)
        if rows:
            await self._communities.bulk_insert(rows)

    async def insert_community_reports(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        reports: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert community report payloads."""
        rows = self._build_rows(collection_id, index_run_id, reports)
        if rows:
            await self._community_reports.bulk_insert(rows)

    async def insert_text_units(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        text_units: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert text unit payloads."""
        rows = self._build_rows(collection_id, index_run_id, text_units)
        if rows:
            await self._text_units.bulk_insert(rows)

    async def insert_covariates(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        covariates: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert covariate payloads."""
        rows = self._build_rows(collection_id, index_run_id, covariates)
        if rows:
            await self._covariates.bulk_insert(rows)

    async def insert_embeddings(
        self,
        collection_id: UUID,
        index_run_id: UUID,
        embeddings: Sequence[Mapping[str, object]],
    ) -> None:
        """Insert embedding payloads."""
        rows = self._build_rows(collection_id, index_run_id, embeddings)
        if rows:
            await self._embeddings.bulk_insert(rows)
