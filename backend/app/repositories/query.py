"""Query repository for GraphRAG data."""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Entity, Relationship, Community, CommunityReport, IndexRun, IndexRunStatus


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
        run_id = result.scalar_one_or_none()
        return UUID(str(run_id)) if run_id else None

    async def get_latest_run_data(self, collection_id: UUID) -> dict[str, Any] | None:
        """Get latest completed run data for a collection."""
        run_id = await self.get_latest_run_id(collection_id)
        if not run_id:
            return None

        return {
            "run_id": run_id,
            "entities": await self._get_entities(collection_id, run_id),
            "relationships": await self._get_relationships(collection_id, run_id),
            "communities": await self._get_communities(collection_id, run_id),
            "community_reports": await self._get_community_reports(collection_id, run_id),
        }

    async def _get_entities(self, collection_id: UUID, run_id: UUID) -> list[dict[str, Any]]:
        """Fetch entities for a run."""
        result = await self.session.execute(
            select(Entity).where(
                Entity.collection_id == collection_id,
                Entity.index_run_id == run_id,
            )
        )
        entities = result.scalars().all()
        return [
            {
                "id": e.id,
                "title": e.title,
                "type": e.type,
                "description": e.description,
            }
            for e in entities
        ]

    async def _get_relationships(self, collection_id: UUID, run_id: UUID) -> list[dict[str, Any]]:
        """Fetch relationships for a run."""
        result = await self.session.execute(
            select(Relationship).where(
                Relationship.collection_id == collection_id,
                Relationship.index_run_id == run_id,
            )
        )
        relationships = result.scalars().all()
        return [
            {
                "id": r.id,
                "source": r.source,
                "target": r.target,
                "description": r.description,
                "weight": r.weight,
            }
            for r in relationships
        ]

    async def _get_communities(self, collection_id: UUID, run_id: UUID) -> list[dict[str, Any]]:
        """Fetch communities for a run."""
        result = await self.session.execute(
            select(Community).where(
                Community.collection_id == collection_id,
                Community.index_run_id == run_id,
            )
        )
        communities = result.scalars().all()
        return [
            {
                "id": c.id,
                "community": c.community,
                "title": c.title,
                "level": c.level,
                "size": c.size,
            }
            for c in communities
        ]

    async def _get_community_reports(self, collection_id: UUID, run_id: UUID) -> list[dict[str, Any]]:
        """Fetch community reports for a run."""
        result = await self.session.execute(
            select(CommunityReport).where(
                CommunityReport.collection_id == collection_id,
                CommunityReport.index_run_id == run_id,
            )
        )
        reports = result.scalars().all()
        return [
            {
                "id": r.id,
                "community": r.community,
                "title": r.title,
                "summary": r.summary,
                "full_content": r.full_content,
                "rank": r.rank,
                "level": r.level,
            }
            for r in reports
        ]
