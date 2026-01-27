"""Database-backed query service for GraphRAG search."""

import logging
from uuid import UUID

from app.models import SearchMethod, SearchResponse
from app.repositories import QueryRepository

logger = logging.getLogger(__name__)


class QueryServiceDB:
    """Service for managing query/search operations using database."""

    def __init__(self, session):
        self.session = session
        self.repo = QueryRepository(session)

    async def _load_run_data(self, collection_id: UUID):
        """Load data for the latest completed run."""
        return await self.repo.get_latest_run_data(collection_id)

    async def global_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        """Perform global search using community reports from Postgres."""
        run_data = await self._load_run_data(collection_id)
        if not run_data:
            return SearchResponse(
                query=query,
                response="No indexed data found for this collection.",
                context_data=None,
                method=SearchMethod.GLOBAL,
            )

        reports = run_data.get("community_reports", [])
        if not reports:
            return SearchResponse(
                query=query,
                response="No community reports available.",
                context_data={"entities_count": len(run_data.get("entities", []))},
                method=SearchMethod.GLOBAL,
            )

        # Build minimal response from available data
        summaries = [r.get("summary", "") for r in reports if r.get("summary")]
        context = "\n\n".join(summaries[:5])  # Limit to top 5 reports

        return SearchResponse(
            query=query,
            response=f"Found {len(reports)} communities. "
                     f"Top communities: {', '.join(r.get('title', 'Unknown') or 'Unknown' for r in reports[:3])}",
            context_data={"community_count": len(reports)},
            method=SearchMethod.GLOBAL,
        )

    async def local_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        """Perform local search using entities from Postgres."""
        run_data = await self._load_run_data(collection_id)
        if not run_data:
            return SearchResponse(
                query=query,
                response="No indexed data found for this collection.",
                context_data=None,
                method=SearchMethod.LOCAL,
            )

        entities = run_data.get("entities", [])
        if not entities:
            return SearchResponse(
                query=query,
                response="No entities found in the knowledge graph.",
                context_data=None,
                method=SearchMethod.LOCAL,
            )

        # Simple entity matching (minimal implementation)
        matching_entities = [
            e for e in entities
            if query.lower() in (e.get("title", "") or "").lower()
            or query.lower() in (e.get("description", "") or "").lower()
        ]

        if matching_entities:
            entity_names = ", ".join(e.get("title", "Unknown") or "Unknown" for e in matching_entities[:3])
            return SearchResponse(
                query=query,
                response=f"Found {len(matching_entities)} matching entities: {entity_names}",
                context_data={"entity_count": len(matching_entities)},
                method=SearchMethod.LOCAL,
            )

        return SearchResponse(
            query=query,
            response=f"Knowledge graph contains {len(entities)} entities. "
                     f"No direct matches found for '{query}'.",
            context_data={"entity_count": len(entities)},
            method=SearchMethod.LOCAL,
        )

    async def tog_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        """Perform ToG search using entities and relationships from Postgres."""
        run_data = await self._load_run_data(collection_id)
        if not run_data:
            return SearchResponse(
                query=query,
                response="No indexed data found for this collection.",
                context_data=None,
                method=SearchMethod.TOG,
            )

        entities = run_data.get("entities", [])
        relationships = run_data.get("relationships", [])

        if not entities:
            return SearchResponse(
                query=query,
                response="No entities found for ToG exploration.",
                context_data=None,
                method=SearchMethod.TOG,
            )

        return SearchResponse(
            query=query,
            response=f"ToG search would explore {len(entities)} entities and "
                     f"{len(relationships)} relationships.",
            context_data={
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
            method=SearchMethod.TOG,
        )

    async def drift_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        """Perform DRIFT search using entities and text units from Postgres."""
        run_data = await self._load_run_data(collection_id)
        if not run_data:
            return SearchResponse(
                query=query,
                response="No indexed data found for this collection.",
                context_data=None,
                method=SearchMethod.DRIFT,
            )

        entities = run_data.get("entities", [])

        return SearchResponse(
            query=query,
            response=f"DRIFT search over {len(entities)} entities.",
            context_data={"entity_count": len(entities)},
            method=SearchMethod.DRIFT,
        )
