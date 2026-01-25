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
        # Placeholder: actual implementation will load db tables
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.GLOBAL)

    async def local_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.LOCAL)

    async def tog_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.TOG)

    async def drift_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.DRIFT)
