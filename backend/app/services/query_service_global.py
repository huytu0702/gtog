"""Global search handler."""

import logging

import graphrag.api as api

from ..models import SearchMethod, SearchResponse
from ..utils import load_graphrag_config
from .query_service_base import (
    _attach_query_log,
    _detach_query_log,
    _serialize_context_records,
)

logger = logging.getLogger(__name__)


async def run_global_search(
    *,
    collection_id: str,
    query: str,
    community_level: int | None,
    dynamic_community_selection: bool,
    response_type: str,
    load_context,  # callable: async (collection_id, method) -> (version, frames)
) -> SearchResponse:
    """Execute a global search and return a SearchResponse.

    Args:
        collection_id: The collection identifier.
        query: The search query.
        community_level: Community level to search (None = default).
        dynamic_community_selection: Whether to use dynamic community selection.
        response_type: Type of response format.
        load_context: Async callable that returns (active_version, frames).

    Returns
    -------
        SearchResponse with results.
    """
    active_version, frames = await load_context(collection_id, "global")
    config = load_graphrag_config(
        collection_id, version=active_version, use_cloud_vectors=True
    )
    entities = frames["entities"]
    communities = frames["communities"]
    community_reports = frames["community_reports"]

    fh = _attach_query_log(collection_id)
    try:
        logger.info("Global search for collection %s: %s", collection_id, query)

        response_text, context_data = await api.global_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            query=query,
        )

        logger.info("Global search completed for collection %s", collection_id)
    finally:
        _detach_query_log(fh)

    return SearchResponse(
        query=query,
        response=response_text,
        context_data=_serialize_context_records(context_data),
        method=SearchMethod.GLOBAL,
    )
