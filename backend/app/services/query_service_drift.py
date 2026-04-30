"""DRIFT search handler."""

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


async def run_drift_search(
    *,
    collection_id: str,
    query: str,
    community_level: int,
    response_type: str,
    load_context,  # callable: async (collection_id, method) -> (version, frames)
) -> SearchResponse:
    """Execute a DRIFT search and return a SearchResponse.

    Args:
        collection_id: The collection identifier.
        query: The search query.
        community_level: Community level to search.
        response_type: Type of response format.
        load_context: Async callable that returns (active_version, frames).

    Returns
    -------
        SearchResponse with results.
    """
    active_version, frames = await load_context(collection_id, "drift")
    config = load_graphrag_config(
        collection_id, version=active_version, use_cloud_vectors=True
    )
    entities = frames["entities"]
    communities = frames["communities"]
    community_reports = frames["community_reports"]
    text_units = frames["text_units"]
    relationships = frames["relationships"]

    fh = _attach_query_log(collection_id)
    try:
        logger.info("DRIFT search for collection %s: %s", collection_id, query)

        response_text, context_data = await api.drift_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            community_level=community_level,
            response_type=response_type,
            query=query,
        )

        logger.info("DRIFT search completed for collection %s", collection_id)
    finally:
        _detach_query_log(fh)

    return SearchResponse(
        query=query,
        response=response_text,
        context_data=_serialize_context_records(context_data),
        method=SearchMethod.DRIFT,
    )
