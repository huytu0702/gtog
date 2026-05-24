"""ToG (Think-on-Graph) search handler."""

import logging

import graphrag.api as api

from ..models import SearchMethod, SearchResponse
from ..utils import load_graphrag_config
from .query_service_base import (
    _attach_query_log,
    _build_tog_serialized_context,
    _detach_query_log,
    _normalize_tog_citations,
    _preferred_entity_name_column,
    _serialize_json_safe_context,
)

logger = logging.getLogger(__name__)

async def run_tog_search(
    *,
    collection_id: str,
    query: str,
    load_context,  # callable: async (collection_id, method) -> (version, frames)
) -> SearchResponse:
    """Execute a ToG search and return a SearchResponse.

    Args:
        collection_id: The collection identifier.
        query: The search query.
        load_context: Async callable that returns (active_version, frames).

    Returns
    -------
        SearchResponse with results.
    """
    active_version, frames = await load_context(collection_id, "tog")
    config = load_graphrag_config(
        collection_id, version=active_version, use_cloud_vectors=True
    )
    entities = frames["entities"]
    relationships = frames["relationships"]

    fh = _attach_query_log(collection_id)
    try:
        logger.info("ToG search for collection %s: %s", collection_id, query)
        logger.info(
            "Loaded %d entities and %d relationships", len(entities), len(relationships)
        )

        if logger.isEnabledFor(logging.DEBUG) and len(entities) > 0:
            name_column = _preferred_entity_name_column(entities)
            entity_names_preview = entities[name_column].head(10).astype(str).tolist()
            logger.debug("Available entities: %s", entity_names_preview)
        elif len(entities) == 0:
            logger.warning("No entities found in serving context")

        response_text, context_data = await api.tog_search(
            config=config,
            entities=entities,
            relationships=relationships,
            text_units=frames["text_units"],
            query=query,
        )

        logger.info("ToG search completed for collection %s", collection_id)
    finally:
        _detach_query_log(fh)

    serialized: dict | None = None
    known_entity_names: set[str] = set()
    if context_data and isinstance(context_data, dict):
        serialized, known_entity_names = _build_tog_serialized_context(
            context_data,
            entities=entities,
            relationships=relationships,
        )
        raw_context = _serialize_json_safe_context(context_data)
        if serialized is None:
            logger.warning(
                "ToG context_data could not be normalized; preserving raw context envelope"
            )
            serialized = {"RawContext": raw_context}
        else:
            serialized = {
                **serialized,
                "RawContext": raw_context,
            }

    if known_entity_names:
        response_text = _normalize_tog_citations(response_text, known_entity_names)

    return SearchResponse(
        query=query,
        response=response_text,
        context_data=serialized,
        method=SearchMethod.TOG,
    )
