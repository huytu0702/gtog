"""ToG (Think-on-Graph) search handler."""

import logging
import re

import graphrag.api as api

from ..models import SearchMethod, SearchResponse
from ..utils import load_graphrag_config
from .query_service_base import (
    _attach_query_log,
    _detach_query_log,
    _normalize_tog_citations,
    _preferred_entity_name_column,
)

logger = logging.getLogger(__name__)


def _build_tog_serialized_context(
    context_data: dict,
) -> tuple[dict | None, set[str]]:
    """Parse ToG exploration paths into a serialized context dict and known entity names.

    Returns
    -------
        (serialized_context, known_entity_names)
        serialized_context keys: "Entities", "Relationships"
    """
    paths = context_data.get("exploration_paths", [])
    if not paths:
        return None, set()

    entity_paths: dict[str, list[str]] = {}
    rel_lookup: dict[str, dict] = {}
    known_entity_names: set[str] = set()

    for path in paths:
        for segment in path.split(" | "):
            m = re.match(r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment.strip())
            if m:
                src = m.group(1).strip()
                rel = m.group(2).strip()
                tgt = m.group(3).strip()
                entity_paths.setdefault(src, []).append(segment.strip())
                entity_paths.setdefault(tgt, []).append(segment.strip())
                known_entity_names.add(src)
                known_entity_names.add(tgt)
                rel_lookup[rel] = {"name": rel, "description": ""}

    entity_lookup = {
        name: {
            "name": name,
            "description": " | ".join(dict.fromkeys(path_list)),
        }
        for name, path_list in entity_paths.items()
    }

    serialized: dict[str, dict] = {}
    if entity_lookup:
        serialized["Entities"] = entity_lookup
    if rel_lookup:
        serialized["Relationships"] = rel_lookup

    return serialized or None, known_entity_names


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
            query=query,
        )

        logger.info("ToG search completed for collection %s", collection_id)
    finally:
        _detach_query_log(fh)

    serialized: dict | None = None
    known_entity_names: set[str] = set()
    if context_data and isinstance(context_data, dict):
        serialized, known_entity_names = _build_tog_serialized_context(context_data)

    if known_entity_names:
        response_text = _normalize_tog_citations(response_text, known_entity_names)

    return SearchResponse(
        query=query,
        response=response_text,
        context_data=serialized,
        method=SearchMethod.TOG,
    )
