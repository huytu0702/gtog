"""Query service for GraphRAG search operations."""

import logging
import re
from typing import Any, Optional

import pandas as pd
import graphrag.api as api

from ..config import settings
from ..models import SearchMethod, SearchResponse
from ..utils import (
    load_graphrag_config,
    validate_collection_indexed,
    get_search_data_paths,
)

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _get_query_file_handler(collection_id: str) -> logging.FileHandler:
    """Return a FileHandler writing to the collection's query.log."""
    log_dir = settings.collections_dir / collection_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_dir / "query.log"), mode="a")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def _attach_query_log(collection_id: str) -> logging.FileHandler:
    """Attach a query.log FileHandler to the app logger for this request."""
    handler = _get_query_file_handler(collection_id)
    logger.addHandler(handler)
    return handler


def _detach_query_log(handler: logging.FileHandler) -> None:
    """Remove and close the per-request query.log handler."""
    logger.removeHandler(handler)
    handler.close()

# Column name mappings: what to use as the "name" and "description" per dataset
_CONTEXT_COLS: dict[str, tuple[str, str]] = {
    "entities":      ("entity", "description"),
    "relationships": ("source", "description"),
    "reports":       ("title", "summary"),
    "sources":       ("text", "text"),
    "covariates":    ("subject_id", "covariate_type"),
}


def _serialize_context_records(
    context_data: dict | None,
) -> dict[str, dict[str, dict]] | None:
    """Convert context_records DataFrames into a JSON-serializable lookup dict.

    Returns: {dataset_name: {short_id: {name, description}}}
    """
    if not context_data:
        return None
    result: dict[str, dict[str, dict]] = {}
    for key, df in context_data.items():
        if df is None or df.empty:
            continue
        key_lower = key.lower()
        name_col, desc_col = _CONTEXT_COLS.get(key_lower, ("id", ""))
        lookup: dict[str, dict] = {}
        for _, row in df.iterrows():
            short_id = str(row.get("id", ""))
            name = str(row.get(name_col, "")) if name_col in df.columns else short_id
            desc = str(row.get(desc_col, "")) if desc_col and desc_col in df.columns else ""
            lookup[short_id] = {"name": name, "description": desc}
        result[key] = lookup
    return result or None


def _normalize_tog_citations(text: str, entity_names: set[str]) -> str:
    """Normalize ToG LLM citations to the frontend-expected [Data: Entities (...)] format.

    The LLM often emits [Data: NAME1, NAME2] instead of [Data: Entities (NAME1, NAME2)].
    This detects bare [Data: ...] blocks that contain known entity names and rewrites them.
    """
    # Build case-insensitive name map: lowercase -> original
    name_map = {n.lower(): n for n in entity_names}

    def _rewrite(match: re.Match) -> str:
        inner = match.group(1).strip()
        # Already in correct format: "Entities (...)" or "Relationships (...)"
        if re.match(r"^(Entities|Relationships|Sources|Reports)\s*\(", inner, re.IGNORECASE):
            return match.group(0)
        # Bare names: "GRAPHRAG, MICROSOFT RESEARCH" — check if they're entity names
        raw_names = [n.strip() for n in inner.split(",")]
        matched = [name_map[n.lower()] if n.lower() in name_map else n for n in raw_names if n.strip()]
        if matched:
            return f"[Data: Entities ({', '.join(matched)})]"
        return match.group(0)

    return re.sub(r"\[Data:\s*([^\]]+)\]", _rewrite, text)


class QueryService:
    """Service for managing query/search operations."""

    def __init__(self):
        """Initialize the query service."""
        pass

    async def global_search(
        self,
        collection_id: str,
        query: str,
        community_level: Optional[int] = None,
        dynamic_community_selection: bool = False,
        response_type: str = "Multiple Paragraphs",
    ) -> SearchResponse:
        """
        Perform a global search on a collection.

        Args:
            collection_id: The collection identifier
            query: The search query
            community_level: Community level to search
            dynamic_community_selection: Whether to use dynamic community selection
            response_type: Type of response format

        Returns:
            SearchResponse with results
        """
        # Validate collection is indexed for global search
        is_indexed, error = validate_collection_indexed(collection_id, method="global")
        if not is_indexed:
            raise ValueError(error)

        # Load config and data
        config = load_graphrag_config(collection_id)
        data_paths = get_search_data_paths(collection_id, "global")

        # Load required dataframes
        entities = pd.read_parquet(data_paths["entities"])
        communities = pd.read_parquet(data_paths["communities"])
        community_reports = pd.read_parquet(data_paths["community_reports"])

        fh = _attach_query_log(collection_id)
        try:
            logger.info(f"Global search for collection {collection_id}: {query}")

            # Perform search - API returns (response, context_data) tuple
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

            logger.info(f"Global search completed for collection {collection_id}")
        finally:
            _detach_query_log(fh)

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=_serialize_context_records(context_data),
            method=SearchMethod.GLOBAL,
        )

    async def local_search(
        self,
        collection_id: str,
        query: str,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
    ) -> SearchResponse:
        """
        Perform a local search on a collection.

        Args:
            collection_id: The collection identifier
            query: The search query
            community_level: Community level to search
            response_type: Type of response format

        Returns:
            SearchResponse with results
        """
        # Validate collection is indexed for local search
        is_indexed, error = validate_collection_indexed(collection_id, method="local")
        if not is_indexed:
            raise ValueError(error)

        # Load config and data
        config = load_graphrag_config(collection_id)
        data_paths = get_search_data_paths(collection_id, "local")

        # Load required dataframes
        entities = pd.read_parquet(data_paths["entities"])
        communities = pd.read_parquet(data_paths["communities"])
        community_reports = pd.read_parquet(data_paths["community_reports"])
        text_units = pd.read_parquet(data_paths["text_units"])
        relationships = pd.read_parquet(data_paths["relationships"])

        # Load covariates if available
        covariates = None
        if "covariates" in data_paths:
            covariates = pd.read_parquet(data_paths["covariates"])

        fh = _attach_query_log(collection_id)
        try:
            logger.info(f"Local search for collection {collection_id}: {query}")

            # Perform search - API returns (response, context_data) tuple
            response_text, context_data = await api.local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=covariates,
                community_level=community_level,
                response_type=response_type,
                query=query,
            )

            logger.info(f"Local search completed for collection {collection_id}")
        finally:
            _detach_query_log(fh)

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=_serialize_context_records(context_data),
            method=SearchMethod.LOCAL,
        )

    async def tog_search(
        self,
        collection_id: str,
        query: str,
    ) -> SearchResponse:
        """
        Perform a ToG (Tree-of-Graph) search on a collection.

        Args:
            collection_id: The collection identifier
            query: The search query

        Returns:
            SearchResponse with results
        """
        # Validate collection is indexed for ToG
        is_indexed, error = validate_collection_indexed(collection_id, method="tog")
        if not is_indexed:
            raise ValueError(error)

        # Load config and data
        config = load_graphrag_config(collection_id)
        data_paths = get_search_data_paths(collection_id, "tog")

        # Load required dataframes
        entities = pd.read_parquet(data_paths["entities"])
        relationships = pd.read_parquet(data_paths["relationships"])

        fh = _attach_query_log(collection_id)
        try:
            logger.info(f"ToG search for collection {collection_id}: {query}")
            logger.info(
                f"Loaded {len(entities)} entities and {len(relationships)} relationships"
            )

            # Debug: Show entity names
            if len(entities) > 0:
                entity_names = entities["title"].tolist()[:10]
                logger.info(f"Available entities: {entity_names}")
            else:
                logger.warning("No entities found in parquet file")

            # Perform search - API returns (response, context_data) tuple
            response_text, context_data = await api.tog_search(
                config=config,
                entities=entities,
                relationships=relationships,
                query=query,
            )

            logger.info(f"ToG search completed for collection {collection_id}")
        finally:
            _detach_query_log(fh)

        serialized: dict | None = None
        known_entity_names: set[str] = set()
        if context_data and isinstance(context_data, dict):
            paths = context_data.get("exploration_paths", [])
            if paths:
                entity_paths: dict[str, list[str]] = {}  # entity -> paths it appears in
                relationships: dict[str, dict] = {}
                for path in paths:
                    # Each path: "A --[rel]--> B | B --[rel2]--> C"
                    for segment in path.split(" | "):
                        m = re.match(r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment.strip())
                        if m:
                            src, rel, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                            entity_paths.setdefault(src, []).append(segment.strip())
                            entity_paths.setdefault(tgt, []).append(segment.strip())
                            known_entity_names.add(src)
                            known_entity_names.add(tgt)
                            relationships[rel] = {"name": rel, "description": ""}
                entities = {
                    name: {"name": name, "description": " | ".join(dict.fromkeys(path_list))}
                    for name, path_list in entity_paths.items()
                }
                serialized = {}
                if entities:
                    serialized["Entities"] = entities
                if relationships:
                    serialized["Relationships"] = relationships

        # Normalize LLM citations: [Data: NAME1, NAME2] -> [Data: Entities (NAME1, NAME2)]
        # The LLM often drops the "Entities (...)" wrapper despite prompt instructions
        if known_entity_names:
            response_text = _normalize_tog_citations(response_text, known_entity_names)

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=serialized,
            method=SearchMethod.TOG,
        )

    async def drift_search(
        self,
        collection_id: str,
        query: str,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
    ) -> SearchResponse:
        """
        Perform a DRIFT search on a collection.

        Args:
            collection_id: The collection identifier
            query: The search query
            community_level: Community level to search
            response_type: Type of response format

        Returns:
            SearchResponse with results
        """
        # Validate collection is indexed for drift search
        is_indexed, error = validate_collection_indexed(collection_id, method="drift")
        if not is_indexed:
            raise ValueError(error)

        # Load config and data
        config = load_graphrag_config(collection_id)
        data_paths = get_search_data_paths(collection_id, "drift")

        # Load required dataframes
        entities = pd.read_parquet(data_paths["entities"])
        communities = pd.read_parquet(data_paths["communities"])
        community_reports = pd.read_parquet(data_paths["community_reports"])
        text_units = pd.read_parquet(data_paths["text_units"])
        relationships = pd.read_parquet(data_paths["relationships"])

        fh = _attach_query_log(collection_id)
        try:
            logger.info(f"DRIFT search for collection {collection_id}: {query}")

            # Perform search - API returns (response, context_data) tuple
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

            logger.info(f"DRIFT search completed for collection {collection_id}")
        finally:
            _detach_query_log(fh)

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=_serialize_context_records(context_data),
            method=SearchMethod.DRIFT,
        )


# Global query service instance
query_service = QueryService()
