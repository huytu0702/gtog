"""Query service for GraphRAG search operations."""

import logging
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

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=None,  # Avoid serialization issues with pandas DataFrames
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

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=None,  # Avoid serialization issues with pandas DataFrames
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

        return SearchResponse(
            query=query,
            response=response_text,
            context_data=None,  # Avoid serialization issues with pandas DataFrames
            method=SearchMethod.DRIFT,
        )


# Global query service instance
query_service = QueryService()
