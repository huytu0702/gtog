"""Query service for GraphRAG search operations."""

import logging
import re
import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import graphrag.api as api
from ..config import settings
from ..models import SearchMethod, SearchResponse
from ..repositories import get_control_plane_repository, get_serving_repository
from ..utils import (
    load_graphrag_config,
    validate_collection_indexed,
    get_search_data_paths,
)
from ..utils.helpers import _blob_client, _collection_container
from ..utils.helpers import _storage_connection_string

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_BLOB_PARQUET_FALLBACK_WARNING_EMITTED = False


def _attach_query_log(collection_id: str):
    """No-op query logger in cloud mode (avoid local file writes)."""
    return None


def _detach_query_log(handler) -> None:
    """No-op query logger in cloud mode."""
    return None


def _blob_parquet(collection_id: str, relative_path: Path) -> pd.DataFrame:
    """Read parquet from collection blob output/<relative_path> via authenticated download."""
    global _BLOB_PARQUET_FALLBACK_WARNING_EMITTED
    if not _BLOB_PARQUET_FALLBACK_WARNING_EMITTED:
        logger.warning(
            "Using temporary blob/parquet fallback in query hot path; "
            "this remains only until Phase 3 cutover."
        )
        _BLOB_PARQUET_FALLBACK_WARNING_EMITTED = True

    client = _blob_client()
    container = client.get_container_client(_collection_container(collection_id))
    data = container.get_blob_client(f"output/{relative_path.as_posix()}").download_blob().readall()
    return pd.read_parquet(io.BytesIO(data))

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
        self.control_plane = get_control_plane_repository()
        self.serving_repo = get_serving_repository()

    def _load_context_from_serving(
        self, collection_id: str, method: str
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        if self.control_plane is None or self.serving_repo is None:
            raise RuntimeError("Cosmos serving repository is not configured")

        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Collection '{collection_id}' not found")

        active_version = collection.get("activeVersion")
        if not active_version:
            raise ValueError("Collection has not been indexed yet (no active serving version)")

        required = {
            "global": ["entities", "communities", "community_reports"],
            "local": [
                "entities",
                "communities",
                "community_reports",
                "text_units",
                "relationships",
            ],
            "tog": ["entities", "relationships"],
            "drift": [
                "entities",
                "communities",
                "community_reports",
                "text_units",
                "relationships",
            ],
        }[method]

        frames: dict[str, pd.DataFrame] = {}
        for dataset in required:
            frame = self.serving_repo.load_dataframe(
                collection_id=collection_id,
                version=str(active_version),
                dataset=dataset,
            )
            if frame.empty:
                raise ValueError(
                    f"Serving context is incomplete for active version {active_version} "
                    f"(dataset={dataset})"
                )
            frames[dataset] = frame

        if method == "local":
            covariates = self.serving_repo.load_dataframe(
                collection_id=collection_id,
                version=str(active_version),
                dataset="covariates",
            )
            if not covariates.empty:
                frames["covariates"] = covariates

        return str(active_version), frames

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
        if self.control_plane is not None and self.serving_repo is not None:
            active_version, frames = self._load_context_from_serving(collection_id, "global")
            config = load_graphrag_config(collection_id, version=active_version)
            entities = frames["entities"]
            communities = frames["communities"]
            community_reports = frames["community_reports"]
        else:
            is_indexed, error = validate_collection_indexed(collection_id, method="global")
            if not is_indexed:
                raise ValueError(error)
            config = load_graphrag_config(collection_id)
            data_paths = get_search_data_paths(collection_id, "global")
            if _storage_connection_string():
                entities = _blob_parquet(collection_id, data_paths["entities"])
                communities = _blob_parquet(collection_id, data_paths["communities"])
                community_reports = _blob_parquet(collection_id, data_paths["community_reports"])
            else:
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
        if self.control_plane is not None and self.serving_repo is not None:
            active_version, frames = self._load_context_from_serving(collection_id, "local")
            config = load_graphrag_config(collection_id, version=active_version)
            entities = frames["entities"]
            communities = frames["communities"]
            community_reports = frames["community_reports"]
            text_units = frames["text_units"]
            relationships = frames["relationships"]
            covariates = frames.get("covariates")
        else:
            is_indexed, error = validate_collection_indexed(collection_id, method="local")
            if not is_indexed:
                raise ValueError(error)
            config = load_graphrag_config(collection_id)
            data_paths = get_search_data_paths(collection_id, "local")
            if _storage_connection_string():
                entities = _blob_parquet(collection_id, data_paths["entities"])
                communities = _blob_parquet(collection_id, data_paths["communities"])
                community_reports = _blob_parquet(collection_id, data_paths["community_reports"])
                text_units = _blob_parquet(collection_id, data_paths["text_units"])
                relationships = _blob_parquet(collection_id, data_paths["relationships"])
            else:
                entities = pd.read_parquet(data_paths["entities"])
                communities = pd.read_parquet(data_paths["communities"])
                community_reports = pd.read_parquet(data_paths["community_reports"])
                text_units = pd.read_parquet(data_paths["text_units"])
                relationships = pd.read_parquet(data_paths["relationships"])

            covariates = None
            if "covariates" in data_paths:
                if _storage_connection_string():
                    covariates = _blob_parquet(collection_id, data_paths["covariates"])
                else:
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
        if self.control_plane is not None and self.serving_repo is not None:
            active_version, frames = self._load_context_from_serving(collection_id, "tog")
            config = load_graphrag_config(collection_id, version=active_version)
            entities = frames["entities"]
            relationships = frames["relationships"]
        else:
            is_indexed, error = validate_collection_indexed(collection_id, method="tog")
            if not is_indexed:
                raise ValueError(error)
            config = load_graphrag_config(collection_id)
            data_paths = get_search_data_paths(collection_id, "tog")
            if _storage_connection_string():
                entities = _blob_parquet(collection_id, data_paths["entities"])
                relationships = _blob_parquet(collection_id, data_paths["relationships"])
            else:
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
                rel_lookup: dict[str, dict] = {}
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
                            rel_lookup[rel] = {"name": rel, "description": ""}
                entity_lookup = {
                    name: {"name": name, "description": " | ".join(dict.fromkeys(path_list))}
                    for name, path_list in entity_paths.items()
                }
                serialized = {}
                if entity_lookup:
                    serialized["Entities"] = entity_lookup
                if rel_lookup:
                    serialized["Relationships"] = rel_lookup

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
        if self.control_plane is not None and self.serving_repo is not None:
            active_version, frames = self._load_context_from_serving(collection_id, "drift")
            config = load_graphrag_config(collection_id, version=active_version)
            entities = frames["entities"]
            communities = frames["communities"]
            community_reports = frames["community_reports"]
            text_units = frames["text_units"]
            relationships = frames["relationships"]
        else:
            is_indexed, error = validate_collection_indexed(collection_id, method="drift")
            if not is_indexed:
                raise ValueError(error)

            config = load_graphrag_config(collection_id)
            data_paths = get_search_data_paths(collection_id, "drift")
            if _storage_connection_string():
                entities = _blob_parquet(collection_id, data_paths["entities"])
                communities = _blob_parquet(collection_id, data_paths["communities"])
                community_reports = _blob_parquet(collection_id, data_paths["community_reports"])
                text_units = _blob_parquet(collection_id, data_paths["text_units"])
                relationships = _blob_parquet(collection_id, data_paths["relationships"])
            else:
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

    def get_tog_entities_preview(self, collection_id: str, limit: int = 20) -> dict[str, Any]:
        """Return ToG entity preview for debugging."""
        if self.control_plane is not None and self.serving_repo is not None:
            active_version, frames = self._load_context_from_serving(collection_id, "tog")
            entities_df = frames["entities"]
            source = f"cosmos:{active_version}"
        else:
            data_paths = get_search_data_paths(collection_id, "tog")
            if _storage_connection_string():
                entities_df = _blob_parquet(collection_id, data_paths["entities"])
                source = "blob"
            else:
                entities_df = pd.read_parquet(data_paths["entities"])
                source = "file"

        entities_info = []
        for _, row in entities_df.head(limit).iterrows():
            description = str(row.get("description", ""))
            entities_info.append(
                {
                    "id": row.get("title") or row.get("id"),
                    "description": description[:100] + "..." if len(description) > 100 else description,
                    "type": row.get("type", "unknown"),
                }
            )

        return {
            "collection_id": collection_id,
            "source": source,
            "total_entities": len(entities_df),
            "showing_first": len(entities_info),
            "entities": entities_info,
        }


# Global query service instance
query_service = QueryService()
