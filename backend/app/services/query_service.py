"""Query service for GraphRAG search operations."""

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import pandas as pd
import graphrag.api as api
from ..errors import ServingContextNotReadyError, ServingContextUnavailableError
from ..models import SearchMethod, SearchResponse
from ..repositories import get_control_plane_repository, get_serving_repository
from ..utils import load_graphrag_config
from .serving_context_cache import serving_context_cache

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _attach_query_log(collection_id: str):
    """No-op query logger in cloud mode (avoid local file writes)."""
    return None


def _detach_query_log(handler) -> None:
    """No-op query logger in cloud mode."""
    return None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, bool):
            return is_na
    except Exception:
        pass
    return False


def _preferred_entity_name_column(entities: pd.DataFrame) -> str:
    for col in ("title", "name", "entity", "id"):
        if col in entities.columns:
            return col
    return entities.columns[0] if len(entities.columns) > 0 else "id"


def _non_empty_text(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return str(value).strip()


def _coerce_findings(value: Any) -> list[dict[str, Any]]:
    if _is_missing_value(value):
        return []

    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return []
    elif hasattr(parsed, "tolist") and not isinstance(parsed, (bytes, bytearray)):
        try:
            parsed = parsed.tolist()
        except Exception:
            return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _community_report_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw_payload = row.get("full_content_json")
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    elif isinstance(raw_payload, dict):
        payload = raw_payload
    return payload


def _render_community_report_full_content(row: pd.Series) -> str:
    payload = _community_report_payload(row)
    title = (
        _non_empty_text(payload.get("title") or row.get("title")) or "Community Report"
    )
    summary = _non_empty_text(payload.get("summary") or row.get("summary"))
    findings = _coerce_findings(payload.get("findings") or row.get("findings"))
    rating = payload.get("rating")
    if _is_missing_value(rating):
        rating = row.get("rank")
    rating_text = _non_empty_text(rating)
    rating_explanation = _non_empty_text(
        payload.get("rating_explanation") or row.get("rating_explanation")
    )

    sections = [f"# {title}"]
    if summary:
        sections.append(summary)

    for finding in findings:
        finding_title = _non_empty_text(finding.get("summary")) or "Finding"
        explanation = _non_empty_text(finding.get("explanation"))
        if explanation:
            sections.append(f"## {finding_title}\n\n{explanation}")

    if rating_text or rating_explanation:
        rating_body = rating_text
        if rating_explanation:
            rating_body = (
                f"{rating_body}\n\n{rating_explanation}"
                if rating_body
                else rating_explanation
            )
        sections.append(f"## Impact Severity Rating\n\n{rating_body}")

    rendered = "\n\n".join(section for section in sections if section)
    return rendered.strip() or title


def _normalize_community_reports_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    normalized = frame.copy()
    if "full_content" not in normalized.columns:
        normalized["full_content"] = ""

    missing_mask = normalized["full_content"].apply(_is_missing_value)
    if missing_mask.any():
        normalized.loc[missing_mask, "full_content"] = normalized.loc[
            missing_mask
        ].apply(
            _render_community_report_full_content,
            axis=1,
        )

    return normalized


# Column name mappings: what to use as the "name" and "description" per dataset
_CONTEXT_COLS: dict[str, tuple[str, str]] = {
    "entities": ("entity", "description"),
    "relationships": ("source", "description"),
    "reports": ("title", "summary"),
    "sources": ("text", "text"),
    "covariates": ("subject_id", "covariate_type"),
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
            desc = (
                str(row.get(desc_col, ""))
                if desc_col and desc_col in df.columns
                else ""
            )
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
        if re.match(
            r"^(Entities|Relationships|Sources|Reports)\s*\(", inner, re.IGNORECASE
        ):
            return match.group(0)
        # Bare names: "GRAPHRAG, MICROSOFT RESEARCH" — check if they're entity names
        raw_names = [n.strip() for n in inner.split(",")]
        matched = [
            name_map[n.lower()] if n.lower() in name_map else n
            for n in raw_names
            if n.strip()
        ]
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
        self.context_cache = serving_context_cache

    async def _load_dataset_frame(
        self,
        *,
        collection_id: str,
        version: str,
        dataset: str,
    ) -> pd.DataFrame:
        if self.serving_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos serving repository is not configured"
            )

        def _loader() -> pd.DataFrame:
            return self.serving_repo.load_dataframe(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )

        started = time.perf_counter()
        try:
            cache_hit, frame = await asyncio.to_thread(
                self.context_cache.get_or_load_with_status,
                collection_id=collection_id,
                version=version,
                dataset=dataset,
                loader=_loader,
            )
        except Exception as exc:
            raise ServingContextUnavailableError(
                f"Failed loading serving dataset '{dataset}' for version '{version}'"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "serving_context_load collection=%s version=%s dataset=%s cache_hit=%s rows=%s load_ms=%.2f",
            collection_id,
            version,
            dataset,
            cache_hit,
            len(frame),
            elapsed_ms,
        )
        return frame

    async def _load_context_from_serving(
        self, collection_id: str, method: str
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        if self.control_plane is None or self.serving_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos serving repository is not configured"
            )

        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise FileNotFoundError(f"Collection '{collection_id}' not found")

        active_version = collection.get("activeVersion")
        if not active_version:
            raise ServingContextNotReadyError(
                "Collection has not been indexed yet (no active serving version)"
            )

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
            frame = await self._load_dataset_frame(
                collection_id=collection_id,
                version=str(active_version),
                dataset=dataset,
            )
            if dataset == "community_reports":
                frame = _normalize_community_reports_frame(frame)
            if frame.empty:
                raise ServingContextNotReadyError(
                    f"Serving context is incomplete for active version {active_version} "
                    f"(dataset={dataset})"
                )
            frames[dataset] = frame

        if method == "local":
            covariates = await self._load_dataset_frame(
                collection_id=collection_id,
                version=str(active_version),
                dataset="covariates",
            )
            if not covariates.empty:
                frames["covariates"] = covariates

        return str(active_version), frames

    def invalidate_collection_cache(self, collection_id: str) -> None:
        """Invalidate in-process serving context cache for one collection."""
        self.context_cache.invalidate_collection(collection_id)

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
        active_version, frames = await self._load_context_from_serving(
            collection_id, "global"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, query_runtime=True
        )
        entities = frames["entities"]
        communities = frames["communities"]
        community_reports = frames["community_reports"]

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
        active_version, frames = await self._load_context_from_serving(
            collection_id, "local"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, query_runtime=True
        )
        entities = frames["entities"]
        communities = frames["communities"]
        community_reports = frames["community_reports"]
        text_units = frames["text_units"]
        relationships = frames["relationships"]
        covariates = frames.get("covariates")

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
        active_version, frames = await self._load_context_from_serving(
            collection_id, "tog"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, query_runtime=True
        )
        entities = frames["entities"]
        relationships = frames["relationships"]

        fh = _attach_query_log(collection_id)
        try:
            logger.info(f"ToG search for collection {collection_id}: {query}")
            logger.info(
                f"Loaded {len(entities)} entities and {len(relationships)} relationships"
            )

            # Debug: Show entity names
            if len(entities) > 0:
                name_column = _preferred_entity_name_column(entities)
                entity_names = entities[name_column].astype(str).tolist()[:10]
                logger.info(f"Available entities: {entity_names}")
            else:
                logger.warning("No entities found in serving context")

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
                        m = re.match(
                            r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment.strip()
                        )
                        if m:
                            src, rel, tgt = (
                                m.group(1).strip(),
                                m.group(2).strip(),
                                m.group(3).strip(),
                            )
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
        active_version, frames = await self._load_context_from_serving(
            collection_id, "drift"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, query_runtime=True
        )
        entities = frames["entities"]
        communities = frames["communities"]
        community_reports = frames["community_reports"]
        text_units = frames["text_units"]
        relationships = frames["relationships"]

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

    def get_tog_entities_preview(
        self, collection_id: str, limit: int = 20
    ) -> dict[str, Any]:
        """Return ToG entity preview for debugging."""
        if self.control_plane is None or self.serving_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos serving repository is not configured"
            )
        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise FileNotFoundError(f"Collection '{collection_id}' not found")
        active_version = str(collection.get("activeVersion") or "")
        if not active_version:
            raise ServingContextNotReadyError(
                "Collection has not been indexed yet (no active serving version)"
            )
        cache_hit, entities_df = self.context_cache.get_or_load_with_status(
            collection_id=collection_id,
            version=active_version,
            dataset="entities",
            loader=lambda: self.serving_repo.load_dataframe(
                collection_id=collection_id,
                version=active_version,
                dataset="entities",
            ),
        )
        logger.info(
            "serving_context_preview collection=%s version=%s dataset=entities cache_hit=%s rows=%s",
            collection_id,
            active_version,
            cache_hit,
            len(entities_df),
        )
        source = f"cosmos:{active_version}"

        entities_info = []
        for _, row in entities_df.head(limit).iterrows():
            description = str(row.get("description", ""))
            entity_id = row.get("title")
            if _is_missing_value(entity_id):
                entity_id = row.get("id")
            if _is_missing_value(entity_id):
                entity_id = row.get("name")
            entities_info.append({
                "id": str(entity_id) if not _is_missing_value(entity_id) else "",
                "description": description[:100] + "..."
                if len(description) > 100
                else description,
                "type": row.get("type", "unknown"),
            })

        return {
            "collection_id": collection_id,
            "source": source,
            "total_entities": len(entities_df),
            "showing_first": len(entities_info),
            "entities": entities_info,
        }


# Global query service instance
query_service = QueryService()
