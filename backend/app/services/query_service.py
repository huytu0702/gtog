"""Query service - orchestrates search operations against the knowledge graph.

Shared data-normalisation helpers live in query_service_base.  All symbols
that test code patches directly (api, pd, load_graphrag_config,
_normalize_community_reports_frame) are imported at module level here so
that patch.object(query_service_module, ...) calls work unchanged.
"""

import asyncio
import logging
import time
from typing import Any, Optional

import graphrag.api as api  # tests patch query_service_module.api.*
import pandas as pd  # tests patch query_service_module.pd

from ..errors import ServingContextNotReadyError, ServingContextUnavailableError
from ..models import SearchMethod, SearchResponse
from ..repositories import get_control_plane_repository, get_pipeline_output_repository
from ..utils import (
    load_graphrag_config,
)  # tests patch query_service_module.load_graphrag_config
from .query_service_base import (
    _attach_query_log,
    _build_tog_entity_sources_map,
    _build_tog_relationships_context,
    _build_tog_sources_context,
    _detach_query_log,
    _extract_tog_entity_names_from_context,
    _is_missing_value,
    _normalize_community_reports_frame,  # tests patch this name on this module
    _normalize_context_citations,
    _normalize_tog_citations,
    _preferred_entity_name_column,
    _serialize_context_records,
    _serialize_tog_context_data,
)
from .serving_context_cache import serving_context_cache

logger = logging.getLogger(__name__)

_REQUIRED_DATASETS: dict[str, list[str]] = {
    "global": ["entities", "communities", "community_reports"],
    "local": [
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    ],
    "tog": ["entities", "relationships", "text_units"],
    "drift": [
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    ],
}


class QueryService:
    """Service for managing query/search operations."""

    def __init__(self) -> None:
        """Initialize the query service."""
        self.control_plane = get_control_plane_repository()
        self.pipeline_repo = get_pipeline_output_repository()
        self.context_cache = serving_context_cache

    async def _load_dataset_frame(
        self,
        *,
        collection_id: str,
        version: str,
        dataset: str,
    ) -> pd.DataFrame:
        if self.pipeline_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos pipeline output repository is not configured"
            )

        def _loader() -> pd.DataFrame:
            pipeline_repo = self.pipeline_repo
            if pipeline_repo is None:
                raise ServingContextUnavailableError(
                    "Cosmos pipeline output repository is not configured"
                )
            return pipeline_repo.load_dataframe(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )

        started = time.perf_counter()
        try:
            # NOTE: azure-cosmos SDK is synchronous; wrapping in asyncio.to_thread()
            # to avoid blocking the event loop while loading from Cosmos or the LRU
            # cache (which may invoke the synchronous loader under the cache lock).
            cache_hit, frame = await asyncio.to_thread(
                self.context_cache.get_or_load_with_status,
                collection_id=collection_id,
                version=version,
                dataset=dataset,
                loader=_loader,
            )
        except Exception as exc:
            raise ServingContextUnavailableError(
                f"Failed loading pipeline dataset '{dataset}' for version '{version}'"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "pipeline_context_load collection=%s version=%s dataset=%s "
            "cache_hit=%s rows=%s load_ms=%.2f",
            collection_id,
            version,
            dataset,
            cache_hit,
            len(frame),
            elapsed_ms,
        )
        return frame

    def _load_context_from_local(
        self, collection_id: str, method: str
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        raise ServingContextUnavailableError(
            "Local filesystem context loading is no longer supported"
        )

    async def _load_context_from_pipeline(
        self, collection_id: str, method: str
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        if self.control_plane is None or self.pipeline_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos control-plane or pipeline repository is not configured"
            )

        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise FileNotFoundError(f"Collection '{collection_id}' not found")

        active_version = collection.get("activeVersion")
        if not active_version:
            raise ServingContextNotReadyError(
                "Collection has not been indexed yet (no active pipeline version)"
            )

        required = _REQUIRED_DATASETS[method]

        loaded_frames = await asyncio.gather(*[
            self._load_dataset_frame(
                collection_id=collection_id,
                version=str(active_version),
                dataset=dataset,
            )
            for dataset in required
        ])

        frames: dict[str, pd.DataFrame] = {}
        for dataset, frame in zip(required, loaded_frames, strict=False):
            if dataset == "community_reports":
                frame = _normalize_community_reports_frame(frame)
            if frame.empty:
                if dataset == "community_reports":
                    logger.warning(
                        "community_reports is empty for version %s, skipping",
                        active_version,
                    )
                    continue
                raise ServingContextNotReadyError(
                    f"Pipeline context is incomplete for active version {active_version} "
                    f"(dataset={dataset})"
                )
            frames[dataset] = frame

        if method == "local":
            try:
                covariates = await self._load_dataset_frame(
                    collection_id=collection_id,
                    version=str(active_version),
                    dataset="covariates",
                )
            except Exception:
                covariates = pd.DataFrame()
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
        """Perform a global search on a collection."""
        active_version, frames = await self._load_context_from_pipeline(
            collection_id, "global"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, use_cloud_vectors=True
        )
        fh = _attach_query_log(collection_id)
        try:
            logger.info("Global search for collection %s: %s", collection_id, query)
            response_text, context_data = await api.global_search(
                config=config,
                entities=frames["entities"],
                communities=frames["communities"],
                community_reports=frames.get("community_reports", pd.DataFrame()),
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

    async def local_search(
        self,
        collection_id: str,
        query: str,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
    ) -> SearchResponse:
        """Perform a local search on a collection."""
        active_version, frames = await self._load_context_from_pipeline(
            collection_id, "local"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, use_cloud_vectors=True
        )
        covariates: pd.DataFrame | None = frames.get("covariates")

        fh = _attach_query_log(collection_id)
        try:
            logger.info("Local search for collection %s: %s", collection_id, query)
            response_text, context_data = await api.local_search(
                config=config,
                entities=frames["entities"],
                communities=frames["communities"],
                community_reports=frames.get("community_reports", pd.DataFrame()),
                text_units=frames["text_units"],
                relationships=frames["relationships"],
                covariates=covariates,
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
            logger.info("Local search completed for collection %s", collection_id)
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
        """Perform a ToG (Think-on-Graph) search on a collection."""
        active_version, frames = await self._load_context_from_pipeline(
            collection_id, "tog"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, use_cloud_vectors=True
        )
        entities = frames["entities"]
        relationships = frames["relationships"]

        fh = _attach_query_log(collection_id)
        try:
            logger.info("ToG search for collection %s: %s", collection_id, query)
            logger.info(
                "Loaded %d entities and %d relationships",
                len(entities),
                len(relationships),
            )
            if logger.isEnabledFor(logging.DEBUG) and len(entities) > 0:
                name_column = _preferred_entity_name_column(entities)
                entity_names_preview = (
                    entities[name_column].head(10).astype(str).tolist()
                )
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

        serialized = _serialize_tog_context_data(context_data)
        known_entity_names = _extract_tog_entity_names_from_context(serialized)
        relationships_context = _build_tog_relationships_context(
            context_data=serialized,
            relationships=relationships,
        )
        entity_sources, ambiguous_entities = _build_tog_entity_sources_map(
            entity_names=known_entity_names,
            entities=entities,
            text_units=frames["text_units"],
        )
        sources = _build_tog_sources_context(
            entity_names=known_entity_names,
            entities=entities,
            text_units=frames["text_units"],
        )
        if serialized is not None:
            serialized = {
                **serialized,
                **(
                    {"Relationships": relationships_context}
                    if relationships_context
                    else {}
                ),
                **({"Sources": sources} if sources else {}),
            }

        if known_entity_names and isinstance(response_text, str):
            response_text = _normalize_tog_citations(response_text, known_entity_names)
        if isinstance(response_text, str):
            response_text = _normalize_context_citations(
                response_text,
                context_data=serialized,
                dataset_order=["sources", "entities", "relationships"],
                entity_names=known_entity_names,
                entity_sources=entity_sources,
                ambiguous_entities=ambiguous_entities,
            )

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
        """Perform a DRIFT search on a collection."""
        active_version, frames = await self._load_context_from_pipeline(
            collection_id, "drift"
        )
        config = load_graphrag_config(
            collection_id, version=active_version, use_cloud_vectors=True
        )
        fh = _attach_query_log(collection_id)
        try:
            logger.info("DRIFT search for collection %s: %s", collection_id, query)
            response_text, context_data = await api.drift_search(
                config=config,
                entities=frames["entities"],
                communities=frames["communities"],
                community_reports=frames.get("community_reports", pd.DataFrame()),
                text_units=frames["text_units"],
                relationships=frames["relationships"],
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

    def get_tog_entities_preview(
        self, collection_id: str, limit: int = 20
    ) -> dict[str, Any]:
        """Return ToG entity preview for debugging."""
        if self.control_plane is None or self.pipeline_repo is None:
            raise ServingContextUnavailableError(
                "Cosmos pipeline output repository is not configured"
            )
        collection = self.control_plane.get_collection(collection_id)
        if collection is None:
            raise FileNotFoundError(f"Collection '{collection_id}' not found")
        active_version = str(collection.get("activeVersion") or "")
        if not active_version:
            raise ServingContextNotReadyError(
                "Collection has not been indexed yet (no active serving version)"
            )
        pipeline_repo = self.pipeline_repo
        assert pipeline_repo is not None  # guarded by None-check above
        cache_hit, entities_df = self.context_cache.get_or_load_with_status(
            collection_id=collection_id,
            version=active_version,
            dataset="entities",
            loader=lambda: pipeline_repo.load_dataframe(
                collection_id=collection_id,
                version=active_version,
                dataset="entities",
            ),
        )
        logger.info(
            "pipeline_context_preview collection=%s version=%s dataset=entities "
            "cache_hit=%s rows=%s",
            collection_id,
            active_version,
            cache_hit,
            len(entities_df),
        )
        source = f"cosmos:{active_version}"

        entities_info: list[dict[str, Any]] = []
        for _, row in entities_df.head(limit).iterrows():
            description = str(row.get("description", ""))
            entity_id = row.get("title")
            if _is_missing_value(entity_id):
                entity_id = row.get("id")
            if _is_missing_value(entity_id):
                entity_id = row.get("name")
            truncated_desc = (
                description[:100] + "..." if len(description) > 100 else description
            )
            entities_info.append({
                "id": str(entity_id) if not _is_missing_value(entity_id) else "",
                "description": truncated_desc,
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
