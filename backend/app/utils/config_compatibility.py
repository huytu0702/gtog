"""Phase 0 GraphRAG settings compatibility checkpoint."""

from __future__ import annotations

from pathlib import Path

import yaml

from graphrag.config.enums import CacheType, ReportingType, StorageType, VectorStoreType

PHASE0_COMPATIBILITY_CHECKS: tuple[str, ...] = (
    "vector_store.default_vector_store uses embeddings_schema (not index_schema)",
    "input/output/cache/reporting type values match GraphRAG enums",
    "azure_ai_search vector store has required url",
    "cosmosdb vector store has required url + database_name",
)


def _require_enum(value: str | None, allowed: set[str], key_path: str) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{key_path} must be one of [{allowed_values}], got {value!r}")


def validate_graphrag_settings_compatibility(settings_yaml_path: Path) -> None:
    """Validate backend settings.yaml keys against current GraphRAG config contracts."""
    data = yaml.safe_load(settings_yaml_path.read_text(encoding="utf-8")) or {}

    input_type = ((data.get("input") or {}).get("storage") or {}).get("type")
    output_type = (data.get("output") or {}).get("type")
    cache_type = (data.get("cache") or {}).get("type")
    reporting_type = (data.get("reporting") or {}).get("type")

    _require_enum(input_type, {e.value for e in StorageType}, "input.storage.type")
    _require_enum(output_type, {e.value for e in StorageType}, "output.type")
    _require_enum(cache_type, {e.value for e in CacheType}, "cache.type")
    _require_enum(reporting_type, {e.value for e in ReportingType}, "reporting.type")

    default_store = ((data.get("vector_store") or {}).get("default_vector_store") or {})

    if "index_schema" in default_store:
        raise ValueError(
            "vector_store.default_vector_store.index_schema is not supported; "
            "use embeddings_schema."
        )

    store_type = default_store.get("type")
    _require_enum(
        store_type,
        {e.value for e in VectorStoreType},
        "vector_store.default_vector_store.type",
    )

    if store_type == VectorStoreType.AzureAISearch.value and not default_store.get("url"):
        raise ValueError(
            "vector_store.default_vector_store.url is required for azure_ai_search"
        )

    if store_type == VectorStoreType.CosmosDB.value:
        if not default_store.get("url"):
            raise ValueError("vector_store.default_vector_store.url is required for cosmosdb")
        if not default_store.get("database_name"):
            raise ValueError(
                "vector_store.default_vector_store.database_name is required for cosmosdb"
            )
