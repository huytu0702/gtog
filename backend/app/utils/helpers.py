"""Utility helper functions."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from ..config import settings

logger = logging.getLogger(__name__)


def _is_cosmos_mode() -> bool:
    """Check if storage mode is set to cosmos."""
    return (settings.storage_mode or "file").strip().lower() == "cosmos"


def _normalize_litellm_model_config(config: GraphRagConfig) -> None:
    """
    Normalize model/provider pairs to avoid LiteLLM provider parsing failures.

    Older configs may set model strings like ``google_ai_studio/gemini-embedding-001``.
    For GraphRAG LiteLLM config, provider and model should be split as:
    ``model_provider: gemini`` and ``model: gemini-embedding-001``.
    """
    provider_aliases = {
        "google_ai_studio": "gemini",
    }

    for model_id, model_cfg in config.models.items():
        raw_model = (model_cfg.model or "").strip()
        raw_provider = (model_cfg.model_provider or "").strip()

        # Provider aliases to canonical provider names.
        if raw_provider in provider_aliases:
            canonical_provider = provider_aliases[raw_provider]
            model_cfg.model_provider = canonical_provider
            logger.warning(
                "Normalized model_provider for %s: '%s' -> '%s'",
                model_id,
                raw_provider,
                canonical_provider,
            )

        # Split provider-prefixed model names into separate provider/model fields.
        if "/" in raw_model:
            prefix, normalized_model = raw_model.split("/", 1)
            prefix = prefix.strip()
            normalized_model = normalized_model.strip()

            if prefix in provider_aliases and normalized_model:
                model_cfg.model_provider = provider_aliases[prefix]
                model_cfg.model = normalized_model
                logger.warning(
                    "Normalized model for %s: '%s' -> provider='%s', model='%s'",
                    model_id,
                    raw_model,
                    model_cfg.model_provider,
                    model_cfg.model,
                )


def load_graphrag_config(collection_id: str) -> GraphRagConfig:
    """
    Load shared GraphRAG configuration and override collection-specific paths.

    Args:
        collection_id: The collection identifier

    Returns:
        GraphRagConfig with collection-specific path overrides
    """
    # Get absolute paths
    storage_root = settings.collections_dir.resolve()
    collection_dir = storage_root / collection_id
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Build CLI overrides - only apply file overrides when NOT in cosmos mode
    cli_overrides = {
        "input.storage.base_dir": str(collection_dir / "input"),
        "input.file_pattern": ".*\\.(txt|md)$",
    }

    if not _is_cosmos_mode():
        cli_overrides.update({
            "input.storage.type": "file",
            "output.type": "file",
            "output.base_dir": str(collection_dir / "output"),
            "cache.type": "file",
            "cache.base_dir": str(collection_dir / "cache"),
        })

    # Load the shared settings.yaml with collection-specific overrides
    config = load_config(
        root_dir=str(collection_dir),
        config_filepath=settings.settings_yaml_path,
        cli_overrides=cli_overrides,
    )

    _normalize_litellm_model_config(config)

    return config


def validate_collection_indexed(
    collection_id: str, method: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if a collection has been successfully indexed.

    Args:
        collection_id: The collection identifier
        method: Optional search method for method-specific validation

    Returns:
        Tuple of (is_indexed, error_message)
    """
    collection_dir = settings.collections_dir / collection_id
    output_dir = collection_dir / "output"

    if not output_dir.exists():
        return False, "Collection has not been indexed yet"

    # Base required files for all methods
    required_files = [
        "entities.parquet",
        "communities.parquet",
        "community_reports.parquet",
    ]

    # Method-specific requirements
    if method in ["local", "drift", "tog"]:
        required_files.extend(["text_units.parquet", "relationships.parquet"])

    # ToG has strict requirements
    if method == "tog":
        # ToG specifically needs entities and relationships
        if not (output_dir / "entities.parquet").exists():
            return False, "ToG search requires entities.parquet"
        if not (output_dir / "relationships.parquet").exists():
            return False, "ToG search requires relationships.parquet"

    missing_files = []
    for file in required_files:
        if not (output_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        return False, f"Missing indexed files: {', '.join(missing_files)}"

    return True, None


def get_search_data_paths(collection_id: str, method: str) -> Dict[str, Path]:
    """
    Get paths to required parquet files for a search method.

    Args:
        collection_id: The collection identifier
        method: The search method (global, local, tog, drift)

    Returns:
        Dictionary of data file paths
    """
    output_dir = settings.collections_dir / collection_id / "output"

    # Common files for all methods
    paths = {
        "entities": output_dir / "entities.parquet",
        "communities": output_dir / "communities.parquet",
        "community_reports": output_dir / "community_reports.parquet",
    }

    # Method-specific files
    if method in ["local", "drift", "tog"]:
        paths.update(
            {
                "text_units": output_dir / "text_units.parquet",
                "relationships": output_dir / "relationships.parquet",
            }
        )

    if method == "local":
        # Local search may use covariates if available
        covariates_path = output_dir / "covariates.parquet"
        if covariates_path.exists():
            paths["covariates"] = covariates_path

    # ToG-specific validation
    if method == "tog":
        # Ensure ToG has required files
        required_files = ["entities.parquet", "relationships.parquet"]
        missing_files = []
        for file in required_files:
            if not (output_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            raise FileNotFoundError(
                f"ToG search requires missing files: {', '.join(missing_files)}"
            )

    return paths


def get_collection_info(collection_id: str) -> Optional[Dict]:
    """
    Get basic information about a collection.

    Args:
        collection_id: The collection identifier

    Returns:
        Dictionary with collection info or None if not found
    """
    collection_dir = settings.collections_dir / collection_id

    if not collection_dir.exists():
        return None

    input_dir = collection_dir / "input"
    output_dir = collection_dir / "output"

    # Count documents
    document_count = 0
    if input_dir.exists():
        document_count = len([f for f in input_dir.iterdir() if f.is_file()])

    # Check if indexed
    is_indexed, _ = validate_collection_indexed(collection_id)

    # Get creation time
    created_at = collection_dir.stat().st_ctime

    return {
        "id": collection_id,
        "name": collection_id,
        "document_count": document_count,
        "indexed": is_indexed,
        "created_at": created_at,
    }
