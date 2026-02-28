"""Utility helper functions."""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from ..config import settings

logger = logging.getLogger(__name__)


def _blob_client():
    """Return an Azure BlobServiceClient if connection string is configured."""
    conn_str = settings.azure_storage_connection_string
    if not conn_str:
        return None
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(conn_str)


def _collection_container(collection_id: str) -> str:
    """Azure Blob container name for a collection's output."""
    return f"col-{collection_id}"


def _ensure_blob_container(collection_id: str) -> None:
    """Create the per-collection blob container if it doesn't exist."""
    client = _blob_client()
    if client is None:
        return
    container_name = _collection_container(collection_id)
    container = client.get_container_client(container_name)
    if not container.exists():
        container.create_container()


def _blob_file_exists(collection_id: str, blob_path: str) -> bool:
    """Check if a blob exists in the collection container."""
    client = _blob_client()
    if client is None:
        return False
    container = client.get_container_client(_collection_container(collection_id))
    blob = container.get_blob_client(blob_path)
    return blob.exists()


def read_parquet_from_blob(collection_id: str, blob_path: str) -> pd.DataFrame:
    """Download a parquet file from blob and return as DataFrame."""
    client = _blob_client()
    if client is None:
        raise FileNotFoundError(f"Azure storage not configured, cannot read {blob_path}")
    container = client.get_container_client(_collection_container(collection_id))
    data = container.get_blob_client(blob_path).download_blob().readall()
    return pd.read_parquet(io.BytesIO(data))


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
    use_blob = bool(settings.azure_storage_connection_string)

    if use_blob:
        _ensure_blob_container(collection_id)
        container_name = _collection_container(collection_id)
        conn_str = settings.azure_storage_connection_string
        # Use a temp local dir as graphrag root (for prompts/settings resolution only)
        storage_root = settings.collections_dir.resolve()
        collection_dir = storage_root / collection_id
        collection_dir.mkdir(parents=True, exist_ok=True)
        (collection_dir / "input").mkdir(exist_ok=True)

        cli_overrides = {
            "input.storage.type": "blob",
            "input.storage.connection_string": conn_str,
            "input.storage.container_name": container_name,
            "input.storage.base_dir": "input",
            "input.file_pattern": ".*\\.(txt|md)$",
            "output.type": "blob",
            "output.connection_string": conn_str,
            "output.container_name": container_name,
            "output.base_dir": "output",
            "cache.type": "blob",
            "cache.connection_string": conn_str,
            "cache.container_name": container_name,
            "cache.base_dir": "cache",
            "reporting.type": "blob",
            "reporting.connection_string": conn_str,
            "reporting.container_name": container_name,
            "reporting.base_dir": "logs",
            "vector_store.default_vector_store.container_name": collection_id,
        }
    else:
        storage_root = settings.collections_dir.resolve()
        collection_dir = storage_root / collection_id
        collection_dir.mkdir(parents=True, exist_ok=True)
        cli_overrides = {
            "input.storage.type": "file",
            "input.storage.base_dir": str(collection_dir / "input"),
            "input.file_pattern": ".*\\.(txt|md)$",
            "output.type": "file",
            "output.base_dir": str(collection_dir / "output"),
            "cache.type": "file",
            "cache.base_dir": str(collection_dir / "cache"),
        }

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
    use_blob = bool(settings.azure_storage_connection_string)

    required_files = [
        "entities.parquet",
        "communities.parquet",
        "community_reports.parquet",
    ]
    if method in ["local", "drift", "tog"]:
        required_files.extend(["text_units.parquet", "relationships.parquet"])

    if use_blob:
        for fname in required_files:
            if not _blob_file_exists(collection_id, f"output/{fname}"):
                return False, f"Collection has not been indexed yet (missing {fname} in blob)"
        return True, None
    else:
        collection_dir = settings.collections_dir / collection_id
        output_dir = collection_dir / "output"
        if not output_dir.exists():
            return False, "Collection has not been indexed yet"
        missing = [f for f in required_files if not (output_dir / f).exists()]
        if missing:
            return False, f"Missing indexed files: {', '.join(missing)}"
        return True, None


def get_search_data_paths(collection_id: str, method: str) -> Dict[str, Path]:
    """
    Get paths to required parquet files for a search method.
    When Azure blob is configured, downloads parquets to a local cache dir first.

    Args:
        collection_id: The collection identifier
        method: The search method (global, local, tog, drift)

    Returns:
        Dictionary of data file paths (always local paths for pandas compatibility)
    """
    use_blob = bool(settings.azure_storage_connection_string)

    file_names = {
        "entities": "entities.parquet",
        "communities": "communities.parquet",
        "community_reports": "community_reports.parquet",
    }
    if method in ["local", "drift", "tog"]:
        file_names["text_units"] = "text_units.parquet"
        file_names["relationships"] = "relationships.parquet"

    if use_blob:
        # Download from blob to local cache dir
        cache_dir = settings.collections_dir / collection_id / "output"
        cache_dir.mkdir(parents=True, exist_ok=True)

        client = _blob_client()
        container = client.get_container_client(_collection_container(collection_id))

        paths = {}
        for key, fname in file_names.items():
            local_path = cache_dir / fname
            if not local_path.exists():
                logger.info("Downloading %s from blob for collection %s", fname, collection_id)
                data = container.get_blob_client(f"output/{fname}").download_blob().readall()
                local_path.write_bytes(data)
            paths[key] = local_path

        if method == "local":
            cov_blob = f"output/covariates.parquet"
            cov_local = cache_dir / "covariates.parquet"
            if not cov_local.exists() and _blob_file_exists(collection_id, cov_blob):
                data = container.get_blob_client(cov_blob).download_blob().readall()
                cov_local.write_bytes(data)
            if cov_local.exists():
                paths["covariates"] = cov_local

        if method == "tog":
            missing = [k for k in ["entities", "relationships"] if not paths.get(k, Path("x")).exists()]
            if missing:
                raise FileNotFoundError(f"ToG search requires missing files: {', '.join(missing)}")
    else:
        output_dir = settings.collections_dir / collection_id / "output"
        paths = {key: output_dir / fname for key, fname in file_names.items()}

        if method == "local":
            cov = output_dir / "covariates.parquet"
            if cov.exists():
                paths["covariates"] = cov

        if method == "tog":
            missing = [f for f in ["entities.parquet", "relationships.parquet"] if not (output_dir / f).exists()]
            if missing:
                raise FileNotFoundError(f"ToG search requires missing files: {', '.join(missing)}")

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
