"""Utility helper functions."""

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from ..config import settings
from ..repositories import get_control_plane_repository, get_serving_repository
from ..azure_runtime import create_blob_service_client, resolve_storage_connection_string

logger = logging.getLogger(__name__)


def _storage_connection_string() -> str:
    """Resolve Azure Storage connection string from runtime profile."""
    return resolve_storage_connection_string()


def _blob_client():
    """Return an Azure BlobServiceClient if storage auth is configured."""
    return create_blob_service_client()


def _collection_container(collection_id: str) -> str:
    """Azure Blob container name for a collection's data."""
    return f"col-{collection_id}"


def _ensure_blob_container(collection_id: str) -> None:
    """Create the per-collection blob container if it doesn't exist."""
    client = _blob_client()
    if client is None:
        return
    container = client.get_container_client(_collection_container(collection_id))
    if not container.exists():
        container.create_container()


def _blob_file_exists(collection_id: str, blob_path: str) -> bool:
    """Check if a blob exists in the collection container."""
    client = _blob_client()
    if client is None:
        return False
    container = client.get_container_client(_collection_container(collection_id))
    return container.get_blob_client(blob_path).exists()


def read_parquet_from_blob(collection_id: str, blob_path: str) -> pd.DataFrame:
    """Download a parquet file from blob and return as DataFrame."""
    client = _blob_client()
    if client is None:
        raise FileNotFoundError(f"Azure storage not configured, cannot read {blob_path}")
    container = client.get_container_client(_collection_container(collection_id))
    data = container.get_blob_client(blob_path).download_blob().readall()
    return pd.read_parquet(io.BytesIO(data))


_REQUIRED_PROMPT_FILES: tuple[str, ...] = (
    "extract_graph.txt",
    "summarize_descriptions.txt",
    "extract_claims.txt",
    "community_report_graph.txt",
    "community_report_text.txt",
    "local_search_system_prompt.txt",
    "global_search_map_system_prompt.txt",
    "global_search_reduce_system_prompt.txt",
    "global_search_knowledge_system_prompt.txt",
    "drift_search_system_prompt.txt",
    "drift_search_reduce_prompt.txt",
    "basic_search_system_prompt.txt",
    "question_gen_system_prompt.txt",
    "tog_relation_scoring_prompt.txt",
    "tog_entity_scoring_prompt.txt",
    "tog_reasoning_prompt.txt",
)


def _validate_shared_prompt_files(prompt_dir: Path) -> None:
    """Fail fast when required shared prompt files are missing."""
    missing = [
        filename for filename in _REQUIRED_PROMPT_FILES if not (prompt_dir / filename).exists()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required prompt files in {prompt_dir}: {missing_list}"
        )


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

        if raw_provider in provider_aliases:
            canonical_provider = provider_aliases[raw_provider]
            model_cfg.model_provider = canonical_provider
            logger.warning(
                "Normalized model_provider for %s: '%s' -> '%s'",
                model_id,
                raw_provider,
                canonical_provider,
            )

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


def _build_vector_index_name(collection_id: str, version: str | None = None) -> str:
    """Build an Azure AI Search-safe index name for collection/version isolation."""
    base = collection_id if version is None else f"{collection_id}-{version}"
    normalized = re.sub(r"[^a-z0-9-]", "-", base.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    if not normalized:
        normalized = "gtog-index"
    if len(normalized) <= 128:
        return normalized

    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:117]}-{digest}"


def _search_index_client() -> SearchIndexClient | None:
    if not settings.azure_search_endpoint or not settings.azure_search_api_key:
        return None
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


def delete_search_indexes_for_collection(collection_id: str) -> int:
    """Delete Azure AI Search indexes associated with one collection."""
    client = _search_index_client()
    if client is None:
        return 0

    # GraphRAG creates indexes with suffixes per vector schema field.
    prefix = f"{_build_vector_index_name(collection_id)}-"
    deleted = 0
    for index_name in client.list_index_names():
        if str(index_name).startswith(prefix):
            client.delete_index(index_name)
            deleted += 1
    return deleted


def load_graphrag_config(collection_id: str, version: str | None = None) -> GraphRagConfig:
    """
    Load shared GraphRAG configuration with collection-specific storage overrides.
    All collections use one shared prompt folder at backend/prompts.
    """
    conn_str = _storage_connection_string()
    use_blob = bool(conn_str)
    shared_root = settings.settings_yaml_path.parent.resolve()
    _validate_shared_prompt_files(shared_root / "prompts")
    # Keep one stable index prefix per collection to avoid index explosion
    # on Azure AI Search Free tier (max 3 indexes/service).
    vector_index_name = _build_vector_index_name(collection_id)

    if use_blob:
        _ensure_blob_container(collection_id)
        container_name = _collection_container(collection_id)
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
            "vector_store.default_vector_store.container_name": vector_index_name,
        }
    else:
        storage_root = settings.collections_dir.resolve()
        collection_dir = storage_root / collection_id
        collection_dir.mkdir(parents=True, exist_ok=True)
        (collection_dir / "input").mkdir(parents=True, exist_ok=True)
        (collection_dir / "output").mkdir(parents=True, exist_ok=True)
        (collection_dir / "cache").mkdir(parents=True, exist_ok=True)
        cli_overrides = {
            "input.storage.type": "file",
            "input.storage.base_dir": str(collection_dir / "input"),
            "input.file_pattern": ".*\\.(txt|md)$",
            "output.type": "file",
            "output.base_dir": str(collection_dir / "output"),
            "cache.type": "file",
            "cache.base_dir": str(collection_dir / "cache"),
            "vector_store.default_vector_store.container_name": vector_index_name,
        }

    config = load_config(
        root_dir=str(shared_root),
        config_filepath=settings.settings_yaml_path,
        cli_overrides=cli_overrides,
    )

    _normalize_litellm_model_config(config)
    return config


def validate_collection_indexed(
    collection_id: str, method: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Check if a collection has been successfully indexed."""
    control_plane = get_control_plane_repository()
    serving_repo = get_serving_repository()
    if control_plane is not None and serving_repo is not None:
        collection = control_plane.get_collection(collection_id)
        if collection is None:
            return False, f"Collection '{collection_id}' not found"
        version = collection.get("activeVersion")
        if not version:
            return False, "Collection has not been indexed yet (no active serving version)"

        required_datasets = ["entities", "communities", "community_reports"]
        if method in ["local", "drift", "tog"]:
            required_datasets.extend(["text_units", "relationships"])

        missing = []
        for dataset in required_datasets:
            if serving_repo.count_rows(
                collection_id=collection_id,
                version=str(version),
                dataset=dataset,
            ) == 0:
                missing.append(dataset)

        if missing:
            return (
                False,
                "Collection active serving version is incomplete: "
                + ", ".join(sorted(missing)),
            )
        return True, None

    use_blob = bool(_storage_connection_string())

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

    collection_dir = settings.collections_dir / collection_id
    output_dir = collection_dir / "output"
    if not output_dir.exists():
        return False, "Collection has not been indexed yet"
    missing = [f for f in required_files if not (output_dir / f).exists()]
    if missing:
        return False, f"Missing indexed files: {', '.join(missing)}"
    return True, None


def get_search_data_paths(collection_id: str, method: str) -> Dict[str, Path]:
    """Get logical parquet paths for a search method."""
    use_blob = bool(_storage_connection_string())

    file_names = {
        "entities": "entities.parquet",
        "communities": "communities.parquet",
        "community_reports": "community_reports.parquet",
    }
    if method in ["local", "drift", "tog"]:
        file_names["text_units"] = "text_units.parquet"
        file_names["relationships"] = "relationships.parquet"

    if use_blob:
        paths = {key: Path(fname) for key, fname in file_names.items()}

        if method == "local" and _blob_file_exists(collection_id, "output/covariates.parquet"):
            paths["covariates"] = Path("covariates.parquet")

        if method == "tog":
            missing = [
                name
                for name in ["entities.parquet", "relationships.parquet"]
                if not _blob_file_exists(collection_id, f"output/{name}")
            ]
            if missing:
                raise FileNotFoundError(
                    f"ToG search requires missing files: {', '.join(missing)}"
                )
        return paths

    output_dir = settings.collections_dir / collection_id / "output"
    paths = {key: output_dir / fname for key, fname in file_names.items()}

    if method == "local":
        cov = output_dir / "covariates.parquet"
        if cov.exists():
            paths["covariates"] = cov

    if method == "tog":
        missing = [
            f
            for f in ["entities.parquet", "relationships.parquet"]
            if not (output_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(f"ToG search requires missing files: {', '.join(missing)}")

    return paths


def get_collection_info(collection_id: str) -> Optional[Dict]:
    """Deprecated helper retained for compatibility."""
    is_indexed, _ = validate_collection_indexed(collection_id)
    return {
        "id": collection_id,
        "name": collection_id,
        "document_count": 0,
        "indexed": is_indexed,
        "created_at": None,
    }
