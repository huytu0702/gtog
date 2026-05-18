"""Utility helper functions."""

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import cast

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from graphrag.config.enums import VectorStoreType
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from ..azure_runtime import (
    blob_account_url,
    cosmos_account_url,
    cosmos_client_kwargs,
    create_blob_service_client,
    get_default_credential,
    is_managed_identity_enabled,
    resolve_cosmos_connection_string,
    resolve_storage_connection_string,
)
from ..config import settings
from ..repositories import get_control_plane_repository, get_pipeline_output_repository

logger = logging.getLogger(__name__)


def _storage_connection_string() -> str:
    """Resolve Azure Storage connection string from runtime profile."""
    return resolve_storage_connection_string()


def _blob_client():
    """Return an Azure BlobServiceClient if storage auth is configured."""
    return create_blob_service_client()


def _sanitize_output_container_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "default"


def _pipeline_output_container_name(collection_id: str, version: str | None) -> str:
    collection_part = _sanitize_output_container_part(collection_id)
    version_part = _sanitize_output_container_part(version or "latest")
    return f"pipeline-{collection_part}-{version_part}"[:128]


def _input_storage_cli_overrides(
    *,
    collection_id: str,
    conn_str: str,
    use_blob_input: bool,
    collection_dir: Path,
) -> dict[str, str]:
    if not use_blob_input:
        return {
            "input.storage.type": "file",
            "input.storage.base_dir": str(collection_dir / "input"),
            "input.file_pattern": ".*\\.(txt|md)$",
        }

    container_name = _collection_container(collection_id)
    overrides = {
        "input.storage.type": "blob",
        "input.storage.container_name": container_name,
        "input.storage.base_dir": "input",
        "input.file_pattern": ".*\\.(txt|md)$",
    }
    if conn_str:
        overrides["input.storage.connection_string"] = conn_str
        return overrides

    account_url = blob_account_url()
    if not account_url:
        raise ValueError(
            "Blob input storage requires AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL for managed identity."
        )
    overrides["input.storage.storage_account_blob_url"] = account_url
    return overrides


def _cosmos_output_cli_overrides(
    *,
    collection_id: str,
    version: str | None,
) -> dict[str, str]:
    database_name = settings.azure_cosmos_database_name.strip()
    if not database_name:
        raise ValueError(
            "AZURE_COSMOS_DATABASE_NAME is required for cosmos_pipeline mode."
        )

    connection_string = resolve_cosmos_connection_string()
    account_url = cosmos_account_url()
    if not connection_string and not account_url:
        raise ValueError(
            "AZURE_COSMOS runtime is required for cosmos_pipeline mode. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or AZURE_COSMOS_ENDPOINT."
        )

    overrides = {
        "output.type": "cosmosdb",
        "output.base_dir": database_name,
        "output.container_name": _pipeline_output_container_name(collection_id, version),
        "output.client_kwargs": cosmos_client_kwargs(),
    }
    if connection_string:
        overrides["output.connection_string"] = connection_string
    else:
        overrides["output.cosmosdb_account_url"] = account_url
    return overrides


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
        raise FileNotFoundError(
            f"Azure storage not configured, cannot read {blob_path}"
        )
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
        filename
        for filename in _REQUIRED_PROMPT_FILES
        if not (prompt_dir / filename).exists()
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
    endpoint = settings.azure_search_endpoint.strip()
    if not endpoint:
        return None

    if settings.azure_search_api_key:
        credential = AzureKeyCredential(settings.azure_search_api_key)
    elif is_managed_identity_enabled():
        credential = get_default_credential()
    else:
        return None

    return SearchIndexClient(
        endpoint=endpoint,
        credential=credential,
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


def _vector_store_cli_overrides(
    vector_index_name: str,
    *,
    use_cloud_vectors: bool,
) -> dict[str, object]:
    """Build runtime vector-store overrides for local vs cloud serving/indexing."""
    overrides: dict[str, object] = {
        "vector_store.default_vector_store.container_name": vector_index_name,
    }
    cloud_vector_runtime = (
        settings.index_output_mode.lower() == "cosmos_pipeline"
        and use_cloud_vectors
    )
    if not cloud_vector_runtime:
        return overrides

    store_type = (
        settings.cloud_vector_store_type.strip().lower()
        or VectorStoreType.AzureAISearch.value
    )

    if store_type == VectorStoreType.AzureAISearch.value:
        endpoint = settings.azure_search_endpoint.strip()
        if not endpoint:
            raise ValueError(
                "Cloud/runtime vector embeddings require AZURE_SEARCH_ENDPOINT for Azure AI Search."
            )
        if not settings.azure_search_api_key and not is_managed_identity_enabled():
            raise ValueError(
                "Cloud/runtime vector embeddings require AZURE_SEARCH_API_KEY or Azure managed identity for Azure AI Search."
            )

        overrides.update({
            "vector_store.default_vector_store.type": VectorStoreType.AzureAISearch.value,
            "vector_store.default_vector_store.db_uri": None,
            "vector_store.default_vector_store.url": endpoint,
        })
        if settings.azure_search_api_key:
            overrides["vector_store.default_vector_store.api_key"] = (
                settings.azure_search_api_key
            )
        return overrides

    if store_type != VectorStoreType.CosmosDB.value:
        raise ValueError(
            "CLOUD_VECTOR_STORE_TYPE must be one of ['azure_ai_search', 'cosmosdb']."
        )

    endpoint = cosmos_account_url()
    if not endpoint:
        raise ValueError(
            "Cloud/runtime vector embeddings require AZURE_COSMOS_ENDPOINT or AZURE_COSMOS_CONNECTION_STRING for Cosmos DB."
        )
    connection_string = resolve_cosmos_connection_string()
    if not connection_string and not is_managed_identity_enabled():
        raise ValueError(
            "Cloud/runtime vector embeddings require AZURE_COSMOS_CONNECTION_STRING, AZURE_COSMOS_KEY, or Azure managed identity for Cosmos DB."
        )
    database_name = settings.azure_cosmos_database_name.strip()
    if not database_name:
        raise ValueError(
            "Cloud/runtime vector embeddings require AZURE_COSMOS_DATABASE_NAME for Cosmos DB."
        )

    overrides.update({
        "vector_store.default_vector_store.type": VectorStoreType.CosmosDB.value,
        "vector_store.default_vector_store.db_uri": None,
        "vector_store.default_vector_store.url": endpoint,
        "vector_store.default_vector_store.database_name": database_name,
        "vector_store.default_vector_store.client_kwargs": cosmos_client_kwargs(),
    })
    if connection_string:
        overrides["vector_store.default_vector_store.connection_string"] = (
            connection_string
        )
    return overrides


def load_graphrag_config(
    collection_id: str,
    version: str | None = None,
    *,
    use_cloud_vectors: bool = False,
) -> GraphRagConfig:
    """
    Load shared GraphRAG configuration with collection-specific storage overrides.
    All collections use one shared prompt folder at backend/prompts.
    """
    conn_str = _storage_connection_string()
    shared_root = settings.settings_yaml_path.parent.resolve()
    _validate_shared_prompt_files(shared_root / "prompts")
    # Keep one stable index prefix per collection to avoid index explosion
    # on Azure AI Search Free tier (max 3 indexes/service).
    vector_index_name = _build_vector_index_name(collection_id)

    storage_root = settings.collections_dir.resolve()
    collection_dir = storage_root / collection_id
    collection_dir.mkdir(parents=True, exist_ok=True)
    (collection_dir / "input").mkdir(parents=True, exist_ok=True)

    mode = settings.index_output_mode.lower()
    if mode == "local_file":
        (collection_dir / "output").mkdir(parents=True, exist_ok=True)
        (collection_dir / "cache").mkdir(parents=True, exist_ok=True)
        (collection_dir / "logs").mkdir(parents=True, exist_ok=True)

    use_blob_input = _blob_client() is not None and settings.index_output_mode.lower() == "cosmos_pipeline"
    if use_blob_input:
        _ensure_blob_container(collection_id)

    cli_overrides: dict[str, object] = {
        key: cast("object", value)
        for key, value in _input_storage_cli_overrides(
            collection_id=collection_id,
            conn_str=conn_str,
            use_blob_input=use_blob_input,
            collection_dir=collection_dir,
        ).items()
    }

    if settings.index_output_mode.lower() == "cosmos_pipeline":
        cli_overrides.update(
            {
                key: cast("object", value)
                for key, value in _cosmos_output_cli_overrides(
                    collection_id=collection_id,
                    version=version,
                ).items()
            }
        )
        cli_overrides.update(
            {
                "cache.type": "none",
                "reporting.type": "file",
                "reporting.base_dir": str(collection_dir / "logs"),
            }
        )
        (collection_dir / "logs").mkdir(parents=True, exist_ok=True)
    elif settings.index_output_mode.lower() == "local_file":
        cli_overrides.update(
            {
                "output.type": "file",
                "output.base_dir": str(collection_dir / "output"),
                "cache.type": "file",
                "cache.base_dir": str(collection_dir / "cache"),
                "reporting.type": "file",
                "reporting.base_dir": str(collection_dir / "logs"),
                "vector_store.default_vector_store.db_uri": str(
                    collection_dir / "output" / "lancedb"
                ),
            }
        )
    else:
        raise ValueError(
            "INDEX_OUTPUT_MODE must be one of ['cosmos_pipeline', 'local_file']."
        )

    cli_overrides.update(
        _vector_store_cli_overrides(
            vector_index_name,
            use_cloud_vectors=use_cloud_vectors,
        )
    )

    config = load_config(
        root_dir=str(shared_root),
        config_filepath=settings.settings_yaml_path,
        cli_overrides=cli_overrides,
    )

    _normalize_litellm_model_config(config)
    return config


def validate_collection_indexed(
    collection_id: str, method: str | None = None
) -> tuple[bool, str | None]:
    """Check if a collection has been successfully indexed."""
    _required_by_method: dict[str, list[str]] = {
        "global": ["entities", "communities", "community_reports"],
        "local": ["entities", "communities", "text_units", "relationships"],
        "drift": ["entities", "communities", "text_units", "relationships"],
        "tog": ["entities", "relationships", "text_units"],
        "basic": ["entities"],
    }
    required_datasets = _required_by_method.get(
        method or "", ["entities", "communities"]
    )

    mode = settings.index_output_mode.lower()
    if mode == "cosmos_pipeline":
        control_plane = get_control_plane_repository()
        pipeline_repo = get_pipeline_output_repository()
        if control_plane is None:
            return False, "Cosmos control-plane repository is not configured"

        collection = control_plane.get_collection(collection_id)
        if collection is None:
            return False, f"Collection '{collection_id}' not found"

        version = str(collection.get("activeVersion") or "")
        if not version:
            return (
                False,
                "Collection has not been indexed yet (no active pipeline version)",
            )

        missing = [
            dataset
            for dataset in required_datasets
            if not pipeline_repo.dataset_exists(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )
        ]
        if missing:
            return (
                False,
                "Collection active pipeline version is incomplete: "
                + ", ".join(sorted(missing)),
            )
        return True, None

    required_files = [f"{dataset}.parquet" for dataset in required_datasets]
    collection_dir = settings.collections_dir / collection_id
    output_dir = collection_dir / "output"
    if not output_dir.exists():
        return False, "Collection has not been indexed yet"
    missing = [f for f in required_files if not (output_dir / f).exists()]
    if missing:
        return False, f"Missing indexed files: {', '.join(missing)}"
    return True, None


def get_search_data_paths(collection_id: str, method: str) -> dict[str, Path]:
    """Get local parquet paths for a search method (local_file mode only)."""
    if settings.index_output_mode.lower() != "local_file":
        raise RuntimeError(
            "get_search_data_paths is only available in local_file mode."
        )

    file_names = {
        "entities": "entities.parquet",
        "communities": "communities.parquet",
        "community_reports": "community_reports.parquet",
    }
    if method in ["local", "drift", "tog"]:
        file_names["text_units"] = "text_units.parquet"
        file_names["relationships"] = "relationships.parquet"

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
            raise FileNotFoundError(
                f"ToG search requires missing files: {', '.join(missing)}"
            )

    return paths


def get_collection_info(collection_id: str) -> dict | None:
    """Deprecated helper retained for compatibility."""
    is_indexed, _ = validate_collection_indexed(collection_id)
    return {
        "id": collection_id,
        "name": collection_id,
        "document_count": 0,
        "indexed": is_indexed,
        "created_at": None,
    }
