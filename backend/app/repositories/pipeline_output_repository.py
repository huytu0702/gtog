"""Repository for reading GraphRAG pipeline output directly from Cosmos DB."""

from __future__ import annotations

import asyncio
import io
import logging
import re
from functools import lru_cache

import pandas as pd

from graphrag.storage.cosmosdb_pipeline_storage import CosmosDBPipelineStorage

from ..azure_runtime import (
    cosmos_account_url,
    cosmos_client_kwargs,
    is_managed_identity_enabled,
    resolve_cosmos_connection_string,
)
from ..config import settings

logger = logging.getLogger(__name__)


def _sanitize_container_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "default"


def build_pipeline_container_name(collection_id: str, version: str) -> str:
    collection_part = _sanitize_container_part(collection_id)
    version_part = _sanitize_container_part(version)
    container_name = f"pipeline-{collection_part}-{version_part}"
    return container_name[:128]


class PipelineOutputRepository:
    """Read parquet datasets for one collection/version from Cosmos pipeline storage."""

    def __init__(self) -> None:
        self._database_name = settings.azure_cosmos_database_name.strip()

    def _ensure_cosmos_runtime(self) -> None:
        connection_string = resolve_cosmos_connection_string()
        endpoint = cosmos_account_url()
        if connection_string:
            return
        if endpoint and is_managed_identity_enabled():
            return
        if endpoint and settings.azure_cosmos_key:
            return
        raise ValueError(
            "AZURE_COSMOS runtime is required for cosmos_pipeline mode. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
            "or enable managed identity."
        )

    def _storage_for(self, collection_id: str, version: str) -> CosmosDBPipelineStorage:
        self._ensure_cosmos_runtime()
        if not self._database_name:
            raise ValueError("AZURE_COSMOS_DATABASE_NAME is required for cosmos_pipeline mode.")

        connection_string = resolve_cosmos_connection_string()
        account_url = cosmos_account_url()
        container_name = build_pipeline_container_name(collection_id, version)

        kwargs: dict[str, object] = {
            "base_dir": self._database_name,
            "container_name": container_name,
            "client_kwargs": cosmos_client_kwargs(),
        }
        if connection_string:
            kwargs["connection_string"] = connection_string
        else:
            kwargs["cosmosdb_account_url"] = account_url
        return CosmosDBPipelineStorage(**kwargs)

    @staticmethod
    def _run_storage_coro(coro):
        return asyncio.run(coro)

    def _load_parquet_bytes(self, *, collection_id: str, version: str, dataset: str) -> bytes:
        storage = self._storage_for(collection_id, version)
        key = f"{dataset}.parquet"
        payload = self._run_storage_coro(storage.get(key, as_bytes=True))
        if payload is None:
            raise FileNotFoundError(
                f"Dataset '{dataset}' is missing in pipeline output for {collection_id}:{version}."
            )
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError(
                f"Dataset '{dataset}' payload is malformed for {collection_id}:{version}."
            )
        return bytes(payload)

    def load_dataframe(self, *, collection_id: str, version: str, dataset: str) -> pd.DataFrame:
        payload = self._load_parquet_bytes(
            collection_id=collection_id,
            version=version,
            dataset=dataset,
        )
        frame = pd.read_parquet(io.BytesIO(payload))
        logger.info(
            "pipeline_context_load collection=%s version=%s dataset=%s rows=%s",
            collection_id,
            version,
            dataset,
            len(frame),
        )
        return frame

    def dataset_exists(self, *, collection_id: str, version: str, dataset: str) -> bool:
        try:
            self._load_parquet_bytes(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )
            return True
        except FileNotFoundError:
            return False

    def count_rows(self, *, collection_id: str, version: str, dataset: str) -> int:
        frame = self.load_dataframe(
            collection_id=collection_id,
            version=version,
            dataset=dataset,
        )
        return len(frame)

    def load_required_frames(
        self,
        *,
        collection_id: str,
        version: str,
        datasets: list[str],
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for dataset in datasets:
            frames[dataset] = self.load_dataframe(
                collection_id=collection_id,
                version=version,
                dataset=dataset,
            )
        return frames


@lru_cache(maxsize=1)
def get_pipeline_output_repository() -> PipelineOutputRepository:
    return PipelineOutputRepository()
