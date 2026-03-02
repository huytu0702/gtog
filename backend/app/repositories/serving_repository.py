"""Cosmos DB repository for serving-context data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey

from ..config import settings
from ..azure_runtime import (
    bootstrap_runtime_secrets,
    cosmos_client_kwargs,
    cosmos_endpoint_credential,
    is_cosmos_configured,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _normalize_value(value: Any) -> Any:
    """Normalize dataframe values for Cosmos JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(v) for v in value]

    # Pandas/NumPy array-like values (for example embeddings) must not go
    # through boolean checks such as `pd.isna(value)` directly.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            list_value = value.tolist()
            if isinstance(list_value, list):
                return [_normalize_value(v) for v in list_value]
        except Exception:
            pass

    try:
        is_na = pd.isna(value)
        if isinstance(is_na, bool) and is_na:
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in row.items()}


def _is_missing_id(value: Any) -> bool:
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


def _source_id_from_record(record: dict[str, Any], index: int) -> str:
    for key in ("id", "title", "name", "short_id"):
        candidate = _normalize_value(record.get(key))
        if _is_missing_id(candidate):
            continue
        if isinstance(candidate, (dict, list)):
            return json.dumps(candidate, sort_keys=True)
        return str(candidate)
    return str(index)


class CosmosServingRepository:
    """Repository for entities/relationships/text-units serving context."""

    def __init__(
        self,
        *,
        connection_string: str,
        endpoint: str,
        key: str,
        credential: Any | None,
        database_name: str,
        entities_container: str,
        relationships_container: str,
        text_units_container: str,
        communities_container: str,
        community_reports_container: str,
        covariates_container: str,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        kwargs = client_kwargs or {}
        if connection_string:
            self._client = CosmosClient.from_connection_string(connection_string, **kwargs)
        elif endpoint and (key or credential):
            self._client = CosmosClient(url=endpoint, credential=key or credential, **kwargs)
        else:
            raise ValueError(
                "Cosmos DB is not configured. Set AZURE_COSMOS_CONNECTION_STRING "
                "or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
                "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
            )

        self._database = self._client.create_database_if_not_exists(id=database_name)
        self._container_names = {
            "entities": entities_container,
            "relationships": relationships_container,
            "text_units": text_units_container,
            "communities": communities_container,
            "community_reports": community_reports_container,
            "covariates": covariates_container,
        }
        self._containers = {}
        for name in self._container_names.values():
            self._containers[name] = self._database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path="/collectionId"),
            )

    def _container(self, dataset: str):
        if dataset not in self._container_names:
            raise ValueError(f"Unsupported serving dataset: {dataset}")
        return self._containers[self._container_names[dataset]]

    def _delete_dataset_version(self, *, collection_id: str, version: str, dataset: str) -> None:
        container = self._container(dataset)
        rows = list(
            container.query_items(
                query="SELECT c.id FROM c WHERE c.collectionId = @collectionId AND c.version = @version",
                parameters=[
                    {"name": "@collectionId", "value": collection_id},
                    {"name": "@version", "value": version},
                ],
                partition_key=collection_id,
            )
        )
        for row in rows:
            container.delete_item(item=row["id"], partition_key=collection_id)

    def upsert_dataframe(
        self, *, collection_id: str, version: str, dataset: str, frame: pd.DataFrame
    ) -> int:
        """Replace one dataset for a version with the dataframe rows."""
        container = self._container(dataset)
        self._delete_dataset_version(collection_id=collection_id, version=version, dataset=dataset)

        now = _utcnow_iso()
        row_count = 0
        records = frame.to_dict(orient="records")
        for idx, raw_record in enumerate(records):
            record = _normalize_row(raw_record)
            source_id = _source_id_from_record(record, idx)
            item_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{collection_id}:{version}:{dataset}:{source_id}:{idx}",
                )
            )
            container.upsert_item(
                body={
                    "id": item_id,
                    "collectionId": collection_id,
                    "version": version,
                    "sourceId": str(source_id),
                    "dataset": dataset,
                    "data": record,
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            row_count += 1
        return row_count

    def load_dataframe(self, *, collection_id: str, version: str, dataset: str) -> pd.DataFrame:
        """Load one serving dataset for collection/version as a dataframe."""
        container = self._container(dataset)
        rows = list(
            container.query_items(
                query=(
                    "SELECT c.data FROM c "
                    "WHERE c.collectionId = @collectionId AND c.version = @version"
                ),
                parameters=[
                    {"name": "@collectionId", "value": collection_id},
                    {"name": "@version", "value": version},
                ],
                partition_key=collection_id,
            )
        )
        records = [row.get("data", {}) for row in rows]
        return pd.DataFrame(records)

    def count_rows(self, *, collection_id: str, version: str, dataset: str) -> int:
        """Count rows for one dataset in collection/version."""
        container = self._container(dataset)
        result = list(
            container.query_items(
                query=(
                    "SELECT VALUE COUNT(1) FROM c "
                    "WHERE c.collectionId = @collectionId AND c.version = @version"
                ),
                parameters=[
                    {"name": "@collectionId", "value": collection_id},
                    {"name": "@version", "value": version},
                ],
                partition_key=collection_id,
            )
        )
        return int(result[0]) if result else 0

    def purge_collection(self, collection_id: str) -> None:
        """Delete all serving documents for a collection across datasets."""
        for dataset in self._container_names:
            container = self._container(dataset)
            rows = list(
                container.query_items(
                    query="SELECT c.id FROM c WHERE c.collectionId = @collectionId",
                    parameters=[{"name": "@collectionId", "value": collection_id}],
                    partition_key=collection_id,
                )
            )
            for row in rows:
                container.delete_item(item=row["id"], partition_key=collection_id)


@lru_cache(maxsize=1)
def get_serving_repository() -> CosmosServingRepository | None:
    """Return singleton serving repository when Cosmos is configured."""
    bootstrap_runtime_secrets()
    if is_cosmos_configured():
        return CosmosServingRepository(
            connection_string=settings.azure_cosmos_connection_string,
            endpoint=settings.azure_cosmos_endpoint,
            key=settings.azure_cosmos_key,
            credential=cosmos_endpoint_credential(),
            database_name=settings.azure_cosmos_database_name,
            entities_container=settings.azure_cosmos_entities_container,
            relationships_container=settings.azure_cosmos_relationships_container,
            text_units_container=settings.azure_cosmos_text_units_container,
            communities_container=settings.azure_cosmos_communities_container,
            community_reports_container=settings.azure_cosmos_community_reports_container,
            covariates_container=settings.azure_cosmos_covariates_container,
            client_kwargs=cosmos_client_kwargs(),
        )
    return None


def require_serving_repository() -> CosmosServingRepository:
    """Return configured serving repository or raise a runtime error."""
    repository = get_serving_repository()
    if repository is None:
        raise RuntimeError(
            "Azure Cosmos DB is required for serving context storage. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or "
            "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY, "
            "or enable managed identity with AZURE_USE_MANAGED_IDENTITY=true."
        )
    return repository
