"""Cosmos DB repository for serving-context data."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey

from ..config import settings


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _normalize_value(value: Any) -> Any:
    """Normalize dataframe values for Cosmos JSON serialization."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in row.items()}


class CosmosServingRepository:
    """Repository for entities/relationships/text-units serving context."""

    def __init__(
        self,
        *,
        connection_string: str,
        endpoint: str,
        key: str,
        database_name: str,
        entities_container: str,
        relationships_container: str,
        text_units_container: str,
        communities_container: str,
        community_reports_container: str,
        covariates_container: str,
    ) -> None:
        if connection_string:
            self._client = CosmosClient.from_connection_string(connection_string)
        elif endpoint and key:
            self._client = CosmosClient(url=endpoint, credential=key)
        else:
            raise ValueError(
                "Cosmos DB is not configured. Set AZURE_COSMOS_CONNECTION_STRING "
                "or AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY."
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
            source_id = (
                record.get("id")
                or record.get("title")
                or record.get("name")
                or record.get("short_id")
                or str(idx)
            )
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
    if settings.azure_cosmos_connection_string or (
        settings.azure_cosmos_endpoint and settings.azure_cosmos_key
    ):
        return CosmosServingRepository(
            connection_string=settings.azure_cosmos_connection_string,
            endpoint=settings.azure_cosmos_endpoint,
            key=settings.azure_cosmos_key,
            database_name=settings.azure_cosmos_database_name,
            entities_container=settings.azure_cosmos_entities_container,
            relationships_container=settings.azure_cosmos_relationships_container,
            text_units_container=settings.azure_cosmos_text_units_container,
            communities_container=settings.azure_cosmos_communities_container,
            community_reports_container=settings.azure_cosmos_community_reports_container,
            covariates_container=settings.azure_cosmos_covariates_container,
        )
    return None


def require_serving_repository() -> CosmosServingRepository:
    """Return configured serving repository or raise a runtime error."""
    repository = get_serving_repository()
    if repository is None:
        raise RuntimeError(
            "Azure Cosmos DB is required for serving context storage. "
            "Configure AZURE_COSMOS_CONNECTION_STRING or "
            "AZURE_COSMOS_ENDPOINT + AZURE_COSMOS_KEY."
        )
    return repository
