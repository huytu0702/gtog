"""Backend-scoped Cosmos DB vector store override."""

from __future__ import annotations

import json
import logging
from typing import Any

from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.cosmos.partition_key import PartitionKey
from azure.identity import DefaultAzureCredential

from ..azure_runtime import (
    cosmos_account_url,
    cosmos_client_kwargs,
    is_managed_identity_enabled,
    resolve_cosmos_connection_string,
)
from ..config import settings
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)

_FIXED_VECTOR_CONTAINER_NAME = "vectors"
_PARTITION_KEY_FIELD = "partitionKey"
_SOURCE_ID_FIELD = "sourceId"
_COLLECTION_ID_FIELD = "collectionId"
_VERSION_FIELD = "version"
_COLLECTION_VERSION_FIELD = "collectionVersion"
_EMBEDDING_KIND_FIELD = "embeddingKind"
_SUPPORTED_EMBEDDING_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("entity-description", "entity.description"),
    ("relationship-description", "relationship.description"),
    ("community-full_content", "community.full_content"),
    ("text_unit-text", "text_unit.text"),
)


def delete_collection_vector_documents(collection_id: str) -> int:
    database_name = settings.azure_cosmos_database_name.strip()
    if not database_name:
        raise ValueError("AZURE_COSMOS_DATABASE_NAME is required for vector cleanup.")

    connection_string = resolve_cosmos_connection_string()
    client_kwargs = cosmos_client_kwargs()
    if connection_string:
        cosmos_client = CosmosClient.from_connection_string(connection_string, **client_kwargs)
    else:
        account_url = cosmos_account_url()
        if not account_url:
            raise ValueError("AZURE_COSMOS_ENDPOINT is required for vector cleanup.")

        if is_managed_identity_enabled():
            credential: Any = DefaultAzureCredential()
        elif settings.azure_cosmos_key:
            credential = settings.azure_cosmos_key
        else:
            raise ValueError(
                "Cosmos vector cleanup requires connection_string, endpoint+key, or managed identity."
            )

        cosmos_client = CosmosClient(
            url=account_url,
            credential=credential,
            **client_kwargs,
        )

    try:
        database = cosmos_client.get_database_client(database_name)
        container = database.get_container_client(_FIXED_VECTOR_CONTAINER_NAME)
        container.read()
    except CosmosResourceNotFoundError:
        cosmos_client.close()
        return 0

    try:
        deleted_count = 0
        rows = container.query_items(
            query=(
                "SELECT c.id, c.partitionKey FROM c "
                "WHERE c.collectionId = @collectionId"
            ),
            parameters=[{"name": "@collectionId", "value": collection_id}],
            enable_cross_partition_query=True,
        )
        for row in rows:
            item_id = str(row.get("id") or "").strip()
            partition_key = str(row.get(_PARTITION_KEY_FIELD) or "").strip()
            if not item_id or not partition_key:
                logger.warning(
                    "vector_cleanup_skipped_invalid_row collection=%s id=%s partitionKey=%s",
                    collection_id,
                    item_id,
                    partition_key,
                )
                continue
            container.delete_item(item=item_id, partition_key=partition_key)
            deleted_count += 1

        return deleted_count
    finally:
        cosmos_client.close()


class ScopedCosmosDBVectorStore(BaseVectorStore):
    """Store all Cosmos vectors in one scoped physical container."""

    _cosmos_client: CosmosClient
    _database_client: DatabaseProxy
    _container_client: ContainerProxy

    def __init__(self, vector_store_schema_config, **kwargs: Any) -> None:
        super().__init__(vector_store_schema_config=vector_store_schema_config, **kwargs)
        self._client_kwargs: dict[str, Any] = {}
        self._database_name = ""
        self._container_name = _FIXED_VECTOR_CONTAINER_NAME
        self._collection_id = ""
        self._version = ""
        self._collection_version = ""
        self._embedding_kind = ""
        self._partition_key_value = ""
        self._allowed_source_ids: set[str] | None = None

    def connect(self, **kwargs: Any) -> None:
        """Connect to a shared Cosmos vector container scoped by partition key."""
        connection_string = str(kwargs.get("connection_string") or "").strip()
        client_kwargs = kwargs.get("client_kwargs") or {}
        self._client_kwargs = dict(client_kwargs)
        scoped_collection_id = str(self._client_kwargs.pop("__collection_id", "") or "").strip()
        scoped_version = str(self._client_kwargs.pop("__version", "") or "").strip()
        scoped_collection_version = str(self._client_kwargs.pop("__collection_version", "") or "").strip()
        if connection_string:
            self._cosmos_client = CosmosClient.from_connection_string(
                connection_string, **self._client_kwargs
            )
        else:
            url = str(kwargs.get("url") or "").strip()
            if not url:
                raise ValueError("Either connection_string or url must be provided.")
            credential = kwargs.get("key")
            if credential is None:
                credential = DefaultAzureCredential()
            self._cosmos_client = CosmosClient(
                url=url,
                credential=credential,
                **self._client_kwargs,
            )

        database_name = str(kwargs.get("database_name") or "").strip()
        if not database_name:
            raise ValueError("Database name must be provided.")

        collection_id = str(kwargs.get("collection_id") or scoped_collection_id or "").strip()
        version = str(kwargs.get("version") or scoped_version or "").strip()
        if not collection_id or not version:
            raise ValueError(
                "Cosmos vector scope requires both collection_id and version."
            )

        collection_version = str(
            kwargs.get("collection_version") or scoped_collection_version or ""
        ).strip()
        if not collection_version:
            collection_version = f"{collection_id}:{version}"

        self._database_name = database_name
        self._container_name = _FIXED_VECTOR_CONTAINER_NAME
        self._collection_id = collection_id
        self._version = version
        self._collection_version = collection_version
        self._embedding_kind = self._resolve_embedding_kind(kwargs)
        self._partition_key_value = (
            f"{self._collection_id}:{self._version}|{self._embedding_kind}"
        )
        self._allowed_source_ids = None

        self._create_database()
        self._create_container()

    def _resolve_embedding_kind(self, kwargs: dict[str, Any]) -> str:
        configured_kind = str(kwargs.get("embedding_kind") or "").strip()
        if configured_kind:
            return configured_kind

        index_name = str(self.index_name or "").strip().lower()
        for suffix, embedding_kind in _SUPPORTED_EMBEDDING_SUFFIXES:
            if index_name.endswith(suffix):
                return embedding_kind
        raise ValueError(f"Unsupported Cosmos embedding scope for index '{self.index_name}'.")

    def _create_database(self) -> None:
        self._cosmos_client.create_database_if_not_exists(id=self._database_name)
        self._database_client = self._cosmos_client.get_database_client(self._database_name)

    def _create_container(self) -> None:
        partition_key = PartitionKey(path=f"/{_PARTITION_KEY_FIELD}", kind="Hash")
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": f"/{self.vector_field}",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": self.vector_size,
                }
            ]
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/_etag/?"},
                {"path": f"/{self.vector_field}/*"},
            ],
        }

        try:
            indexing_policy["vectorIndexes"] = [
                {"path": f"/{self.vector_field}", "type": "diskANN"}
            ]
            self._database_client.create_container_if_not_exists(
                id=self._container_name,
                partition_key=partition_key,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
            )
        except CosmosHttpResponseError:
            indexing_policy.pop("vectorIndexes", None)
            self._database_client.create_container_if_not_exists(
                id=self._container_name,
                partition_key=partition_key,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
            )

        self._container_client = self._database_client.get_container_client(self._container_name)

    def _deterministic_item_id(self, source_id: str) -> str:
        return f"{self._partition_key_value}|{source_id}"

    def _build_item(self, document: VectorStoreDocument) -> dict[str, Any]:
        source_id = str(document.id)
        return {
            self.id_field: self._deterministic_item_id(source_id),
            _SOURCE_ID_FIELD: source_id,
            _PARTITION_KEY_FIELD: self._partition_key_value,
            _COLLECTION_ID_FIELD: self._collection_id,
            _VERSION_FIELD: self._version,
            _COLLECTION_VERSION_FIELD: self._collection_version,
            _EMBEDDING_KIND_FIELD: self._embedding_kind,
            self.vector_field: document.vector,
            self.text_field: document.text,
            self.attributes_field: json.dumps(document.attributes),
        }

    def _query_items(
        self,
        *,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | None = None,
    ) -> list[dict[str, Any]]:
        query_kwargs: dict[str, Any] = {
            "query": query,
            "parameters": parameters or [],
        }
        if partition_key is not None:
            query_kwargs["partition_key"] = partition_key
        else:
            query_kwargs["enable_cross_partition_query"] = True
        return list(self._container_client.query_items(**query_kwargs))

    def _delete_scope_documents(self) -> None:
        rows = self._query_items(
            query=f"SELECT c.{self.id_field} FROM c WHERE c.{_PARTITION_KEY_FIELD} = @partitionKey",
            parameters=[{"name": "@partitionKey", "value": self._partition_key_value}],
            partition_key=self._partition_key_value,
        )
        for row in rows:
            item_id = str(row.get(self.id_field) or "")
            if item_id:
                self._container_client.delete_item(
                    item=item_id,
                    partition_key=self._partition_key_value,
                )

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into the shared container for the current scope."""
        import time

        if overwrite:
            self._delete_scope_documents()
        self._allowed_source_ids = None

        for document in documents:
            if document.vector is None:
                continue
            item = self._build_item(document)
            for attempt in range(5):
                try:
                    self._container_client.upsert_item(body=item)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(0.5 * (attempt + 1))

    def _base_vector_query(self) -> tuple[str, list[dict[str, Any]]]:
        query = (
            f"SELECT c.{self.id_field}, c.{_SOURCE_ID_FIELD}, c.{self.text_field}, "
            f"c.{self.vector_field}, c.{self.attributes_field} "
            f"FROM c WHERE c.{_PARTITION_KEY_FIELD} = @partitionKey"
        )
        parameters = [{"name": "@partitionKey", "value": self._partition_key_value}]
        if self._allowed_source_ids:
            query += f" AND ARRAY_CONTAINS(@allowedSourceIds, c.{_SOURCE_ID_FIELD})"
            parameters.append(
                {"name": "@allowedSourceIds", "value": sorted(self._allowed_source_ids)}
            )
        return query, parameters

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform scoped vector similarity search."""
        try:
            query = (
                f"SELECT TOP {k} c.{self.id_field}, c.{_SOURCE_ID_FIELD}, c.{self.text_field}, "
                f"c.{self.vector_field}, c.{self.attributes_field}, "
                f"VectorDistance(c.{self.vector_field}, @embedding) AS SimilarityScore "
                f"FROM c WHERE c.{_PARTITION_KEY_FIELD} = @partitionKey"
            )
            parameters = [
                {"name": "@embedding", "value": query_embedding},
                {"name": "@partitionKey", "value": self._partition_key_value},
            ]
            if self._allowed_source_ids:
                query += f" AND ARRAY_CONTAINS(@allowedSourceIds, c.{_SOURCE_ID_FIELD})"
                parameters.append(
                    {"name": "@allowedSourceIds", "value": sorted(self._allowed_source_ids)}
                )
            query += f" ORDER BY VectorDistance(c.{self.vector_field}, @embedding)"
            items = self._query_items(
                query=query,
                parameters=parameters,
                partition_key=self._partition_key_value,
            )
        except (CosmosHttpResponseError, ValueError):
            base_query, parameters = self._base_vector_query()
            items = self._query_items(
                query=base_query,
                parameters=parameters,
                partition_key=self._partition_key_value,
            )

            from numpy import dot
            from numpy.linalg import norm

            def cosine_similarity(a: list[float], b: list[float]) -> float:
                if norm(a) * norm(b) == 0:
                    return 0.0
                return float(dot(a, b) / (norm(a) * norm(b)))

            for item in items:
                item_vector = item.get(self.vector_field, [])
                item["SimilarityScore"] = cosine_similarity(query_embedding, item_vector)
            items = sorted(
                items,
                key=lambda item: float(item.get("SimilarityScore", 0.0)),
                reverse=True,
            )[:k]

        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=item.get(_SOURCE_ID_FIELD, ""),
                    text=item.get(self.text_field, ""),
                    vector=item.get(self.vector_field, []),
                    attributes=json.loads(item.get(self.attributes_field, "{}")),
                ),
                score=float(item.get("SimilarityScore", 0.0)),
            )
            for item in items
        ]

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform text search via embedding lookup."""
        query_embedding = text_embedder(text)
        if not query_embedding:
            return []
        return self.similarity_search_by_vector(query_embedding=query_embedding, k=k)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Restrict search results to specific source ids in the current scope."""
        if not include_ids:
            self._allowed_source_ids = None
            self.query_filter = None
            return self.query_filter

        self._allowed_source_ids = {str(value) for value in include_ids}
        self.query_filter = {
            "partitionKey": self._partition_key_value,
            "sourceIds": sorted(self._allowed_source_ids),
        }
        return self.query_filter

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for one document by original source id in the current scope."""
        source_id = str(id)
        item_id = self._deterministic_item_id(source_id)
        try:
            item = self._container_client.read_item(
                item=item_id,
                partition_key=self._partition_key_value,
            )
        except CosmosResourceNotFoundError:
            rows = self._query_items(
                query=(
                    f"SELECT TOP 1 c.{self.id_field}, c.{_SOURCE_ID_FIELD}, c.{self.text_field}, "
                    f"c.{self.vector_field}, c.{self.attributes_field} "
                    f"FROM c WHERE c.{_PARTITION_KEY_FIELD} = @partitionKey "
                    f"AND c.{_SOURCE_ID_FIELD} = @sourceId"
                ),
                parameters=[
                    {"name": "@partitionKey", "value": self._partition_key_value},
                    {"name": "@sourceId", "value": source_id},
                ],
                partition_key=self._partition_key_value,
            )
            if not rows:
                raise
            item = rows[0]
        return VectorStoreDocument(
            id=item.get(_SOURCE_ID_FIELD, ""),
            text=item.get(self.text_field, ""),
            vector=item.get(self.vector_field, []),
            attributes=json.loads(item.get(self.attributes_field, "{}")),
        )

    def clear(self) -> None:
        """Clear only the current scoped partition."""
        self._delete_scope_documents()
        self._allowed_source_ids = None
        self.query_filter = None
