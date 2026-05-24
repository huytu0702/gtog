from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.base import VectorStoreDocument

from backend.app.vector_stores.scoped_cosmosdb import ScopedCosmosDBVectorStore


@pytest.fixture
def schema_config() -> VectorStoreSchemaConfig:
    return VectorStoreSchemaConfig(
        index_name="vec-c1-v1-entity-description",
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=3072,
    )


@pytest.fixture
def store(schema_config: VectorStoreSchemaConfig) -> ScopedCosmosDBVectorStore:
    return ScopedCosmosDBVectorStore(vector_store_schema_config=schema_config)


def test_connect_uses_fixed_vectors_container(store: ScopedCosmosDBVectorStore) -> None:
    container_client = MagicMock()
    database_client = MagicMock()
    database_client.get_container_client.return_value = container_client
    cosmos_client = MagicMock()
    cosmos_client.get_database_client.return_value = database_client
    cosmos_client.create_database_if_not_exists.return_value = None
    store._cosmos_client = cosmos_client

    store._create_database = MagicMock()
    store._create_container = MagicMock()
    store.connect = ScopedCosmosDBVectorStore.connect.__get__(store, ScopedCosmosDBVectorStore)

    from unittest.mock import patch

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient.from_connection_string", return_value=cosmos_client):
        store.connect(
            connection_string="AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key;",
            database_name="gtog-control",
            container_name="ignored-dynamic-name",
            collection_id="c1",
            version="v1",
            collection_version="c1:v1",
            client_kwargs={"connection_timeout": 30},
        )

    assert store._container_name == "vectors"
    assert store._partition_key_value == "c1:v1|entity.description"
    store._create_container.assert_called_once()


def test_connect_uses_scope_from_client_kwargs_when_top_level_scope_missing(store: ScopedCosmosDBVectorStore) -> None:
    container_client = MagicMock()
    database_client = MagicMock()
    database_client.get_container_client.return_value = container_client
    cosmos_client = MagicMock()
    cosmos_client.get_database_client.return_value = database_client
    cosmos_client.create_database_if_not_exists.return_value = None
    store._create_database = MagicMock()
    store._create_container = MagicMock()
    store.connect = ScopedCosmosDBVectorStore.connect.__get__(store, ScopedCosmosDBVectorStore)

    from unittest.mock import patch

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient", return_value=cosmos_client):
        store.connect(
            url="https://example.documents.azure.com:443/",
            key="key123",
            database_name="gtog-control",
            client_kwargs={
                "__collection_id": "c1",
                "__version": "v1",
                "__collection_version": "c1:v1",
            },
        )

    assert store._partition_key_value == "c1:v1|entity.description"
    assert store._collection_version == "c1:v1"



def test_connect_resolves_community_embedding_scope_from_index_name(schema_config: VectorStoreSchemaConfig) -> None:
    schema_config.index_name = "vectors-community-full_content"
    store = ScopedCosmosDBVectorStore(vector_store_schema_config=schema_config)
    store._create_database = MagicMock()
    store._create_container = MagicMock()

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient"):
        store.connect(
            url="https://example.documents.azure.com:443/",
            key="key123",
            database_name="gtog-control",
            collection_id="c1",
            version="v1",
        )

    assert store._partition_key_value == "c1:v1|community.full_content"



def test_connect_resolves_relationship_embedding_scope_from_index_name(schema_config: VectorStoreSchemaConfig) -> None:
    schema_config.index_name = "vectors-relationship-description"
    store = ScopedCosmosDBVectorStore(vector_store_schema_config=schema_config)
    store._create_database = MagicMock()
    store._create_container = MagicMock()

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient"):
        store.connect(
            url="https://example.documents.azure.com:443/",
            key="key123",
            database_name="gtog-control",
            collection_id="c1",
            version="v1",
        )

    assert store._partition_key_value == "c1:v1|relationship.description"



def test_connect_resolves_text_unit_embedding_scope_from_index_name(schema_config: VectorStoreSchemaConfig) -> None:
    schema_config.index_name = "vectors-text_unit-text"
    store = ScopedCosmosDBVectorStore(vector_store_schema_config=schema_config)
    store._create_database = MagicMock()
    store._create_container = MagicMock()

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient"):
        store.connect(
            url="https://example.documents.azure.com:443/",
            key="key123",
            database_name="gtog-control",
            collection_id="c1",
            version="v1",
        )

    assert store._partition_key_value == "c1:v1|text_unit.text"



def test_connect_uses_endpoint_key_when_connection_string_missing(store: ScopedCosmosDBVectorStore) -> None:
    container_client = MagicMock()
    database_client = MagicMock()
    database_client.get_container_client.return_value = container_client
    cosmos_client = MagicMock()
    cosmos_client.get_database_client.return_value = database_client
    cosmos_client.create_database_if_not_exists.return_value = None
    store._create_database = MagicMock()
    store._create_container = MagicMock()
    store.connect = ScopedCosmosDBVectorStore.connect.__get__(store, ScopedCosmosDBVectorStore)

    from unittest.mock import patch

    with patch("backend.app.vector_stores.scoped_cosmosdb.CosmosClient", return_value=cosmos_client) as cosmos_ctor:
        store.connect(
            url="https://example.documents.azure.com:443/",
            key="key123",
            database_name="gtog-control",
            collection_id="c1",
            version="v1",
        )

    cosmos_ctor.assert_called_once_with(
        url="https://example.documents.azure.com:443/",
        credential="key123",
    )
    assert store._partition_key_value == "c1:v1|entity.description"



def test_load_documents_overwrite_deletes_only_current_partition(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v1|entity.description"
    store._collection_id = "c1"
    store._version = "v1"
    store._collection_version = "c1:v1"
    store._embedding_kind = "entity.description"
    store._container_client = MagicMock()
    store._query_items = MagicMock(return_value=[{"id": "c1:v1|entity.description|s1"}])

    documents = [
        VectorStoreDocument(
            id="s1",
            text="hello",
            vector=[0.1, 0.2],
            attributes={"kind": "entity"},
        )
    ]

    store.load_documents(documents, overwrite=True)

    store._container_client.delete_item.assert_called_once_with(
        item="c1:v1|entity.description|s1",
        partition_key="c1:v1|entity.description",
    )
    upsert_body = store._container_client.upsert_item.call_args.kwargs["body"]
    assert upsert_body["id"] == "c1:v1|entity.description|s1"
    assert upsert_body["sourceId"] == "s1"
    assert upsert_body["partitionKey"] == "c1:v1|entity.description"


def test_similarity_search_fallback_filters_to_current_partition(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v1|entity.description"
    store._container_client = MagicMock()

    def query_side_effect(*, query, parameters, partition_key):
        if "VectorDistance" in query:
            raise CosmosHttpResponseError(message="no vector support")
        assert partition_key == "c1:v1|entity.description"
        assert parameters[0]["value"] == "c1:v1|entity.description"
        return [
            {
                "id": "c1:v1|entity.description|s1",
                "sourceId": "s1",
                "text": "hello",
                "vector": [1.0, 0.0],
                "attributes": "{}",
            }
        ]

    store._query_items = MagicMock(side_effect=query_side_effect)

    results = store.similarity_search_by_vector([1.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].document.id == "s1"
    assert results[0].score == pytest.approx(1.0)


def test_search_by_id_reads_current_partition(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v2|entity.description"
    store._container_client = MagicMock()
    store._container_client.read_item.return_value = {
        "id": "c1:v2|entity.description|same-source",
        "sourceId": "same-source",
        "text": "hello",
        "vector": [0.1],
        "attributes": '{"a": 1}',
    }

    document = store.search_by_id("same-source")

    store._container_client.read_item.assert_called_once_with(
        item="c1:v2|entity.description|same-source",
        partition_key="c1:v2|entity.description",
    )
    assert document.id == "same-source"
    assert document.attributes == {"a": 1}


def test_filter_by_id_scopes_allowed_source_ids(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v1|entity.description"

    query_filter = store.filter_by_id(["a", "b"])

    assert query_filter == {
        "partitionKey": "c1:v1|entity.description",
        "sourceIds": ["a", "b"],
    }


def test_clear_deletes_scope_only(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v1|entity.description"
    store._container_client = MagicMock()
    store._query_items = MagicMock(return_value=[{"id": "c1:v1|entity.description|s1"}])

    store.clear()

    store._container_client.delete_item.assert_called_once_with(
        item="c1:v1|entity.description|s1",
        partition_key="c1:v1|entity.description",
    )


def test_search_by_id_falls_back_to_partition_query(store: ScopedCosmosDBVectorStore) -> None:
    store._partition_key_value = "c1:v1|entity.description"
    store._container_client = MagicMock()
    store._container_client.read_item.side_effect = CosmosResourceNotFoundError(message="missing")
    store._query_items = MagicMock(return_value=[
        {
            "id": "c1:v1|entity.description|s2",
            "sourceId": "s2",
            "text": "fallback",
            "vector": [0.2],
            "attributes": "{}",
        }
    ])

    document = store.search_by_id("s2")

    assert document.id == "s2"
    query_call = store._query_items.call_args.kwargs
    assert query_call["partition_key"] == "c1:v1|entity.description"
