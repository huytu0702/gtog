import pytest

from graphrag.config.models.vector_store_config import VectorStoreConfig
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig


def test_cosmosdb_vector_store_accepts_default_embedding_schema_id_field():
    config = VectorStoreConfig(
        type="cosmosdb",
        url="https://example.documents.azure.com:443/",
        connection_string="AccountEndpoint=https://example.documents.azure.com:443/;AccountKey=key123;",
        database_name="gtog-control",
        embeddings_schema={
            "entity.description": VectorStoreSchemaConfig(vector_size=3072),
            "community.full_content": VectorStoreSchemaConfig(vector_size=3072),
        },
    )

    assert config.embeddings_schema["entity.description"].id_field == "id"
    assert config.embeddings_schema["community.full_content"].id_field == "id"
    assert config.connection_string is not None


def test_cosmosdb_vector_store_rejects_non_id_embedding_schema_id_field():
    with pytest.raises(ValueError, match="id_field in embeddings_schema must be 'id'"):
        VectorStoreConfig(
            type="cosmosdb",
            url="https://example.documents.azure.com:443/",
            database_name="gtog-control",
            embeddings_schema={
                "entity.description": VectorStoreSchemaConfig(
                    id_field="custom_id", vector_size=3072
                )
            },
        )
