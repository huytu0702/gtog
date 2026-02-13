"""Integration tests for collection/document lifecycle in Cosmos mode."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_cosmos_container():
    """Mock Cosmos DB container with in-memory storage."""
    storage = {}
    
    def upsert_item(item):
        storage[item["id"]] = item
        return item
    
    def read_item(item, partition_key):
        if item in storage:
            return storage[item]
        raise Exception("Not found")
    
    def query_items(query, **kwargs):
        # Simple query parsing - just return all items
        items = list(storage.values())
        return items
    
    def delete_item(item, partition_key):
        if item in storage:
            del storage[item]
    
    container = MagicMock()
    container.upsert_item = MagicMock(side_effect=upsert_item)
    container.read_item = MagicMock(side_effect=read_item)
    container.query_items = MagicMock(side_effect=query_items)
    container.delete_item = MagicMock(side_effect=delete_item)
    container._storage = storage
    
    return container


@pytest.fixture
def test_client_cosmos(mock_cosmos_container, tmp_path):
    """Create a test client with cosmos mode enabled."""
    with patch("app.config.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "test-db"
        mock_settings.cosmos_container = "test-container"
        mock_settings.collections_dir = tmp_path
        mock_settings.settings_yaml_path = tmp_path / "settings.yaml"
        mock_settings.openai_api_key = "test-key"
        mock_settings.default_chat_model = "gpt-4o-mini"
        mock_settings.google_api_key = ""
        mock_settings.tavily_api_key = ""
        
        with patch("azure.cosmos.CosmosClient") as mock_client_class:
            mock_client = MagicMock()
            mock_database = MagicMock()
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_cosmos_container
            mock_client_class.return_value = mock_client
            
            # Clear any cached storage service instance
            import app.services.storage_service as storage_module
            storage_module._storage_service_instance = None
            
            # Import and create app after all mocks are set up
            from app.main import app
            
            with TestClient(app) as client:
                yield client, mock_cosmos_container


def test_create_collection_api(test_client_cosmos):
    """Test POST /collections creates a collection in cosmos mode."""
    client, container = test_client_cosmos
    
    response = client.post("/collections", json={
        "collection_id": "test-collection",
        "description": "Test description"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-collection"
    assert data["name"] == "test-collection"
    assert data["description"] == "Test description"
    assert data["document_count"] == 0
    assert data["indexed"] is False


def test_list_collections_api(test_client_cosmos):
    """Test GET /collections lists collections from cosmos."""
    client, container = test_client_cosmos
    
    # Create two collections
    client.post("/collections", json={"collection_id": "col1", "description": "First"})
    client.post("/collections", json={"collection_id": "col2", "description": "Second"})
    
    response = client.get("/collections")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    ids = [c["id"] for c in data]
    assert "col1" in ids
    assert "col2" in ids


def test_get_collection_api(test_client_cosmos):
    """Test GET /collections/{id} retrieves a collection."""
    client, container = test_client_cosmos
    
    # Create collection
    client.post("/collections", json={"collection_id": "my-collection"})
    
    response = client.get("/collections/my-collection")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "my-collection"


def test_upload_and_list_documents_api(test_client_cosmos):
    """Test uploading and listing documents in cosmos mode."""
    client, container = test_client_cosmos
    
    # Create collection first
    client.post("/collections", json={"collection_id": "doc-test"})
    
    # Upload a document
    response = client.post(
        "/collections/doc-test/upload",
        files={"file": ("test.txt", b"Test document content", "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test.txt"
    assert data["size"] == len(b"Test document content")
    
    # List documents
    response = client.get("/collections/doc-test/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test.txt"


def test_delete_document_api(test_client_cosmos):
    """Test deleting a document in cosmos mode."""
    client, container = test_client_cosmos
    
    # Setup
    client.post("/collections", json={"collection_id": "delete-test"})
    client.post(
        "/collections/delete-test/upload",
        files={"file": ("delete-me.txt", b"Content", "text/plain")}
    )
    
    # Verify document exists
    response = client.get("/collections/delete-test/documents")
    assert len(response.json()) == 1
    
    # Delete document
    response = client.delete("/collections/delete-test/documents/delete-me.txt")
    assert response.status_code == 204
    
    # Verify document is gone
    response = client.get("/collections/delete-test/documents")
    assert len(response.json()) == 0


def test_delete_collection_api(test_client_cosmos):
    """Test deleting a collection in cosmos mode."""
    client, container = test_client_cosmos
    
    # Setup
    client.post("/collections", json={"collection_id": "delete-col"})
    
    # Delete collection
    response = client.delete("/collections/delete-col")
    assert response.status_code == 204
    
    # Verify collection is gone
    response = client.get("/collections/delete-col")
    assert response.status_code == 404


def test_full_collection_document_lifecycle(test_client_cosmos):
    """Test complete lifecycle: create, upload, list, delete doc, delete collection."""
    client, container = test_client_cosmos
    
    # 1. Create collection
    response = client.post("/collections", json={
        "collection_id": "lifecycle-test",
        "description": "Lifecycle test collection"
    })
    assert response.status_code == 200
    
    # 2. Upload document
    response = client.post(
        "/collections/lifecycle-test/upload",
        files={"file": ("doc1.txt", b"First document", "text/plain")}
    )
    assert response.status_code == 200
    
    # 3. List documents
    response = client.get("/collections/lifecycle-test/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["name"] == "doc1.txt"
    
    # 4. Get collection details
    response = client.get("/collections/lifecycle-test")
    assert response.status_code == 200
    col = response.json()
    assert col["id"] == "lifecycle-test"
    assert col["document_count"] == 1
    
    # 5. Delete document
    response = client.delete("/collections/lifecycle-test/documents/doc1.txt")
    assert response.status_code == 204
    
    # 6. Verify document deleted
    response = client.get("/collections/lifecycle-test/documents")
    assert len(response.json()) == 0
    
    # 7. Delete collection
    response = client.delete("/collections/lifecycle-test")
    assert response.status_code == 204
    
    # 8. Verify collection deleted
    response = client.get("/collections/lifecycle-test")
    assert response.status_code == 404
