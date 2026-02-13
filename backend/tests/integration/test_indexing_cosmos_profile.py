"""Integration test for indexing startup path under cosmos profile."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

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
    
    def query_items(query, parameters=None, **kwargs):
        """Simple query implementation that filters by collection_id if provided."""
        items = list(storage.values())
        
        # Extract collection_id from parameters if present
        collection_id = None
        if parameters:
            for param in parameters:
                if param.get("name") == "@collection_id":
                    collection_id = param.get("value")
                    break
        
        # Filter items by collection_id and kind=document
        if collection_id:
            items = [
                item for item in items 
                if item.get("collection_id") == collection_id and item.get("kind") == "document"
            ]
        
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
def test_client_cosmos_indexing(mock_cosmos_container, tmp_path):
    """Create a test client with cosmos mode enabled and indexing mocked."""
    # Create settings file
    settings_file = tmp_path / "settings.cosmos-emulator.yaml"
    settings_file.write_text("""
models:
  default_chat_model:
    type: chat
    model_provider: gemini
    api_key: test-key
    model: gemini-2.5-flash
  default_embedding_model:
    type: embedding
    model_provider: gemini
    api_key: test-key
    model: gemini-embedding-001

input:
  storage:
    type: file
    base_dir: "input"

output:
  type: cosmosdb
  base_dir: test-db
  connection_string: "AccountEndpoint=https://localhost:8081;AccountKey=test-key;"
  container_name: test-container
     
cache:
  type: cosmosdb
  base_dir: test-db
  connection_string: "AccountEndpoint=https://localhost:8081;AccountKey=test-key;"
  container_name: test-container

vector_store:
  default_vector_store:
    type: cosmosdb
    url: https://localhost:8081
    database_name: test-db
    container_name: test-container
    overwrite: true
""")
    
    with patch("app.config.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "test-db"
        mock_settings.cosmos_container = "test-container"
        mock_settings.collections_dir = tmp_path / "collections"
        mock_settings.collections_dir.mkdir(parents=True, exist_ok=True)
        mock_settings.settings_yaml_path = settings_file
        mock_settings.openai_api_key = "test-key"
        mock_settings.default_chat_model = "gpt-4o-mini"
        mock_settings.google_api_key = ""
        mock_settings.tavily_api_key = "test-tavily-key"
        
        with patch("azure.cosmos.CosmosClient") as mock_client_class:
            mock_client = MagicMock()
            mock_database = MagicMock()
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_cosmos_container
            mock_client_class.return_value = mock_client
            
            # Mock GraphRAG indexing to avoid actual LLM calls
            with patch("app.services.indexing_service.api.build_index", new_callable=AsyncMock) as mock_build:
                mock_build.return_value = [MagicMock(errors=[])]
                
                # Clear any cached storage service instance
                import app.services.storage_service as storage_module
                storage_module._storage_service_instance = None
                
                # Import and create app after all mocks are set up
                from app.main import app
                
                with TestClient(app) as client:
                    yield client, mock_cosmos_container


def test_indexing_startup_with_cosmos_profile(test_client_cosmos_indexing):
    """Test indexing startup path under cosmos profile returns 202 without config errors."""
    client, container = test_client_cosmos_indexing
    
    # 1. Create collection (using /api prefix, with correct field name 'name')
    response = client.post("/api/collections", json={
        "name": "index-test",
        "description": "Test collection for indexing"
    })
    assert response.status_code == 201, f"Failed to create collection: {response.text}"
    
    # 2. Upload a document (using /documents endpoint)
    response = client.post(
        "/api/collections/index-test/documents",
        files={"file": ("test.txt", b"Test document content for indexing.", "text/plain")}
    )
    assert response.status_code == 201, f"Failed to upload document: {response.text}"
    
    # 3. Start indexing (POST to /index endpoint)
    response = client.post("/api/collections/index-test/index")
    
    # Should return 202 Accepted
    assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
    
    data = response.json()
    # Status should be running or completed (mock returns completed quickly)
    assert data["status"] in ["running", "completed"], f"Unexpected status: {data['status']}"
    assert data["collection_id"] == "index-test"
    
    # Should not have any config-related errors
    assert "error" not in data or data.get("error") is None, f"Indexing error: {data.get('error')}"


def test_indexing_config_uses_cosmos_output(test_client_cosmos_indexing):
    """Test that indexing uses cosmos output configuration from profile."""
    client, container = test_client_cosmos_indexing
    
    # Setup collection and document (using correct field name 'name')
    client.post("/api/collections", json={"name": "config-test"})
    client.post(
        "/api/collections/config-test/documents",
        files={"file": ("doc.txt", b"Content", "text/plain")}
    )
    
    # Start indexing - the mock in the fixture already handles build_index
    response = client.post("/api/collections/config-test/index")
    assert response.status_code == 202
    
    data = response.json()
    # Verify indexing was started successfully (no config errors)
    assert data["status"] in ["running", "completed"]
    assert data["collection_id"] == "config-test"
    assert "error" not in data or data.get("error") is None
