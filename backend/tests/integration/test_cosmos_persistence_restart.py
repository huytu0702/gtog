"""Integration test for Cosmos-backed data persistence across service restarts.

This test verifies that data stored in Cosmos DB persists across backend service restarts.
It uses mocked Cosmos DB to simulate the persistence layer.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


# In-memory storage that persists across test functions
_persistent_storage = {
    "collections": {},
    "documents": {},
}


class PersistentMockCosmosContainer:
    """Mock Cosmos DB container that maintains state across instances."""
    
    def __init__(self, storage_dict):
        self._storage = storage_dict
    
    def upsert_item(self, item):
        self._storage[item["id"]] = item
        return item
    
    def read_item(self, item_id, partition_key=None):
        if item_id in self._storage:
            return self._storage[item_id]
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(
            status_code=404,
            message=f"Item {item_id} not found"
        )
    
    def query_items(self, query, **kwargs):
        return list(self._storage.values())
    
    def delete_item(self, item_id, partition_key=None):
        if item_id in self._storage:
            del self._storage[item_id]


@pytest.fixture
def persistent_container():
    """Fixture providing a persistent mock Cosmos container."""
    # Clear storage before each test
    _persistent_storage["collections"].clear()
    _persistent_storage["documents"].clear()
    
    container = PersistentMockCosmosContainer(_persistent_storage)
    return container


def create_client_with_persistence(tmp_path, container):
    """Create a test client with persistent Cosmos storage."""
    # Create mock settings.yaml
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text("""
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: test-key
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true
chunks:
  size: 1200
  overlap: 100
input:
  type: file
  base_dir: /tmp/input
storage:
  type: cosmos
""")
    
    # Clear modules to allow fresh import
    for module in list(sys.modules.keys()):
        if module.startswith('app'):
            del sys.modules[module]
    
    # Patch web_search at module level before any imports
    with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-tavily-key'}):
        with patch("app.services.web_search.AsyncTavilyClient") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            
            with patch("app.config.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                mock_settings.cosmos_endpoint = "https://localhost:8081"
                mock_settings.cosmos_key = "test-key"
                mock_settings.cosmos_database = "test-db"
                mock_settings.cosmos_container = "test-container"
                mock_settings.collections_dir = tmp_path
                mock_settings.settings_yaml_path = settings_yaml
                mock_settings.openai_api_key = "test-key"
                mock_settings.default_chat_model = "gpt-4o-mini"
                mock_settings.tavily_api_key = "test-tavily-key"
                
                with patch("azure.cosmos.CosmosClient") as mock_client_class:
                    mock_client = MagicMock()
                    mock_database = MagicMock()
                    mock_client.get_database_client.return_value = mock_database
                    mock_database.get_container_client.return_value = container
                    mock_client_class.return_value = mock_client
                    
                    # Clear storage service cache
                    import app.services.storage_service as storage_module
                    storage_module._storage_service_instance = None
                    
                    # Import app after all mocks
                    from app.main import app
                    
                    with TestClient(app) as client:
                        return client


class TestCosmosPersistence:
    """Test data persistence across simulated backend restarts."""
    
    def test_collection_and_document_persist_after_restart(self, persistent_container, tmp_path):
        """Test that collections and documents survive backend restart."""
        import uuid
        coll_name = f"test-coll-{uuid.uuid4().hex[:8]}"
        
        # Phase 1: Create data with first client
        client1 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Create collection
        response = client1.post("/api/collections", json={
            "name": coll_name,
            "description": "Test collection"
        })
        assert response.status_code == 201, f"Failed to create collection: {response.text}"
        
        # Upload document
        response = client1.post(
            f"/api/collections/{coll_name}/documents",
            files={"file": ("doc.txt", b"Test content", "text/plain")}
        )
        assert response.status_code == 201
        
        # Verify data exists
        response = client1.get(f"/api/collections/{coll_name}")
        assert response.status_code == 200
        assert response.json()["name"] == coll_name
        
        response = client1.get(f"/api/collections/{coll_name}/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) >= 1
        assert any(d["name"] == "doc.txt" for d in docs)
        
        # Phase 2: Create new client (simulating restart)
        client2 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Verify data still exists
        response = client2.get(f"/api/collections/{coll_name}")
        assert response.status_code == 200
        assert response.json()["name"] == coll_name
        assert response.json()["description"] == "Test collection"
        
        response = client2.get(f"/api/collections/{coll_name}/documents")
        assert response.status_code == 200
        docs = response.json()
        assert any(d["name"] == "doc.txt" for d in docs)
    
    def test_collection_deletion_persists(self, persistent_container, tmp_path):
        """Test that deletion persists across restarts."""
        client1 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Create and delete
        client1.post("/api/collections", json={"name": "delete-me"})
        response = client1.delete("/api/collections/delete-me")
        assert response.status_code == 204
        
        # Verify deleted
        response = client1.get("/api/collections/delete-me")
        assert response.status_code == 404
        
        # Restart and verify still deleted
        client2 = create_client_with_persistence(tmp_path, persistent_container)
        response = client2.get("/api/collections/delete-me")
        assert response.status_code == 404
        
        response = client2.get("/api/collections")
        assert response.status_code == 200
        assert "delete-me" not in [c["name"] for c in response.json()]
    
    def test_multiple_collections_persist(self, persistent_container, tmp_path):
        """Test multiple collections survive restart."""
        client1 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Create collections
        for i in range(3):
            client1.post("/api/collections", json={
                "name": f"coll-{i}",
                "description": f"Collection {i}"
            })
        
        # Restart
        client2 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Verify all exist
        response = client2.get("/api/collections")
        assert response.status_code == 200
        names = [c["name"] for c in response.json()]
        assert all(f"coll-{i}" in names for i in range(3))
    
    def test_document_operations_persist(self, persistent_container, tmp_path):
        """Test document CRUD persists across restarts."""
        client1 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Setup
        client1.post("/api/collections", json={"name": "docs-test"})
        client1.post(
            "/api/collections/docs-test/documents",
            files={"file": ("a.txt", b"content a", "text/plain")}
        )
        client1.post(
            "/api/collections/docs-test/documents",
            files={"file": ("b.txt", b"content b", "text/plain")}
        )
        
        # Delete one
        client1.delete("/api/collections/docs-test/documents/a.txt")
        
        # Restart
        client2 = create_client_with_persistence(tmp_path, persistent_container)
        
        # Verify only b.txt remains
        response = client2.get("/api/collections/docs-test/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["name"] == "b.txt"
