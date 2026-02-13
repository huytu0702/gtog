import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


@pytest.fixture
def mock_cosmos_container():
    """Mock Cosmos DB container."""
    container = MagicMock()
    container.upsert_item = MagicMock(return_value={})
    container.read_item = MagicMock(side_effect=Exception("Not found"))
    container.query_items = MagicMock(return_value=[])
    container.delete_item = MagicMock(return_value=True)
    return container


def test_create_collection_cosmos_mode(mock_cosmos_container):
    """Test that creating a collection in cosmos mode works."""
    # Mock settings to return cosmos mode with valid endpoint
    with patch("app.services.storage_service.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "test-db"
        mock_settings.cosmos_container = "test-container"
        mock_settings.collections_dir = Path("./test_collections")
        
        # Mock CosmosClient
        with patch("azure.cosmos.CosmosClient") as mock_client_class:
            mock_client = MagicMock()
            mock_database = MagicMock()
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_cosmos_container
            mock_client_class.return_value = mock_client
            
            # Import and create service after mocks are set up
            from app.services.storage_service import StorageService
            svc = StorageService()
            result = svc.create_collection("demo")
            assert result.id == "demo"
            # Verify cosmos upsert was called (1 for collection + 16 for prompts)
            assert mock_cosmos_container.upsert_item.call_count >= 1


def test_list_collections_cosmos_mode(mock_cosmos_container):
    """Test listing collections in cosmos mode."""
    with patch("app.services.storage_service.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "test-db"
        mock_settings.cosmos_container = "test-container"
        mock_settings.collections_dir = Path("./test_collections")
        
        with patch("azure.cosmos.CosmosClient") as mock_client_class:
            mock_client = MagicMock()
            mock_database = MagicMock()
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_cosmos_container
            mock_client_class.return_value = mock_client
            
            # Set up collections query
            mock_cosmos_container.query_items.side_effect = [
                # First call: collections
                [
                    {"collection_id": "col1", "name": "col1", "description": "desc1", "created_at": "2024-01-01T00:00:00"},
                    {"collection_id": "col2", "name": "col2", "description": "desc2", "created_at": "2024-01-01T00:00:00"},
                ],
                # Second call: documents for col1
                [],
                # Third call: documents for col2
                [],
            ]
            
            from app.services.storage_service import StorageService
            svc = StorageService()
            collections = svc.list_collections()
            assert len(collections) == 2


def test_list_documents_cosmos_mode(mock_cosmos_container):
    """Test listing documents in cosmos mode."""
    with patch("app.services.storage_service.settings") as mock_settings:
        mock_settings.storage_mode = "cosmos"
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "test-db"
        mock_settings.cosmos_container = "test-container"
        mock_settings.collections_dir = Path("./test_collections")
        
        with patch("azure.cosmos.CosmosClient") as mock_client_class:
            mock_client = MagicMock()
            mock_database = MagicMock()
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_cosmos_container
            mock_client_class.return_value = mock_client
            
            # Set up collection exists check
            mock_cosmos_container.read_item.side_effect = None
            mock_cosmos_container.read_item.return_value = {
                "collection_id": "demo",
                "name": "demo",
                "description": None,
                "created_at": "2024-01-01T00:00:00",
            }
            
            # Set up documents query
            mock_cosmos_container.query_items.return_value = [
                {"collection_id": "demo", "name": "a.txt", "size": 5, "uploaded_at": "2024-01-01T00:00:00"},
            ]
            
            from app.services.storage_service import StorageService
            svc = StorageService()
            
            docs = svc.list_documents("demo")
            assert len(docs) == 1
            assert docs[0].name == "a.txt"
