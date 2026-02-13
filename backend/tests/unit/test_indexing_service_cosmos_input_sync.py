import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

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


@pytest.fixture
def mock_storage_service():
    """Mock storage service."""
    service = MagicMock()
    service._is_cosmos_mode = True
    service._document_repo = MagicMock()
    return service


def test_indexing_service_materializes_cosmos_documents_to_input_dir(mock_cosmos_container, tmp_path):
    """Test that indexing service syncs cosmos documents to local input dir."""
    from app.services.indexing_service import IndexingService
    from app.services.storage_service import StorageService
    
    # Set up mock document repo to return documents
    mock_doc_repo = MagicMock()
    doc1 = MagicMock()
    doc1.name = "doc1.txt"
    doc2 = MagicMock()
    doc2.name = "doc2.md"
    mock_doc_repo.list.return_value = [doc1, doc2]
    mock_doc_repo.get_content.side_effect = [
        b"content of doc1",
        b"content of doc2",
    ]
    
    with patch("app.services.indexing_service.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.collections_dir = tmp_path
        
        with patch("app.services.storage_service.settings") as mock_storage_settings:
            mock_storage_settings.is_cosmos_mode = True
            mock_storage_settings.cosmos_endpoint = "https://localhost:8081"
            mock_storage_settings.cosmos_key = "test-key"
            mock_storage_settings.cosmos_database = "test-db"
            mock_storage_settings.cosmos_container = "test-container"
            mock_storage_settings.collections_dir = tmp_path
            
            with patch("azure.cosmos.CosmosClient") as mock_client_class:
                mock_client = MagicMock()
                mock_database = MagicMock()
                mock_client.get_database_client.return_value = mock_database
                mock_database.get_container_client.return_value = mock_cosmos_container
                mock_client_class.return_value = mock_client
                
                # Create indexing service
                svc = IndexingService()
                
                # Manually set up the document repo mock after service creation
                svc._document_repo = mock_doc_repo
                
                # Call the sync method
                input_dir = tmp_path / "demo" / "input"
                input_dir.mkdir(parents=True, exist_ok=True)
                
                svc._sync_cosmos_documents_to_input("demo", input_dir)
                
                # Verify files were written
                assert (input_dir / "doc1.txt").exists()
                assert (input_dir / "doc2.md").exists()
                assert (input_dir / "doc1.txt").read_bytes() == b"content of doc1"
                assert (input_dir / "doc2.md").read_bytes() == b"content of doc2"


def test_sync_cosmos_documents_overwrites_existing(mock_cosmos_container, tmp_path):
    """Test that document sync overwrites existing files (idempotent)."""
    from app.services.indexing_service import IndexingService
    
    mock_doc_repo = MagicMock()
    doc = MagicMock()
    doc.name = "existing.txt"
    mock_doc_repo.list.return_value = [doc]
    mock_doc_repo.get_content.return_value = b"new content"
    
    with patch("app.services.indexing_service.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.collections_dir = tmp_path
        
        svc = IndexingService()
        svc._document_repo = mock_doc_repo
        
        input_dir = tmp_path / "demo" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-create file with old content
        (input_dir / "existing.txt").write_text("old content")
        
        # Sync should overwrite
        svc._sync_cosmos_documents_to_input("demo", input_dir)
        
        assert (input_dir / "existing.txt").read_bytes() == b"new content"
