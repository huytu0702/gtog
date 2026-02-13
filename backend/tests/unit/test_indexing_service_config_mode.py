"""Test indexing service receives cosmos-aligned config."""

import os
import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set env var before importing to avoid MissingAPIKeyError
os.environ["TAVILY_API_KEY"] = "test-tavily-key"

import pytest
from app.services.indexing_service import IndexingService
from app.models import IndexStatus, IndexStatusResponse


@patch("app.services.indexing_service.api.build_index", new_callable=AsyncMock)
@patch("app.services.indexing_service.load_graphrag_config")
@patch("app.services.indexing_service.settings")
def test_indexing_uses_cosmos_profile_config(mock_settings, mock_load, mock_build, monkeypatch, tmp_path):
    """Test that indexing service passes cosmos-aligned config to api.build_index."""
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    
    # Create a fake cosmos config
    fake_cfg = MagicMock()
    fake_cfg.output.type = "cosmosdb"
    fake_cfg.cache.type = "cosmosdb"
    mock_vector_store = MagicMock()
    mock_vector_store.type = "cosmosdb"
    fake_cfg.vector_store = {"default_vector_store": mock_vector_store}
    
    mock_load.return_value = fake_cfg
    
    # Mock settings for cosmos mode
    mock_settings.is_cosmos_mode = True
    mock_settings.collections_dir = tmp_path / "collections"
    mock_settings.collections_dir.mkdir(parents=True, exist_ok=True)
    
    svc = IndexingService()
    
    # Initialize the indexing task status (normally done by start_indexing)
    svc.indexing_tasks["demo"] = IndexStatusResponse(
        collection_id="demo",
        status=IndexStatus.RUNNING,
        progress=0.0,
        message="Starting indexing...",
        started_at=datetime.now(),
    )
    
    # Mock _sync_cosmos_documents_to_input to avoid actual file operations
    with patch.object(svc, '_sync_cosmos_documents_to_input'):
        # Run the async method
        asyncio.run(svc._run_indexing_task("demo"))
    
    # Verify build_index was called with the config
    mock_build.assert_awaited_once()
    passed_cfg = mock_build.await_args.kwargs["config"]
    
    # Verify the config has cosmos settings
    assert passed_cfg.output.type == "cosmosdb"
    assert passed_cfg.cache.type == "cosmosdb"


@patch("app.services.indexing_service.api.build_index", new_callable=AsyncMock)
@patch("app.services.indexing_service.load_graphrag_config")
@patch("app.services.indexing_service.settings")
def test_indexing_preserves_file_mode_config(mock_settings, mock_load, mock_build, monkeypatch, tmp_path):
    """Test that indexing service preserves file mode config when not in cosmos mode."""
    monkeypatch.setenv("STORAGE_MODE", "file")
    
    # Create a fake file config
    fake_cfg = MagicMock()
    fake_cfg.output.type = "file"
    fake_cfg.cache.type = "file"
    fake_cfg.vector_store = {}
    
    mock_load.return_value = fake_cfg
    
    # Mock settings for file mode
    mock_settings.is_cosmos_mode = False
    mock_settings.collections_dir = tmp_path / "collections"
    mock_settings.collections_dir.mkdir(parents=True, exist_ok=True)
    
    svc = IndexingService()
    
    # Initialize the indexing task status (normally done by start_indexing)
    svc.indexing_tasks["demo"] = IndexStatusResponse(
        collection_id="demo",
        status=IndexStatus.RUNNING,
        progress=0.0,
        message="Starting indexing...",
        started_at=datetime.now(),
    )
    
    # Run the async method
    asyncio.run(svc._run_indexing_task("demo"))
    
    # Verify build_index was called
    mock_build.assert_awaited_once()
    passed_cfg = mock_build.await_args.kwargs["config"]
    
    # Verify the config has file settings
    assert passed_cfg.output.type == "file"
    assert passed_cfg.cache.type == "file"
