"""Tests for query service storage mode awareness."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mock_settings_cosmos():
    """Mock settings for cosmos mode."""
    mock = MagicMock()
    mock.storage_mode = "cosmos"
    mock.collections_dir = Path("/tmp/collections")
    mock.settings_yaml_path = Path("settings.yaml")
    type(mock).is_cosmos_mode = property(lambda _: True)
    return mock


@pytest.fixture
def mock_config():
    """Mock GraphRAG config."""
    config = MagicMock()
    config.output = MagicMock()
    return config


class TestQueryServiceStorageMode:
    """Test that query service uses storage abstraction, not direct parquet reads."""

    @patch("app.services.query_service.pd.read_parquet")
    def test_global_search_does_not_call_read_parquet_in_cosmos_mode(
        self, mock_read_parquet, monkeypatch, mock_settings_cosmos, mock_config
    ):
        """Global search should not call pd.read_parquet when in cosmos mode."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        async def run_test():
            with patch("app.services.query_service.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                with patch("app.services.query_service.load_graphrag_config", return_value=mock_config):
                    with patch("app.services.query_service.validate_collection_indexed", return_value=(True, None)):
                        with patch("app.services.query_service.create_storage_from_config") as mock_create_storage:
                            mock_storage = MagicMock()
                            mock_storage.has = AsyncMock(return_value=True)
                            mock_storage.get = AsyncMock(return_value=b"parquet_data")
                            mock_create_storage.return_value = mock_storage
                            
                            with patch("app.services.query_service.load_table_from_storage") as mock_load_table:
                                # Return empty dataframes for all tables
                                mock_load_table.return_value = pd.DataFrame()
                                
                                with patch("app.services.query_service.api.global_search") as mock_search:
                                    mock_search.return_value = ("response text", {})
                                    
                                    from app.services.query_service import QueryService
                                    svc = QueryService()
                                    return await svc.global_search("demo", "test query")
        
        result = asyncio.run(run_test())
        mock_read_parquet.assert_not_called()
        assert result.response == "response text"

    @patch("app.services.query_service.pd.read_parquet")
    def test_local_search_does_not_call_read_parquet_in_cosmos_mode(
        self, mock_read_parquet, monkeypatch, mock_settings_cosmos, mock_config
    ):
        """Local search should not call pd.read_parquet when in cosmos mode."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        async def run_test():
            with patch("app.services.query_service.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                with patch("app.services.query_service.load_graphrag_config", return_value=mock_config):
                    with patch("app.services.query_service.validate_collection_indexed", return_value=(True, None)):
                        with patch("app.services.query_service.create_storage_from_config") as mock_create_storage:
                            mock_storage = MagicMock()
                            mock_storage.has = AsyncMock(return_value=True)
                            mock_storage.get = AsyncMock(return_value=b"parquet_data")
                            mock_create_storage.return_value = mock_storage
                            
                            with patch("app.services.query_service.load_table_from_storage") as mock_load_table:
                                mock_load_table.return_value = pd.DataFrame()
                                
                                with patch("app.services.query_service.api.local_search") as mock_search:
                                    mock_search.return_value = ("local response", {})
                                    
                                    from app.services.query_service import QueryService
                                    svc = QueryService()
                                    return await svc.local_search("demo", "test query")
        
        result = asyncio.run(run_test())
        mock_read_parquet.assert_not_called()
        assert result.response == "local response"

    @patch("app.services.query_service.pd.read_parquet")
    def test_tog_search_does_not_call_read_parquet_in_cosmos_mode(
        self, mock_read_parquet, monkeypatch, mock_settings_cosmos, mock_config
    ):
        """ToG search should not call pd.read_parquet when in cosmos mode."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        async def run_test():
            with patch("app.services.query_service.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                with patch("app.services.query_service.load_graphrag_config", return_value=mock_config):
                    with patch("app.services.query_service.validate_collection_indexed", return_value=(True, None)):
                        with patch("app.services.query_service.create_storage_from_config") as mock_create_storage:
                            mock_storage = MagicMock()
                            mock_storage.has = AsyncMock(return_value=True)
                            mock_storage.get = AsyncMock(return_value=b"parquet_data")
                            mock_create_storage.return_value = mock_storage
                            
                            with patch("app.services.query_service.load_table_from_storage") as mock_load_table:
                                mock_load_table.return_value = pd.DataFrame({"title": ["Entity1"]})
                                
                                with patch("app.services.query_service.api.tog_search") as mock_search:
                                    mock_search.return_value = ("tog response", {})
                                    
                                    from app.services.query_service import QueryService
                                    svc = QueryService()
                                    return await svc.tog_search("demo", "test query")
        
        result = asyncio.run(run_test())
        mock_read_parquet.assert_not_called()
        assert result.response == "tog response"

    @patch("app.services.query_service.pd.read_parquet")
    def test_drift_search_does_not_call_read_parquet_in_cosmos_mode(
        self, mock_read_parquet, monkeypatch, mock_settings_cosmos, mock_config
    ):
        """DRIFT search should not call pd.read_parquet when in cosmos mode."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        async def run_test():
            with patch("app.services.query_service.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                with patch("app.services.query_service.load_graphrag_config", return_value=mock_config):
                    with patch("app.services.query_service.validate_collection_indexed", return_value=(True, None)):
                        with patch("app.services.query_service.create_storage_from_config") as mock_create_storage:
                            mock_storage = MagicMock()
                            mock_storage.has = AsyncMock(return_value=True)
                            mock_storage.get = AsyncMock(return_value=b"parquet_data")
                            mock_create_storage.return_value = mock_storage
                            
                            with patch("app.services.query_service.load_table_from_storage") as mock_load_table:
                                mock_load_table.return_value = pd.DataFrame()
                                
                                with patch("app.services.query_service.api.drift_search") as mock_search:
                                    mock_search.return_value = ("drift response", {})
                                    
                                    from app.services.query_service import QueryService
                                    svc = QueryService()
                                    return await svc.drift_search("demo", "test query")
        
        result = asyncio.run(run_test())
        mock_read_parquet.assert_not_called()
        assert result.response == "drift response"
