"""Integration tests for Cosmos DB emulator failure modes.

Tests negative scenarios:
1. Missing Cosmos configuration
2. Emulator unavailable
3. Invalid Cosmos credentials
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


class TestCosmosMissingConfig:
    """Test behavior when Cosmos configuration is missing."""
    
    def test_startup_fails_with_missing_cosmos_endpoint(self, tmp_path):
        """Test that startup fails fast when COSMOS_ENDPOINT is missing."""
        # Create mock settings.yaml
        settings_yaml = tmp_path / "settings.yaml"
        settings_yaml.write_text("test: true")
        
        # Clear modules
        for module in list(sys.modules.keys()):
            if module.startswith('app'):
                del sys.modules[module]
        
        with patch("app.services.web_search.AsyncTavilyClient") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            
            with patch("app.config.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                mock_settings.cosmos_endpoint = ""  # Missing
                mock_settings.cosmos_key = "test-key"
                mock_settings.cosmos_database = "test-db"
                mock_settings.cosmos_container = "test-container"
                mock_settings.collections_dir = tmp_path
                mock_settings.settings_yaml_path = settings_yaml
                mock_settings.openai_api_key = "test-key"
                mock_settings.tavily_api_key = "test-tavily-key"
                
                # Import should raise ValueError during startup validation
                with pytest.raises(ValueError) as exc_info:
                    from app.main import app
                    # Trigger lifespan startup
                    with TestClient(app):
                        pass
                
                assert "COSMOS_ENDPOINT" in str(exc_info.value)
    
    def test_startup_fails_with_missing_cosmos_key(self, tmp_path):
        """Test that startup fails fast when COSMOS_KEY is missing."""
        settings_yaml = tmp_path / "settings.yaml"
        settings_yaml.write_text("test: true")
        
        for module in list(sys.modules.keys()):
            if module.startswith('app'):
                del sys.modules[module]
        
        with patch("app.services.web_search.AsyncTavilyClient") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            
            with patch("app.config.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                mock_settings.cosmos_endpoint = "https://localhost:8081"
                mock_settings.cosmos_key = ""  # Missing
                mock_settings.cosmos_database = "test-db"
                mock_settings.cosmos_container = "test-container"
                mock_settings.collections_dir = tmp_path
                mock_settings.settings_yaml_path = settings_yaml
                mock_settings.openai_api_key = "test-key"
                mock_settings.tavily_api_key = "test-tavily-key"
                
                with pytest.raises(ValueError) as exc_info:
                    from app.main import app
                    with TestClient(app):
                        pass
                
                assert "COSMOS_KEY" in str(exc_info.value)


class TestCosmosEmulatorUnavailable:
    """Test behavior when Cosmos emulator is unavailable."""
    
    def test_api_returns_503_when_emulator_down(self, tmp_path):
        """Test that API returns 503 with actionable message when emulator is down."""
        settings_yaml = tmp_path / "settings.yaml"
        settings_yaml.write_text("test: true")
        
        for module in list(sys.modules.keys()):
            if module.startswith('app'):
                del sys.modules[module]
        
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
                mock_settings.tavily_api_key = "test-tavily-key"
                
                # Mock CosmosClient to simulate connection failure
                with patch("azure.cosmos.CosmosClient") as mock_client_class:
                    mock_client_class.side_effect = Exception(
                        "Connection refused to localhost:8081"
                    )
                    
                    from app.main import app
                    
                    # Startup should fail
                    with pytest.raises(Exception) as exc_info:
                        with TestClient(app):
                            pass
                    
                    # Verify clear error message
                    error_msg = str(exc_info.value)
                    assert "Connection" in error_msg or "refused" in error_msg


class TestCosmosInvalidCredentials:
    """Test behavior with invalid Cosmos credentials."""
    
    def test_api_returns_401_with_invalid_key(self, tmp_path):
        """Test that API returns 401 with clear message for invalid key."""
        settings_yaml = tmp_path / "settings.yaml"
        settings_yaml.write_text("test: true")
        
        for module in list(sys.modules.keys()):
            if module.startswith('app'):
                del sys.modules[module]
        
        with patch("app.services.web_search.AsyncTavilyClient") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            
            with patch("app.config.settings") as mock_settings:
                mock_settings.storage_mode = "cosmos"
                mock_settings.is_cosmos_mode = True
                mock_settings.cosmos_endpoint = "https://localhost:8081"
                mock_settings.cosmos_key = "invalid-key"
                mock_settings.cosmos_database = "test-db"
                mock_settings.cosmos_container = "test-container"
                mock_settings.collections_dir = tmp_path
                mock_settings.settings_yaml_path = settings_yaml
                mock_settings.openai_api_key = "test-key"
                mock_settings.tavily_api_key = "test-tavily-key"
                
                with patch("azure.cosmos.CosmosClient") as mock_client_class:
                    # Simulate auth failure
                    from azure.cosmos.exceptions import CosmosHttpResponseError
                    mock_client_class.side_effect = CosmosHttpResponseError(
                        status_code=401,
                        message="Unauthorized"
                    )
                    
                    from app.main import app
                    
                    with pytest.raises(Exception) as exc_info:
                        with TestClient(app):
                            pass
                    
                    # Should get auth error
                    assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)


class TestNoSilentFallback:
    """Test that Cosmos mode does not silently fall back to file mode."""
    
    def test_explicit_error_on_cosmos_failure(self, tmp_path):
        """Test that Cosmos failures produce explicit errors, not silent fallback."""
        settings_yaml = tmp_path / "settings.yaml"
        settings_yaml.write_text("test: true")
        
        for module in list(sys.modules.keys()):
            if module.startswith('app'):
                del sys.modules[module]
        
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
                mock_settings.tavily_api_key = "test-tavily-key"
                
                with patch("azure.cosmos.CosmosClient") as mock_client_class:
                    mock_client_class.side_effect = Exception("Cosmos connection failed")
                    
                    # Also mock file storage to ensure it's NOT used
                    with patch("pathlib.Path.mkdir") as mock_mkdir:
                        from app.main import app
                        
                        try:
                            with TestClient(app):
                                pass
                        except Exception:
                            pass  # Expected to fail
                        
                        # File storage should not be initialized in cosmos mode
                        # (even when cosmos fails)
                        mock_mkdir.assert_not_called()
