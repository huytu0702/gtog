import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import app.utils.helpers


def test_load_graphrag_config_file_mode_applies_file_overrides(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "file")
    # Patch settings in the helpers module
    mock_settings = MagicMock()
    mock_settings.collections_dir = Path("storage/collections")
    mock_settings.settings_yaml_path = Path("settings.yaml")
    mock_settings.storage_mode = "file"
    
    with patch.object(app.utils.helpers, "settings", mock_settings):
        with patch("app.utils.helpers.load_config") as mock_load_config:
            mock_load_config.return_value = MagicMock()
            cfg = app.utils.helpers.load_graphrag_config("demo")
            call_kwargs = mock_load_config.call_args.kwargs
            cli_overrides = call_kwargs.get("cli_overrides", {})
            assert cli_overrides.get("output.type") == "file"
            assert cli_overrides.get("cache.type") == "file"


def test_load_graphrag_config_cosmos_mode_does_not_force_file_overrides(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    # Patch settings in the helpers module
    mock_settings = MagicMock()
    mock_settings.collections_dir = Path("storage/collections")
    mock_settings.settings_yaml_path = Path("settings.yaml")
    mock_settings.storage_mode = "cosmos"
    
    with patch.object(app.utils.helpers, "settings", mock_settings):
        with patch("app.utils.helpers.load_config") as mock_load_config:
            mock_load_config.return_value = MagicMock()
            cfg = app.utils.helpers.load_graphrag_config("demo")
            call_kwargs = mock_load_config.call_args.kwargs
            cli_overrides = call_kwargs.get("cli_overrides", {})
            assert "output.type" not in cli_overrides or cli_overrides.get("output.type") != "file"
            assert "cache.type" not in cli_overrides or cli_overrides.get("cache.type") != "file"


def test_cosmos_mode_preserves_profile_output_cache_vector(monkeypatch, tmp_path):
    """Test that cosmos mode preserves cosmosdb settings from the profile file."""
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    monkeypatch.setenv("GRAPHRAG_SETTINGS_FILE", "settings.cosmos-emulator.yaml")
    
    # Create a mock config that would be returned from a cosmos profile
    mock_config = MagicMock()
    mock_config.output.type = "cosmosdb"
    mock_config.cache.type = "cosmosdb"
    
    # Setup vector_store as a dict-like object with default_vector_store
    mock_vector_store = MagicMock()
    mock_vector_store.type = "cosmosdb"
    mock_config.vector_store = {"default_vector_store": mock_vector_store}
    
    # Patch settings in the helpers module
    mock_settings = MagicMock()
    mock_settings.collections_dir = tmp_path / "collections"
    mock_settings.settings_yaml_path = tmp_path / "settings.cosmos-emulator.yaml"
    mock_settings.storage_mode = "cosmos"
    type(mock_settings).is_cosmos_mode = PropertyMock(return_value=True)
    
    with patch.object(app.utils.helpers, "settings", mock_settings):
        with patch("app.utils.helpers.load_config", return_value=mock_config):
            with patch("app.utils.helpers._normalize_litellm_model_config"):
                cfg = app.utils.helpers.load_graphrag_config("demo")
                
                assert cfg.output.type == "cosmosdb"
                assert cfg.cache.type == "cosmosdb"
                assert cfg.vector_store["default_vector_store"].type == "cosmosdb"


def test_file_mode_still_uses_file_overrides(monkeypatch, tmp_path):
    """Test that file mode applies file overrides."""
    monkeypatch.setenv("STORAGE_MODE", "file")
    monkeypatch.setenv("GRAPHRAG_SETTINGS_FILE", "settings.yaml")
    
    # Create a mock config that would be returned from a file profile
    mock_config = MagicMock()
    mock_config.output.type = "file"
    mock_config.cache.type = "file"
    mock_config.vector_store = {}
    
    # Patch settings in the helpers module
    mock_settings = MagicMock()
    mock_settings.collections_dir = tmp_path / "collections"
    mock_settings.settings_yaml_path = tmp_path / "settings.yaml"
    mock_settings.storage_mode = "file"
    type(mock_settings).is_cosmos_mode = PropertyMock(return_value=False)
    
    with patch.object(app.utils.helpers, "settings", mock_settings):
        with patch("app.utils.helpers.load_config", return_value=mock_config):
            with patch("app.utils.helpers._normalize_litellm_model_config"):
                cfg = app.utils.helpers.load_graphrag_config("demo")
                
                assert cfg.output.type == "file"
                assert cfg.cache.type == "file"
