import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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
