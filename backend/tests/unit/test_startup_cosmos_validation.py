"""Test startup validation for cosmos mode configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


# Set env var before importing to avoid MissingAPIKeyError
os.environ["TAVILY_API_KEY"] = "test-tavily-key"

from app.main import _validate_startup_configuration


def test_cosmos_mode_requires_endpoint_key_database_container():
    """Test that cosmos mode fails fast when required env vars are missing."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = ""
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "gtog"
        mock_settings.cosmos_container = "graphrag"
        mock_settings.settings_yaml_path = Path("settings.yaml")
        
        with pytest.raises(ValueError) as exc_info:
            _validate_startup_configuration()
        
        assert "COSMOS_ENDPOINT" in str(exc_info.value)


def test_cosmos_mode_requires_key():
    """Test that cosmos mode fails fast when COSMOS_KEY is missing."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = ""
        mock_settings.cosmos_database = "gtog"
        mock_settings.cosmos_container = "graphrag"
        mock_settings.settings_yaml_path = Path("settings.yaml")
        
        with pytest.raises(ValueError) as exc_info:
            _validate_startup_configuration()
        
        assert "COSMOS_KEY" in str(exc_info.value)


def test_cosmos_mode_requires_database():
    """Test that cosmos mode fails fast when COSMOS_DATABASE is missing."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = ""
        mock_settings.cosmos_container = "graphrag"
        mock_settings.settings_yaml_path = Path("settings.yaml")
        
        with pytest.raises(ValueError) as exc_info:
            _validate_startup_configuration()
        
        assert "COSMOS_DATABASE" in str(exc_info.value)


def test_cosmos_mode_requires_container():
    """Test that cosmos mode fails fast when COSMOS_CONTAINER is missing."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "gtog"
        mock_settings.cosmos_container = ""
        mock_settings.settings_yaml_path = Path("settings.yaml")
        
        with pytest.raises(ValueError) as exc_info:
            _validate_startup_configuration()
        
        assert "COSMOS_CONTAINER" in str(exc_info.value)


def test_cosmos_mode_validates_settings_file_exists(tmp_path):
    """Test that cosmos mode fails fast when settings file doesn't exist."""
    non_existent_file = tmp_path / "non_existent.yaml"
    
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "gtog"
        mock_settings.cosmos_container = "graphrag"
        mock_settings.settings_yaml_path = non_existent_file
        
        with pytest.raises(ValueError) as exc_info:
            _validate_startup_configuration()
        
        assert "settings file" in str(exc_info.value).lower() or "GRAPHRAG_SETTINGS_FILE" in str(exc_info.value)


def test_cosmos_mode_passes_with_all_required_vars(tmp_path):
    """Test that cosmos mode passes validation with all required env vars set."""
    # Create a mock settings file
    settings_file = tmp_path / "settings.cosmos-emulator.yaml"
    settings_file.write_text("# mock config")
    
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = True
        mock_settings.cosmos_endpoint = "https://localhost:8081"
        mock_settings.cosmos_key = "test-key"
        mock_settings.cosmos_database = "gtog"
        mock_settings.cosmos_container = "graphrag"
        mock_settings.settings_yaml_path = settings_file
        
        # Should not raise any exception
        _validate_startup_configuration()


def test_file_mode_skips_cosmos_validation():
    """Test that file mode doesn't require cosmos env vars."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.is_cosmos_mode = False
        
        # Should not raise any exception
        _validate_startup_configuration()
