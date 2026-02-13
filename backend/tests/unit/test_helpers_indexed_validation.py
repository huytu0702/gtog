"""Tests for indexed-state validation in cosmos mode.

These tests verify that validate_collection_indexed uses storage checks
in cosmos mode and file checks in file mode.
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.helpers import validate_collection_indexed


class TestValidateCollectionIndexedCosmosMode:
    """Test validation helper in cosmos mode."""

    def test_validate_collection_indexed_uses_storage_has_in_cosmos_mode(self, monkeypatch):
        """Test that cosmos mode uses storage.has() for validation."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "cosmos"
            
            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_storage.has = AsyncMock(return_value=True)
            
            with patch("app.utils.helpers.load_graphrag_config", return_value=mock_config):
                with patch("graphrag.utils.api.create_storage_from_config", return_value=mock_storage):
                    ok, err = asyncio.run(validate_collection_indexed("demo", method="global"))
        
        assert ok is True
        assert err is None
        # Verify storage.has was called for required tables
        mock_storage.has.assert_called()

    def test_validate_collection_indexed_reports_missing_required_artifacts(self, monkeypatch):
        """Test that cosmos mode reports missing artifacts."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "cosmos"
            
            mock_config = MagicMock()
            mock_storage = MagicMock()
            
            # entities exists, communities exists, community_reports exists, relationships missing
            async def mock_has(path):
                return "relationships" not in str(path)
            
            mock_storage.has = mock_has
            
            with patch("app.utils.helpers.load_graphrag_config", return_value=mock_config):
                with patch("graphrag.utils.api.create_storage_from_config", return_value=mock_storage):
                    ok, err = asyncio.run(validate_collection_indexed("demo", method="tog"))
        
        assert ok is False
        assert err is not None
        assert "relationships" in err.lower() or "missing" in err.lower()

    def test_validate_collection_indexed_file_mode_unchanged(self, monkeypatch, tmp_path):
        """Test that file mode still uses file existence checks."""
        monkeypatch.setenv("STORAGE_MODE", "file")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "file"
            mock_settings.collections_dir = tmp_path
            
            # Create collection structure
            collection_dir = tmp_path / "demo"
            output_dir = collection_dir / "output"
            output_dir.mkdir(parents=True)
            
            # Create required files
            (output_dir / "entities.parquet").touch()
            (output_dir / "communities.parquet").touch()
            (output_dir / "community_reports.parquet").touch()
            
            ok, err = asyncio.run(validate_collection_indexed("demo", method="global"))
        
        assert ok is True
        assert err is None

    def test_validate_collection_indexed_file_mode_missing_files(self, monkeypatch, tmp_path):
        """Test that file mode reports missing files."""
        monkeypatch.setenv("STORAGE_MODE", "file")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "file"
            mock_settings.collections_dir = tmp_path
            
            # Create collection structure but no files
            collection_dir = tmp_path / "demo"
            output_dir = collection_dir / "output"
            output_dir.mkdir(parents=True)
            
            ok, err = asyncio.run(validate_collection_indexed("demo", method="global"))
        
        assert ok is False
        assert err is not None
        assert "missing" in err.lower()


class TestValidateCollectionIndexedMethodSpecific:
    """Test method-specific validation requirements."""

    def test_global_method_requires_base_tables(self, monkeypatch):
        """Test global method only requires base tables."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "cosmos"
            
            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_storage.has = AsyncMock(return_value=True)
            
            with patch("app.utils.helpers.load_graphrag_config", return_value=mock_config):
                with patch("graphrag.utils.api.create_storage_from_config", return_value=mock_storage):
                    ok, err = asyncio.run(validate_collection_indexed("demo", method="global"))
        
        assert ok is True

    def test_local_method_requires_text_units_and_relationships(self, monkeypatch):
        """Test local method requires text_units and relationships."""
        monkeypatch.setenv("STORAGE_MODE", "cosmos")
        
        with patch("app.utils.helpers.settings") as mock_settings:
            mock_settings.storage_mode = "cosmos"
            
            mock_config = MagicMock()
            mock_storage = MagicMock()
            
            # Only base tables exist
            async def mock_has(path):
                base_tables = ["entities", "communities", "community_reports"]
                return any(table in str(path) for table in base_tables)
            
            mock_storage.has = mock_has
            
            with patch("app.utils.helpers.load_graphrag_config", return_value=mock_config):
                with patch("graphrag.utils.api.create_storage_from_config", return_value=mock_storage):
                    ok, err = asyncio.run(validate_collection_indexed("demo", method="local"))
        
        assert ok is False
        assert "text_units" in err.lower() or "relationships" in err.lower() or "missing" in err.lower()
