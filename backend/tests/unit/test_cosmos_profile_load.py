"""Smoke test for cosmos profile config loading."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from graphrag.config.load_config import load_config


def test_cosmos_profile_loads_with_env(monkeypatch, tmp_path):
    """Test that cosmos emulator profile loads correctly with required env vars."""
    # Set required Cosmos DB env variables
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081")
    monkeypatch.setenv("COSMOS_KEY", "test-key")
    monkeypatch.setenv("COSMOS_DATABASE", "gtog")
    monkeypatch.setenv("COSMOS_CONTAINER", "graphrag")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "test-api-key")
    
    # Load the cosmos emulator config
    config_path = Path(__file__).parent.parent.parent / "settings.cosmos-emulator.yaml"
    
    cfg = load_config(
        root_dir=str(tmp_path),
        config_filepath=config_path
    )
    
    # Verify cosmos settings are loaded
    assert cfg.output.type == "cosmosdb"
    assert cfg.cache.type == "cosmosdb"
    assert "default_vector_store" in cfg.vector_store
    assert cfg.vector_store["default_vector_store"].type == "cosmosdb"


def test_cosmos_profile_vector_store_has_required_fields(monkeypatch, tmp_path):
    """Test that cosmos profile vector store has all required configuration fields."""
    # Set required env variables
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081")
    monkeypatch.setenv("COSMOS_KEY", "test-key")
    monkeypatch.setenv("COSMOS_DATABASE", "gtog")
    monkeypatch.setenv("COSMOS_CONTAINER", "graphrag")
    monkeypatch.setenv("GRAPHRAG_API_KEY", "test-api-key")
    
    config_path = Path(__file__).parent.parent.parent / "settings.cosmos-emulator.yaml"
    
    cfg = load_config(
        root_dir=str(tmp_path),
        config_filepath=config_path
    )
    
    vector_store = cfg.vector_store["default_vector_store"]
    
    # Verify vector store has required cosmos fields
    assert vector_store.type == "cosmosdb"
    # These should be resolved from env vars
    assert vector_store.url is not None
    assert vector_store.database_name is not None
    assert vector_store.container_name is not None
