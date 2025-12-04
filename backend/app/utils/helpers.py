"""Utility helper functions."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from ..config import settings


def load_graphrag_config(collection_id: str) -> GraphRagConfig:
    """
    Load shared GraphRAG configuration and override collection-specific paths.
    
    Args:
        collection_id: The collection identifier
        
    Returns:
        GraphRagConfig with collection-specific path overrides
    """
    # Get absolute paths
    storage_root = settings.collections_dir.resolve()
    collection_dir = storage_root / collection_id
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the shared settings.yaml with collection-specific overrides
    config = load_config(
        root_dir=str(collection_dir),
        config_filepath=settings.settings_yaml_path,
        cli_overrides={
            "input.storage.type": "file",
            "input.storage.base_dir": str(collection_dir / "input"),
            "output.type": "file",  
            "output.base_dir": str(collection_dir / "output"),
            "cache.type": "file",
            "cache.base_dir": str(collection_dir / "cache"),
        }
    )
    
    return config


def validate_collection_indexed(collection_id: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a collection has been successfully indexed.
    
    Args:
        collection_id: The collection identifier
        
    Returns:
        Tuple of (is_indexed, error_message)
    """
    collection_dir = settings.collections_dir / collection_id
    output_dir = collection_dir / "output"
    
    if not output_dir.exists():
        return False, "Collection has not been indexed yet"
    
    # Check for essential parquet files
    required_files = [
        "entities.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet",
    ]
    
    missing_files = []
    for file in required_files:
        if not (output_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"Missing indexed files: {', '.join(missing_files)}"
    
    return True, None


def get_search_data_paths(collection_id: str, method: str) -> Dict[str, Path]:
    """
    Get paths to required parquet files for a search method.
    
    Args:
        collection_id: The collection identifier
        method: The search method (global, local, tog, drift)
        
    Returns:
        Dictionary of data file paths
    """
    output_dir = settings.collections_dir / collection_id / "output"
    
    # Common files for all methods
    paths = {
        "entities": output_dir / "entities.parquet",
        "communities": output_dir / "communities.parquet",
        "community_reports": output_dir / "community_reports.parquet",
    }
    
    # Method-specific files
    if method in ["local", "drift", "tog"]:
        paths.update({
            "text_units": output_dir / "text_units.parquet",
            "relationships": output_dir / "relationships.parquet",
        })
    
    if method == "local":
        # Local search may use covariates if available
        covariates_path = output_dir / "covariates.parquet"
        if covariates_path.exists():
            paths["covariates"] = covariates_path
    
    return paths


def get_collection_info(collection_id: str) -> Optional[Dict]:
    """
    Get basic information about a collection.
    
    Args:
        collection_id: The collection identifier
        
    Returns:
        Dictionary with collection info or None if not found
    """
    collection_dir = settings.collections_dir / collection_id
    
    if not collection_dir.exists():
        return None
    
    input_dir = collection_dir / "input"
    output_dir = collection_dir / "output"
    
    # Count documents
    document_count = 0
    if input_dir.exists():
        document_count = len([f for f in input_dir.iterdir() if f.is_file()])
    
    # Check if indexed
    is_indexed, _ = validate_collection_indexed(collection_id)
    
    # Get creation time
    created_at = collection_dir.stat().st_ctime
    
    return {
        "id": collection_id,
        "name": collection_id,
        "document_count": document_count,
        "indexed": is_indexed,
        "created_at": created_at,
    }
