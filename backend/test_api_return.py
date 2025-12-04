"""
Quick test to check GraphRAG API return structure
"""
import asyncio
import pandas as pd
import graphrag.api as api
from pathlib import Path
from graphrag.config.load_config import load_config

async def test_api_return():
    # Load config
    collection_dir = Path("storage/collections/test_graphrag")
    config = load_config(
        root_dir=str(collection_dir),
        config_filepath="settings.yaml",
        cli_overrides={
            "input.storage.type": "file",
            "input.storage.base_dir": str(collection_dir / "input"),
            "output.type": "file",  
            "output.base_dir": str(collection_dir / "output"),
            "cache.type": "file",
            "cache.base_dir": str(collection_dir / "cache"),
        }
    )
    
    # Load data
    output_dir = collection_dir / "output"
    entities = pd.read_parquet(output_dir / "entities.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    
    print("Testing global_search...")
    result = await api.global_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=None,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query="What are the main topics?",
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if isinstance(result, tuple) else 'N/A'}")
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            print(f"  Item {i}: type={type(item)}, value preview={str(item)[:200]}")
    else:
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_api_return())
