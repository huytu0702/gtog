"""Quick test to get full traceback from indexing error."""

import asyncio
import traceback
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.helpers import load_graphrag_config
import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

async def test_indexing():
    try:
        collection_id = "test_graphrag"
        print(f"Loading config for {collection_id}...")
        config = load_graphrag_config(collection_id)
        print("Config loaded successfully!")
        print(f"Input base dir: {config.input.storage.base_dir}")
        print(f"Output base dir: {config.output.base_dir}")
        
        print("\nStarting indexing...")
        outputs = await api.build_index(
            config=config,
            verbose=True,
            callbacks=[NoopWorkflowCallbacks()],
        )
        print("Indexing completed!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_indexing())
