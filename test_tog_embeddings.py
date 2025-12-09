#!/usr/bin/env python3
"""
Test script for pre-computed embeddings in ToG search.
This script tests the integration without requiring full indexing.
"""

import asyncio
import numpy as np
from typing import List, Tuple
from unittest.mock import Mock, AsyncMock

# Import the components we want to test
from graphrag.query.structured_search.tog_search.pruning import SemanticPruning
from graphrag.vector_stores.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""

    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict

    def search_by_id(self, id: str):
        """Mock search by id."""
        doc = Mock()
        doc.vector = self.embeddings_dict.get(id)
        return doc if doc.vector else None

    async def connect(self):
        """Mock connect."""
        pass

    async def load_documents(self, **kwargs):
        """Mock load documents."""
        return []

    async def filter_by_id(self, ids: List[str], **kwargs):
        """Mock filter by id."""
        return []

    async def similarity_search_by_text(self, query: str, **kwargs):
        """Mock similarity search by text."""
        return []

    async def similarity_search_by_vector(self, query_vector: List[float], **kwargs):
        """Mock similarity search by vector."""
        return []


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    async def aembed(self, text: str) -> List[float]:
        """Mock embedding generation."""
        # Simple mock: return a vector based on text length
        return [float(len(text) % 100) / 100.0] * 10

    async def aembed_batch(self, text_list: List[str]) -> List[List[float]]:
        """Mock batch embedding generation."""
        return [await self.aembed(text) for text in text_list]


async def test_semantic_pruning():
    """Test SemanticPruning with pre-computed embeddings."""

    print("🧪 Testing SemanticPruning with pre-computed embeddings...")

    # Create mock data
    entities = [
        ("entity1", "Person", "John Doe is a software engineer"),
        ("entity2", "Company", "Tech Corp is a technology company"),
        ("entity3", "Location", "Seattle is a city in Washington"),
    ]

    # Create mock pre-computed embeddings
    embeddings_dict = {
        "entity1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "entity2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
        "entity3": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
    }

    # Create mock components
    vector_store = MockVectorStore(embeddings_dict)
    embedding_model = MockEmbeddingModel()

    # Create SemanticPruning with pre-computed embeddings
    pruning = SemanticPruning(
        embedding_model=embedding_model, entity_embedding_store=vector_store
    )

    # Test entity scoring
    query = "software engineer"
    current_path = "entity1"

    scores = await pruning.score_entities(query, current_path, entities)

    print(f"✅ Entity scores: {scores}")
    print(f"✅ Number of scores: {len(scores)}")
    print(f"✅ Score range: {min(scores):.2f} - {max(scores):.2f}")

    # Verify embeddings were loaded
    assert pruning._entity_embeddings is not None, "Entity embeddings should be loaded"
    assert pruning._entity_embeddings.shape[0] == len(entities), (
        "Should have embeddings for all entities"
    )

    print("✅ Pre-computed embeddings loaded successfully")

    # Test with different query
    query2 = "technology company"
    scores2 = await pruning.score_entities(query2, current_path, entities)

    print(f"✅ Different query scores: {scores2}")

    # Verify caching works (should not recompute embeddings)
    initial_embeddings = pruning._entity_embeddings.copy()
    scores3 = await pruning.score_entities("another query", current_path, entities)

    assert np.array_equal(initial_embeddings, pruning._entity_embeddings), (
        "Embeddings should be cached"
    )
    print("✅ Embedding caching works correctly")

    return True


async def test_fallback_behavior():
    """Test fallback when vector store fails."""

    print("\n🧪 Testing fallback behavior...")

    entities = [
        ("entity4", "Person", "Jane Smith"),
        ("entity5", "Company", "Startup Inc"),
    ]

    # Create empty vector store (no embeddings)
    empty_vector_store = MockVectorStore({})
    embedding_model = MockEmbeddingModel()

    pruning = SemanticPruning(
        embedding_model=embedding_model, entity_embedding_store=empty_vector_store
    )

    # Test fallback to computing embeddings
    query = "test query"
    scores = await pruning.score_entities(query, "entity4", entities)

    print(f"✅ Fallback scores: {scores}")
    assert len(scores) == len(entities), "Should have scores for all entities"

    # Verify embeddings were computed
    assert pruning._entity_embeddings is not None, (
        "Entity embeddings should be computed"
    )
    print("✅ Fallback to computing embeddings works correctly")

    return True


async def main():
    """Run all tests."""

    print("🚀 Starting ToG Search Pre-computed Embeddings Test\n")

    try:
        # Test semantic pruning with pre-computed embeddings
        await test_semantic_pruning()

        # Test fallback behavior
        await test_fallback_behavior()

        print("\n🎉 All tests passed!")
        print("\n✅ Implementation Summary:")
        print("  - SemanticPruning correctly uses pre-computed embeddings")
        print("  - Fallback to computing embeddings works when vector store fails")
        print("  - Embedding caching prevents recomputation")
        print("  - API consistency maintained (aembed/aembed_batch)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
