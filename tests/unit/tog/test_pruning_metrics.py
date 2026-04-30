import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from graphrag.query.structured_search.tog_search.pruning import (
    LLMPruning, SemanticPruning, BM25Pruning, PruningMetrics
)

@pytest.mark.asyncio
async def test_llm_pruning_returns_metrics():
    """LLMPruning.score_relations should return metrics dict."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.output.content = "[8, 7, 5]"
    mock_model.achat = AsyncMock(return_value=mock_response)

    pruning = LLMPruning(model=mock_model)
    relations = [
        ("rel1 description", "target1", "OUTGOING", 1.0),
        ("rel2 description", "target2", "INCOMING", 0.5),
        ("rel3 description", "target3", "OUTGOING", 0.8),
    ]

    scored, metrics = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

    assert len(scored) == 3
    assert [relation[4] for relation in scored] == [8.0, 7.0, 5.0]
    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 1
    assert metrics.prompt_tokens > 0
    assert metrics.output_tokens > 0
    mock_model.achat.assert_awaited_once()

@pytest.mark.asyncio
async def test_semantic_pruning_returns_metrics():
    """SemanticPruning.score_relations should return metrics dict."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.aembed = AsyncMock(return_value=np.array([1.0, 0.0, 0.0]))
    mock_embedding_model.aembed_batch = AsyncMock(return_value=[
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ])

    pruning = SemanticPruning(embedding_model=mock_embedding_model)
    relations = [
        ("rel1 description", "target1", "OUTGOING", 1.0),
        ("rel2 description", "target2", "INCOMING", 0.5),
    ]

    scored, metrics = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

    assert len(scored) == 2
    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 0  # SemanticPruning uses embeddings, not LLM

@pytest.mark.asyncio
async def test_bm25_pruning_returns_metrics():
    """BM25Pruning.score_relations should return metrics dict."""
    pruning = BM25Pruning()
    relations = [
        ("rel1 description with query words", "target1", "OUTGOING", 1.0),
        ("rel2 different content", "target2", "INCOMING", 0.5),
    ]

    scored, metrics = await pruning.score_relations(
        query="query words test",
        entity_name="TestEntity",
        relations=relations,
    )

    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 0  # BM25 uses lexical matching, not LLM
    assert len(scored) == 2
