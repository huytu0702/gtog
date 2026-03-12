import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from graphrag.query.structured_search.tog_search.pruning import (
    LLMPruning, SemanticPruning, BM25Pruning, PruningMetrics
)

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration

@pytest.mark.asyncio
async def test_llm_pruning_returns_metrics():
    """LLMPruning.score_relations should return metrics dict."""
    mock_model = MagicMock()
    # The achat_stream method should return an async iterator directly (not a coroutine)
    mock_model.achat_stream = MagicMock(return_value=AsyncIteratorMock(["[8, 7, 5]"]))

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

    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 1
    assert metrics.prompt_tokens > 0
    assert metrics.output_tokens > 0

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

    _, metrics = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

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


@pytest.mark.asyncio
async def test_semantic_pruning_reloads_embeddings_when_entity_set_changes():
    """SemanticPruning should reload cached embeddings when candidate ids change."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.aembed = AsyncMock(return_value=np.array([1.0, 0.0]))
    mock_embedding_model.aembed_batch = AsyncMock(side_effect=[
        [np.array([1.0, 0.0])],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    ])

    pruning = SemanticPruning(embedding_model=mock_embedding_model)

    first_scores, _ = await pruning.score_entities(
        query="test query",
        current_path="Root",
        entities=[("e1", "Entity 1", "Desc 1")],
    )
    second_scores, _ = await pruning.score_entities(
        query="test query",
        current_path="Root -> Next",
        entities=[
            ("e2", "Entity 2", "Desc 2"),
            ("e3", "Entity 3", "Desc 3"),
        ],
    )

    assert len(first_scores) == 1
    assert len(second_scores) == 2
    assert mock_embedding_model.aembed_batch.await_count == 2


def test_llm_pruning_parse_scores_rescales_probability_outputs():
    """0..1 outputs should be rescaled to 1..10 instead of all becoming 1."""
    pruning = LLMPruning(model=MagicMock())
    scores = pruning._parse_scores("[0.0, 0.5, 1.0]", expected_count=3)
    assert scores == [1.0, 5.5, 10.0]


def test_llm_pruning_parse_scores_uses_last_list():
    """When multiple lists exist in output, parser should use the final answer list."""
    pruning = LLMPruning(model=MagicMock())
    scores = pruning._parse_scores(
        "example [9, 2, 6] final [8, 3, 7]",
        expected_count=3,
    )
    assert scores == [8.0, 3.0, 7.0]


@pytest.mark.asyncio
async def test_llm_pruning_limits_relations_for_llm_and_downweights_excluded():
    """Only top weighted relations should be sent to LLM when relation set is too large."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(return_value=AsyncIteratorMock(["[8, 7]"]))

    pruning = LLMPruning(model=mock_model, max_relations_for_llm=2)
    relations = [
        ("rel high 1", "target1", "OUTGOING", 10.0),
        ("rel high 2", "target2", "OUTGOING", 9.0),
        ("rel low", "target3", "OUTGOING", 1.0),
    ]

    scored, _ = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

    assert len(scored) == 3
    scored_map = {(r[0], r[1]): r[4] for r in scored}
    assert scored_map[("rel high 1", "target1")] == 8.0
    assert scored_map[("rel high 2", "target2")] == 7.0
    assert scored_map[("rel low", "target3")] == 0.0
