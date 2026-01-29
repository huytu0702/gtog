import pytest
from unittest.mock import MagicMock
from graphrag.query.structured_search.tog_search.reasoning import ToGReasoning, ReasoningMetrics
from graphrag.query.structured_search.tog_search.state import ExplorationNode

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
async def test_reasoning_returns_metrics():
    """ToGReasoning.generate_answer should return metrics."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(return_value=AsyncIteratorMock(["The answer is 42."]))

    reasoning = ToGReasoning(model=mock_model)

    node = ExplorationNode(
        entity_id="e1",
        entity_name="Entity1",
        entity_description="Description",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )

    answer, paths, metrics = await reasoning.generate_answer(
        query="What is the answer?",
        exploration_paths=[node],
    )

    assert isinstance(metrics, ReasoningMetrics)
    assert metrics.llm_calls == 1
    assert answer == "The answer is 42."

@pytest.mark.asyncio
async def test_check_early_termination_returns_metrics():
    """ToGReasoning.check_early_termination should return metrics."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(return_value=AsyncIteratorMock(["NO: Need more exploration"]))

    reasoning = ToGReasoning(model=mock_model)

    node = ExplorationNode(
        entity_id="e1",
        entity_name="Entity1",
        entity_description="Description",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )

    should_terminate, answer, metrics = await reasoning.check_early_termination(
        query="What is the answer?",
        current_nodes=[node],
    )

    assert isinstance(metrics, ReasoningMetrics)
    assert metrics.llm_calls == 1
    assert should_terminate == False
    assert answer is None
