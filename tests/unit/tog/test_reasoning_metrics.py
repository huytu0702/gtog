import pytest
from unittest.mock import MagicMock
from graphrag.query.structured_search.tog_search.reasoning import (
    ToGReasoning,
    ReasoningMetrics,
)
from graphrag.query.structured_search.tog_search.state import ExplorationNode


def test_format_paths_merged_relationships():
    """_format_paths should merge relationship sections into one."""
    mock_model = MagicMock()
    reasoning = ToGReasoning(model=mock_model)

    # Create a simple path: Parent -> Child
    parent = ExplorationNode(
        entity_id="e1",
        entity_name="Parent",
        entity_description="Parent entity",
        entity_full_description="Full parent description",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    child = ExplorationNode(
        entity_id="e2",
        entity_name="Child",
        entity_description="Child entity",
        entity_full_description="Full child description",
        depth=1,
        score=0.8,
        parent=parent,
        relation_from_parent="relates_to",
        relation_full_description="Full relationship description",
    )

    result = reasoning._format_paths([child])

    # Should have ENTITIES section
    assert "=== ENTITIES ===" in result
    assert "[Parent]" in result
    assert "[Child]" in result

    # Should have RELATIONSHIPS section
    assert "=== RELATIONSHIPS ===" in result
    assert "Parent --[relates_to]--> Child" in result

    # Should NOT have separate RELATIONSHIP DESCRIPTIONS section
    assert "=== RELATIONSHIP DESCRIPTIONS ===" not in result

    # Relationship description should be inline
    assert "Description: Full relationship description" in result


def test_format_paths_no_duplicate_sections():
    """_format_paths should not duplicate relationship information."""
    mock_model = MagicMock()
    reasoning = ToGReasoning(model=mock_model)

    parent = ExplorationNode(
        entity_id="e1",
        entity_name="Parent",
        entity_description="Parent desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    child = ExplorationNode(
        entity_id="e2",
        entity_name="Child",
        entity_description="Child desc",
        depth=1,
        score=0.8,
        parent=parent,
        relation_from_parent="test_relation",
        relation_full_description="Test relation full desc",
    )

    result = reasoning._format_paths([child])

    # Count occurrences of section headers
    entities_count = result.count("=== ENTITIES ===")
    relationships_count = result.count("=== RELATIONSHIPS ===")

    assert entities_count == 1, "Should have exactly one ENTITIES section"
    assert relationships_count == 1, "Should have exactly one RELATIONSHIPS section"


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
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(["The answer is 42."])
    )

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
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(["NO: Need more exploration"])
    )

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
