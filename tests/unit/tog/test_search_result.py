import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.tog_search.pruning import PruningMetrics
from graphrag.query.structured_search.tog_search.reasoning import ReasoningMetrics

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
async def test_tog_search_returns_search_result():
    """ToGSearch.search should return SearchResult with metrics."""
    # Create mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock pruning strategy
    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [("rel_desc", "target1", "OUTGOING", 1.0, 8.0)],
        PruningMetrics(llm_calls=1, prompt_tokens=100, output_tokens=20)
    ))

    # Mock reasoning module
    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        False, None, ReasoningMetrics(llm_calls=1, prompt_tokens=50, output_tokens=10)
    ))
    mock_reasoning.generate_answer = AsyncMock(return_value=(
        "The answer is 42.",
        ["path1", "path2"],
        ReasoningMetrics(llm_calls=1, prompt_tokens=200, output_tokens=100)
    ))

    # Create mock entities and relationships
    mock_entity = MagicMock()
    mock_entity.id = "e1"
    mock_entity.title = "Entity1"
    mock_entity.description = "Description1"

    mock_rel = MagicMock()
    mock_rel.source = "e1"
    mock_rel.target = "e2"
    mock_rel.description = "relates to"
    mock_rel.weight = 1.0

    search = ToGSearch(
        model=mock_model,
        entities=[mock_entity],
        relationships=[mock_rel],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        width=2,
        depth=1,
    )

    # Mock the explorer methods
    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_entity_info = MagicMock(return_value=("Entity1", "Description1"))
    search.explorer.get_relations = MagicMock(return_value=[
        ("relates to", "e2", "OUTGOING", 1.0)
    ])

    result = await search.search(query="test query")

    assert isinstance(result, SearchResult)
    assert result.response == "The answer is 42."
    assert result.llm_calls >= 1  # At least reasoning calls
    assert result.completion_time >= 0  # May be very small or zero in fast mock tests

@pytest.mark.asyncio
async def test_tog_stream_search_backward_compatible():
    """ToGSearch.stream_search should still yield strings."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=([], PruningMetrics()))
    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        False, None, ReasoningMetrics()
    ))
    mock_reasoning.generate_answer = AsyncMock(return_value=(
        "Answer", [], ReasoningMetrics()
    ))

    search = ToGSearch(
        model=mock_model,
        entities=[],
        relationships=[],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=[])

    chunks = []
    async for chunk in search.stream_search(query="test"):
        chunks.append(chunk)

    # Should have at least one string chunk (error message about no entities)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
