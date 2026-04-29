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
    mock_pruning.score_entities = AsyncMock(return_value=(
        [7.0],
        PruningMetrics(llm_calls=1, prompt_tokens=60, output_tokens=15)
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
    search.explorer.get_full_entity_info = MagicMock(return_value=("e1", "Entity1", "Description1"))
    search.explorer.get_relations = MagicMock(return_value=[
        ("relates to", "e2", "OUTGOING", 1.0)
    ])

    result = await search.search(query="test query")

    assert isinstance(result, SearchResult)
    assert result.response == "The answer is 42."
    assert result.llm_calls == 5
    assert result.prompt_tokens == 460
    assert result.output_tokens == 155
    assert result.llm_calls_categories == {"exploration": 2, "reasoning": 3}
    assert result.prompt_tokens_categories == {"exploration": 160, "reasoning": 300}
    assert result.output_tokens_categories == {"exploration": 35, "reasoning": 120}
    assert result.completion_time >= 0  # May be very small or zero in fast mock tests
    mock_pruning.score_entities.assert_awaited_once()

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


@pytest.mark.asyncio
async def test_tog_search_uses_entity_scores_for_branch_selection():
    """Entity scoring should influence which branch survives beam pruning."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [
            ("rel_a", "e2", "OUTGOING", 1.0, 8.0),
            ("rel_b", "e3", "OUTGOING", 1.0, 8.0),
        ],
        PruningMetrics(llm_calls=1, prompt_tokens=40, output_tokens=10),
    ))
    # Force e3 branch to be preferred even with equal relation scores.
    mock_pruning.score_entities = AsyncMock(return_value=(
        [1.0, 10.0],
        PruningMetrics(llm_calls=1, prompt_tokens=30, output_tokens=8),
    ))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        False, None, ReasoningMetrics()
    ))
    mock_reasoning.generate_answer = AsyncMock(return_value=(
        "answer", ["path"], ReasoningMetrics(llm_calls=1, prompt_tokens=20, output_tokens=5)
    ))

    root = MagicMock()
    root.id = "e1"
    root.title = "Root"
    root.description = "Root description"

    rel1 = MagicMock()
    rel1.source = "e1"
    rel1.target = "e2"
    rel1.description = "rel_a"
    rel1.weight = 1.0

    rel2 = MagicMock()
    rel2.source = "e1"
    rel2.target = "e3"
    rel2.description = "rel_b"
    rel2.weight = 1.0

    search = ToGSearch(
        model=mock_model,
        entities=[root],
        relationships=[rel1, rel2],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        width=1,  # keep only one branch
        depth=1,
        num_retain_entity=2,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_relations = MagicMock(return_value=[
        ("rel_a", "e2", "OUTGOING", 1.0),
        ("rel_b", "e3", "OUTGOING", 1.0),
    ])

    def _get_info(entity_id):
        if entity_id == "e1":
            return ("e1", "Root", "Root description")
        if entity_id == "e2":
            return ("e2", "Entity2", "Desc2")
        if entity_id == "e3":
            return ("e3", "Entity3", "Desc3")
        return None

    search.explorer.get_full_entity_info = MagicMock(side_effect=_get_info)
    search.explorer.get_full_relation_info = MagicMock(side_effect=lambda s, t, r: ("id", r, s, t, 1.0))

    await search.search(query="test query")

    all_paths = mock_reasoning.generate_answer.await_args.args[1]
    depth_one_nodes = [n for n in all_paths if n.depth == 1]
    assert len(depth_one_nodes) == 1
    assert depth_one_nodes[0].entity_id == "e3"


@pytest.mark.asyncio
async def test_num_retain_entity_caps_candidates_before_entity_scoring():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [
            ("rel_a", "e2", "OUTGOING", 1.0, 9.0),
            ("rel_b", "e3", "OUTGOING", 1.0, 8.0),
            ("rel_c", "e4", "OUTGOING", 1.0, 7.0),
        ],
        PruningMetrics(llm_calls=1, prompt_tokens=40, output_tokens=10),
    ))
    mock_pruning.score_entities = AsyncMock(return_value=(
        [7.0, 6.0],
        PruningMetrics(llm_calls=1, prompt_tokens=30, output_tokens=8),
    ))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        False, None, ReasoningMetrics()
    ))
    mock_reasoning.generate_answer = AsyncMock(return_value=(
        "answer", ["path"], ReasoningMetrics(llm_calls=1, prompt_tokens=20, output_tokens=5)
    ))

    root = MagicMock()
    root.id = "e1"
    root.title = "Root"
    root.description = "Root description"

    search = ToGSearch(
        model=mock_model,
        entities=[root],
        relationships=[],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        width=1,
        depth=1,
        num_retain_entity=2,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_relations = MagicMock(return_value=[
        ("rel_a", "e2", "OUTGOING", 1.0),
        ("rel_b", "e3", "OUTGOING", 1.0),
        ("rel_c", "e4", "OUTGOING", 1.0),
    ])

    def _get_info(entity_id):
        entity_info = {
            "e1": ("e1", "Root", "Root description"),
            "e2": ("e2", "Entity2", "Desc2"),
            "e3": ("e3", "Entity3", "Desc3"),
            "e4": ("e4", "Entity4", "Desc4"),
        }
        return entity_info.get(entity_id)

    search.explorer.get_full_entity_info = MagicMock(side_effect=_get_info)
    search.explorer.get_full_relation_info = MagicMock(side_effect=lambda s, t, r: ("id", r, s, t, 1.0))

    with patch(
        "graphrag.query.structured_search.tog_search.search.random.sample",
        side_effect=lambda population, k: population[:k],
    ) as sample_mock:
        await search.search(query="test query")

    sample_mock.assert_called_once()
    sampled_population, sampled_size = sample_mock.call_args.args
    assert [candidate[1] for candidate in sampled_population] == ["e2", "e3", "e4"]
    assert sampled_size == 2

    scored_entities = mock_pruning.score_entities.await_args.kwargs["entities"]
    assert scored_entities == [
        ("e2", "Entity2", "Desc2"),
        ("e3", "Entity3", "Desc3"),
    ]


@pytest.mark.asyncio
async def test_num_retain_entity_samples_after_filtering_missing_entities():
    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [
            ("rel_a", "missing", "OUTGOING", 1.0, 9.0),
            ("rel_b", "e2", "OUTGOING", 1.0, 8.0),
            ("rel_c", "e3", "OUTGOING", 1.0, 7.0),
            ("rel_d", "e4", "OUTGOING", 1.0, 6.0),
        ],
        PruningMetrics(),
    ))
    mock_pruning.score_entities = AsyncMock(return_value=(
        [7.0, 6.0],
        PruningMetrics(),
    ))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(False, None, ReasoningMetrics()))
    mock_reasoning.generate_answer = AsyncMock(return_value=("answer", ["path"], ReasoningMetrics()))

    root = MagicMock()
    root.id = "e1"
    root.title = "Root"
    root.description = "Root description"

    search = ToGSearch(
        model=MagicMock(),
        entities=[root],
        relationships=[],
        tokenizer=MagicMock(),
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        width=1,
        depth=1,
        num_retain_entity=2,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_relations = MagicMock(return_value=[
        ("rel_a", "missing", "OUTGOING", 1.0),
        ("rel_b", "e2", "OUTGOING", 1.0),
        ("rel_c", "e3", "OUTGOING", 1.0),
        ("rel_d", "e4", "OUTGOING", 1.0),
    ])

    def _get_info(entity_id):
        entity_info = {
            "e1": ("e1", "Root", "Root description"),
            "e2": ("e2", "Entity2", "Desc2"),
            "e3": ("e3", "Entity3", "Desc3"),
            "e4": ("e4", "Entity4", "Desc4"),
        }
        return entity_info.get(entity_id)

    search.explorer.get_full_entity_info = MagicMock(side_effect=_get_info)
    search.explorer.get_full_relation_info = MagicMock(side_effect=lambda s, t, r: ("id", r, s, t, 1.0))

    with patch(
        "graphrag.query.structured_search.tog_search.search.random.sample",
        side_effect=lambda population, k: population[:k],
    ) as sample_mock:
        await search.search(query="test query")

    sampled_population, sampled_size = sample_mock.call_args.args
    assert [candidate[1] for candidate in sampled_population] == ["e2", "e3", "e4"]
    assert sampled_size == 2
    assert mock_pruning.score_entities.await_args.kwargs["entities"] == [
        ("e2", "Entity2", "Desc2"),
        ("e3", "Entity3", "Desc3"),
    ]
