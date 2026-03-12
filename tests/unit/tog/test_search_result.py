from typing import cast

import pytest
from unittest.mock import AsyncMock, MagicMock

from graphrag.data_model.text_unit import TextUnit
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.tog_search.pruning import PruningMetrics
from graphrag.query.structured_search.tog_search.reasoning import ReasoningMetrics
from graphrag.query.structured_search.tog_search.search import ToGSearch


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
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.num_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [("rel_desc", "target1", "OUTGOING", 1.0, 8.0)],
        PruningMetrics(llm_calls=1, prompt_tokens=100, output_tokens=20)
    ))
    mock_pruning.score_entities = AsyncMock(return_value=(
        [7.0],
        PruningMetrics(llm_calls=1, prompt_tokens=60, output_tokens=15)
    ))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        False, None, ReasoningMetrics(llm_calls=1, prompt_tokens=50, output_tokens=10)
    ))
    mock_reasoning.generate_answer = AsyncMock(return_value=(
        "The answer is 42.",
        ["path1", "path2"],
        ReasoningMetrics(llm_calls=1, prompt_tokens=200, output_tokens=100)
    ))
    mock_reasoning._format_paths = MagicMock(return_value="=== CHUNKS ===\nchunk text")

    mock_entity = MagicMock()
    mock_entity.id = "e1"
    mock_entity.title = "Entity1"
    mock_entity.description = "Description1"
    mock_entity.text_unit_ids = ["t1"]

    mock_rel = MagicMock()
    mock_rel.id = "r1"
    mock_rel.source = "e1"
    mock_rel.target = "e2"
    mock_rel.description = "relates to"
    mock_rel.weight = 1.0
    mock_rel.text_unit_ids = ["t1"]

    search = ToGSearch(
        model=mock_model,
        entities=[mock_entity],
        relationships=[mock_rel],
        text_units=[TextUnit(id="t1", short_id="t1", text="chunk text")],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        width=2,
        depth=1,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_full_entity_info = MagicMock(return_value=("e1", "Entity1", "Description1"))
    search.explorer.get_relations = MagicMock(return_value=[
        ("relates to", "e2", "OUTGOING", 1.0)
    ])
    search.explorer.get_text_units_for_nodes = MagicMock(return_value=[
        TextUnit(id="t1", short_id="t1", text="chunk text")
    ])

    result = await search.search(query="test query")

    context_data = cast(dict[str, list[str]], result.context_data)

    assert isinstance(result, SearchResult)
    assert result.response == "The answer is 42."
    assert result.context_text == "=== CHUNKS ===\nchunk text"
    assert context_data["exploration_paths"] == ["path1", "path2"]
    assert context_data["chunks"] == ["t1"]
    assert result.llm_calls == 4
    assert result.prompt_tokens == 410
    assert result.output_tokens == 145
    assert result.llm_calls_categories == {"exploration": 2, "reasoning": 2}
    assert result.prompt_tokens_categories == {"exploration": 160, "reasoning": 250}
    assert result.output_tokens_categories == {"exploration": 35, "reasoning": 110}
    assert result.completion_time >= 0
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
        text_units=[],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=[])

    chunks = []
    async for chunk in search.stream_search(query="test"):
        chunks.append(chunk)

    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_tog_search_uses_entity_scores_for_branch_selection():
    """Entity scoring should influence which branch survives beam pruning."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.num_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [
            ("rel_a", "e2", "OUTGOING", 1.0, 8.0),
            ("rel_b", "e3", "OUTGOING", 1.0, 8.0),
        ],
        PruningMetrics(llm_calls=1, prompt_tokens=40, output_tokens=10),
    ))
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
    mock_reasoning._format_paths = MagicMock(return_value="context")

    root = MagicMock()
    root.id = "e1"
    root.title = "Root"
    root.description = "Root description"
    root.text_unit_ids = []

    rel1 = MagicMock()
    rel1.id = "r1"
    rel1.source = "e1"
    rel1.target = "e2"
    rel1.description = "rel_a"
    rel1.weight = 1.0
    rel1.text_unit_ids = []

    rel2 = MagicMock()
    rel2.id = "r2"
    rel2.source = "e1"
    rel2.target = "e3"
    rel2.description = "rel_b"
    rel2.weight = 1.0
    rel2.text_unit_ids = []

    search = ToGSearch(
        model=mock_model,
        entities=[root],
        relationships=[rel1, rel2],
        text_units=[],
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
    ])
    search.explorer.get_text_units_for_nodes = MagicMock(return_value=[])

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
async def test_tog_search_provides_chunk_context_to_early_and_final_reasoning():
    """ToG search should provide chunk context to both reasoning stages."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.num_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [("rel_desc", "target1", "OUTGOING", 1.0, 8.0)],
        PruningMetrics()
    ))
    mock_pruning.score_entities = AsyncMock(return_value=([7.0], PruningMetrics()))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(False, None, ReasoningMetrics()))
    mock_reasoning.generate_answer = AsyncMock(return_value=("answer", ["path"], ReasoningMetrics()))
    mock_reasoning._format_paths = MagicMock(return_value="=== CHUNKS ===\nchunk text")

    entity = MagicMock()
    entity.id = "e1"
    entity.title = "Entity1"
    entity.description = "Description1"
    entity.text_unit_ids = ["t1"]

    rel = MagicMock()
    rel.id = "r1"
    rel.source = "e1"
    rel.target = "e2"
    rel.description = "rel_desc"
    rel.weight = 1.0
    rel.text_unit_ids = ["t1"]

    search = ToGSearch(
        model=mock_model,
        entities=[entity],
        relationships=[rel],
        text_units=[TextUnit(id="t1", short_id="t1", text="chunk text")],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        depth=1,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_full_entity_info = MagicMock(side_effect=[
        ("e1", "Entity1", "Description1"),
        ("e2", "Entity2", "Description2"),
    ])
    search.explorer.get_relations = MagicMock(return_value=[("rel_desc", "e2", "OUTGOING", 1.0)])
    search.explorer.get_full_relation_info = MagicMock(return_value=("r1", "rel_desc", "e1", "e2", 1.0))
    search.explorer.get_text_units_for_nodes = MagicMock(return_value=[
        TextUnit(id="t1", short_id="t1", text="chunk text")
    ])

    await search.search(query="test query")

    assert mock_reasoning.check_early_termination.await_args.kwargs["text_units"][0].text == "chunk text"
    assert mock_reasoning.generate_answer.await_args.kwargs["text_units"][0].text == "chunk text"


@pytest.mark.asyncio
async def test_tog_search_early_termination_keeps_chunk_ids():
    """Early termination should preserve chunk ids in SearchResult context."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_pruning = MagicMock()
    mock_pruning.score_relations = AsyncMock(return_value=(
        [("rel_desc", "e2", "OUTGOING", 1.0, 8.0)],
        PruningMetrics()
    ))
    mock_pruning.score_entities = AsyncMock(return_value=([7.0], PruningMetrics()))

    mock_reasoning = MagicMock()
    mock_reasoning.check_early_termination = AsyncMock(return_value=(
        True,
        "early answer",
        ReasoningMetrics(llm_calls=1, prompt_tokens=10, output_tokens=5),
    ))
    mock_reasoning.get_reasoning_paths = MagicMock(return_value=["path1"])
    mock_reasoning._format_paths = MagicMock(return_value="=== CHUNKS ===\nchunk text")
    mock_reasoning.generate_answer = AsyncMock()

    entity = MagicMock()
    entity.id = "e1"
    entity.title = "Entity1"
    entity.description = "Description1"
    entity.text_unit_ids = ["t1"]

    rel = MagicMock()
    rel.id = "r1"
    rel.source = "e1"
    rel.target = "e2"
    rel.description = "rel_desc"
    rel.weight = 1.0
    rel.text_unit_ids = ["t1"]

    search = ToGSearch(
        model=mock_model,
        entities=[entity],
        relationships=[rel],
        text_units=[TextUnit(id="t1", short_id="t1", text="chunk text")],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
        depth=1,
    )

    search.explorer.find_starting_entities = MagicMock(return_value=["e1"])
    search.explorer.get_full_entity_info = MagicMock(side_effect=[
        ("e1", "Entity1", "Description1"),
        ("e2", "Entity2", "Description2"),
    ])
    search.explorer.get_relations = MagicMock(return_value=[("rel_desc", "e2", "OUTGOING", 1.0)])
    search.explorer.get_full_relation_info = MagicMock(return_value=("r1", "rel_desc", "e1", "e2", 1.0))
    search.explorer.get_text_units_for_nodes = MagicMock(return_value=[
        TextUnit(id="t1", short_id="t1", text="chunk text")
    ])

    result = await search.search(query="test query")
    context_data = cast(dict[str, list[str]], result.context_data)

    assert result.response == "early answer"
    assert result.context_text == "=== CHUNKS ===\nchunk text"
    assert context_data["exploration_paths"] == ["path1"]
    assert context_data["chunks"] == ["t1"]
    mock_reasoning.generate_answer.assert_not_awaited()
