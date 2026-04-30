from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
from graphrag.config.models.tog_search_config import ToGSearchConfig
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
    ConversationRole,
)


def _make_engine(captured_queries: list) -> ToGSearch:
    """Build a minimal ToGSearch that records what query is used for entity linking."""

    async def fake_find_semantic(query, top_k):
        captured_queries.append(query)
        return []  # No entities — triggers early exit

    mock_explorer = MagicMock()
    mock_explorer.find_starting_entities_semantic = fake_find_semantic

    mock_pruning = MagicMock()
    mock_reasoning = MagicMock()

    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = mock_explorer
    engine.pruning_strategy = mock_pruning
    engine.reasoning_module = mock_reasoning
    engine.embedding_model = MagicMock()  # triggers semantic path
    engine.width = 2
    engine.depth = 2
    engine.num_retain_entity = 3
    engine.callbacks = []
    engine._debug = False  # noqa: SLF001
    engine.model = MagicMock()
    engine.tokenizer = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_search_enriches_entity_query_with_history():
    """History user turns are appended to entity-linking query."""
    captured = []
    engine = _make_engine(captured)

    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "Tell me about Inception")
    history.add_turn(ConversationRole.ASSISTANT, "Inception is a film...")

    await engine.search("Who directed it?", conversation_history=history)

    assert len(captured) == 1
    assert "Who directed it?" in captured[0]
    assert "Tell me about Inception" in captured[0]


@pytest.mark.asyncio
async def test_search_no_history_uses_original_query():
    """Without history the query is used as-is."""
    captured = []
    engine = _make_engine(captured)

    await engine.search("Who directed it?", conversation_history=None)

    assert captured[0] == "Who directed it?"


class _CapturingPruning:
    def __init__(self):
        self.relation_kwargs = []
        self.entity_candidates = []

    async def score_relations(self, query, entity_name, relations, **kwargs):
        self.relation_kwargs.append(kwargs)
        return [(*relation, 9.0) for relation in relations], MagicMock()

    async def score_entities(self, query, current_path, entities, **kwargs):
        self.entity_candidates.append(entities)
        return [8.0] * len(entities), MagicMock()


class _LegacyPruning:
    async def score_relations(
        self, query, entity_name, relations, query_embedding=None
    ):
        rel_desc, target_id, direction, weight = relations[0]
        return [(rel_desc, target_id, direction, weight, 9.0)], MagicMock()

    async def score_entities(self, query, current_path, entities, query_embedding=None):
        return [8.0], MagicMock()


@pytest.mark.asyncio
async def test_process_node_preserves_relation_metadata_and_passes_history():
    from graphrag.query.structured_search.tog_search.state import ExplorationNode

    parent = ExplorationNode(
        entity_id="root",
        entity_name="Root",
        entity_description="root desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    node = ExplorationNode(
        entity_id="parent",
        entity_name="Parent",
        entity_description="parent desc",
        depth=1,
        score=1.0,
        parent=parent,
        relation_from_parent="root_rel",
        relation_direction_from_parent="outgoing",
    )

    explorer = MagicMock()
    explorer.get_relations.return_value = [("rel", "child-title", "incoming", 1.0)]
    explorer.get_full_entity_info.side_effect = {
        "parent": ("parent-id", "Parent", "parent desc"),
        "child-title": ("child-id", "Child", "child desc"),
    }.get
    explorer.get_full_relation_info.return_value = ("rel_id", "full rel desc")

    pruning = _CapturingPruning()
    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = cast(Any, pruning)  # noqa: TC006
    engine.width = 1
    engine.num_retain_entity = 5

    new_nodes, metrics = await engine._process_node("query", node)  # noqa: SLF001

    assert len(metrics) == 2
    assert len(new_nodes) == 1
    child = new_nodes[0]
    assert child.parent is node
    assert child.relation_direction_from_parent == "incoming"
    assert child.entity_id == "child-title"
    assert child.relation_source_id == "child-id"
    assert child.relation_target_id == "parent-id"
    assert pruning.entity_candidates == [[("child-id", "Child", "child desc")]]
    assert pruning.relation_kwargs[0]["current_path"] == "Root --[root_rel]--> Parent"
    assert (
        pruning.relation_kwargs[0]["relation_history"] == "Root --[root_rel]--> Parent"
    )


@pytest.mark.asyncio
async def test_process_node_supports_legacy_relation_scorer_signature():
    from graphrag.query.structured_search.tog_search.state import ExplorationNode

    node = ExplorationNode(
        entity_id="parent",
        entity_name="Parent",
        entity_description="parent desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )

    explorer = MagicMock()
    explorer.get_relations.return_value = [("rel", "child", "incoming", 1.0)]
    explorer.get_full_entity_info.return_value = ("child", "Child", "child desc")
    explorer.get_full_relation_info.return_value = ("rel_id", "full rel desc")

    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = cast(Any, _LegacyPruning())  # noqa: TC006
    engine.width = 1
    engine.num_retain_entity = 5

    new_nodes, _metrics = await engine._process_node("query", node)  # noqa: SLF001

    assert len(new_nodes) == 1
    assert new_nodes[0].relation_direction_from_parent == "incoming"


@pytest.mark.asyncio
async def test_process_node_keeps_all_candidates_when_total_is_large_but_each_relation_group_is_small():
    from graphrag.query.structured_search.tog_search.state import ExplorationNode

    node = ExplorationNode(
        entity_id="parent",
        entity_name="Parent",
        entity_description="parent desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )

    relation_a = [("rel_a", f"a{i}", "outgoing", 1.0) for i in range(8)]
    relation_b = [("rel_b", f"b{i}", "outgoing", 1.0) for i in range(8)]
    relation_c = [("rel_c", f"c{i}", "incoming", 1.0) for i in range(8)]

    explorer = MagicMock()
    explorer.get_relations.return_value = relation_a + relation_b + relation_c

    def _get_entity_info(entity_id: str):
        if entity_id == "parent":
            return ("parent-id", "Parent", "parent desc")
        return (entity_id, entity_id.upper(), f"desc {entity_id}")

    explorer.get_full_entity_info.side_effect = _get_entity_info
    explorer.get_full_relation_info.side_effect = (
        lambda source_id, target_id, rel_desc: ("rel_id", rel_desc)
    )

    pruning = _CapturingPruning()
    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = cast(Any, pruning)  # noqa: TC006
    engine.width = 3
    engine.num_retain_entity = 5

    with pytest.MonkeyPatch.context() as mp:
        sample_calls = []
        mp.setattr(
            "graphrag.query.structured_search.tog_search.search.random.sample",
            lambda population, k: sample_calls.append((population, k))
            or population[:k],
        )
        new_nodes, _metrics = await engine._process_node("query", node)  # noqa: SLF001

    assert sample_calls == []
    assert len(new_nodes) == 24
    assert len(pruning.entity_candidates) == 1
    assert len(pruning.entity_candidates[0]) == 24
    assert [entity_id for entity_id, _, _ in pruning.entity_candidates[0]] == [
        *[f"a{i}" for i in range(8)],
        *[f"b{i}" for i in range(8)],
        *[f"c{i}" for i in range(8)],
    ]


class _EarlyTerminationReasoning:
    async def check_early_termination(self, *args, **kwargs):
        return True, "early answer", MagicMock()

    def get_reasoning_paths(self, nodes):
        return [node.entity_name for node in nodes]

    def format_paths(self, nodes, text_units=None):
        return "\n".join(node.entity_name for node in nodes)

    async def generate_answer(self, *args, **kwargs):
        return "final answer", ["Root --[rel]--> Child"], MagicMock()


class _OneHopPruning:
    async def score_relations(self, query, entity_name, relations, **kwargs):
        rel_desc, target_id, direction, weight = relations[0]
        return [(rel_desc, target_id, direction, weight, 9.0)], MagicMock()

    async def score_entities(self, query, current_path, entities, **kwargs):
        return [8.0], MagicMock()


def _make_early_termination_engine(
    force_max_depth: bool,
    debug: bool = False,
) -> ToGSearch:
    explorer = MagicMock()
    explorer.find_starting_entities.return_value = ["root"]
    explorer.get_full_entity_info.side_effect = {
        "root": ("root-id", "Root", "root desc"),
        "child": ("child-id", "Child", "child desc"),
    }.get
    explorer.get_text_units_for_nodes.return_value = []
    explorer.get_relations.side_effect = [[("rel", "child", "outgoing", 1.0)], []]
    explorer.get_full_relation_info.return_value = ("rel-id", "full rel desc")

    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = cast("Any", _OneHopPruning())
    engine.reasoning_module = _EarlyTerminationReasoning()
    engine.embedding_model = None
    engine.width = 1
    engine.depth = 1
    engine.num_retain_entity = 1
    engine.callbacks = []
    engine._debug = debug  # noqa: SLF001
    engine._debug_force_max_depth = force_max_depth  # noqa: SLF001
    engine.model = MagicMock()
    engine.tokenizer = MagicMock()
    return engine


def test_tog_search_config_disables_force_max_depth_by_default():
    assert ToGSearchConfig().debug_force_max_depth is False


def test_tog_search_config_accepts_force_max_depth():
    assert ToGSearchConfig(debug_force_max_depth=True).debug_force_max_depth is True


@pytest.mark.asyncio
async def test_search_early_terminates_by_default():
    engine = _make_early_termination_engine(force_max_depth=False)

    result = await engine.search("query")

    assert result.response == "early answer"
    engine.explorer.get_relations.assert_not_called()


@pytest.mark.asyncio
async def test_search_force_max_depth_requires_debug_mode():
    engine = _make_early_termination_engine(force_max_depth=True, debug=False)

    result = await engine.search("query")

    assert result.response == "early answer"
    engine.explorer.get_relations.assert_not_called()


@pytest.mark.asyncio
async def test_search_force_max_depth_bypasses_early_termination_in_debug_mode():
    engine = _make_early_termination_engine(force_max_depth=True, debug=True)

    result = await engine.search("query")

    assert result.response == "final answer"
    assert engine.explorer.get_relations.call_args_list == [call("root")]
    assert result.context_data == {"exploration_paths": ["Root --[rel]--> Child"]}


def test_node_to_path_string_formats_incoming_edges():
    from graphrag.query.structured_search.tog_search.state import ExplorationNode

    root = ExplorationNode(
        entity_id="root",
        entity_name="Root",
        entity_description="root desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    child = ExplorationNode(
        entity_id="child",
        entity_name="Child",
        entity_description="child desc",
        depth=1,
        score=1.0,
        parent=root,
        relation_from_parent="rel",
        relation_direction_from_parent="incoming",
    )
    engine = ToGSearch.__new__(ToGSearch)

    assert engine._node_to_path_string(child) == "Root <--[rel]-- Child"  # noqa: SLF001
