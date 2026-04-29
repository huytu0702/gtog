from typing import Any, cast
from unittest.mock import MagicMock

import pytest
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
        rel_desc, target_id, direction, weight = relations[0]
        return [(rel_desc, target_id, direction, weight, 9.0)], MagicMock()

    async def score_entities(self, query, current_path, entities, **kwargs):
        self.entity_candidates.append(entities)
        return [8.0], MagicMock()


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
    engine.num_retain_entity = 5

    new_nodes, _metrics = await engine._process_node("query", node)  # noqa: SLF001

    assert len(new_nodes) == 1
    assert new_nodes[0].relation_direction_from_parent == "incoming"


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
