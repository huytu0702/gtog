from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.query.structured_search.tog_search.state import ExplorationNode


def _make_node(
    relation_from_parent: str | None,
    relation_direction_from_parent: str | None,
    depth: int = 1,
) -> ExplorationNode:
    parent = ExplorationNode(
        entity_id="root",
        entity_name="Root",
        entity_description="root desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    return ExplorationNode(
        entity_id="current",
        entity_name="Current",
        entity_description="current desc",
        depth=depth,
        score=1.0,
        parent=parent if relation_from_parent is not None else None,
        relation_from_parent=relation_from_parent,
        relation_direction_from_parent=relation_direction_from_parent,
    )


# --- Unit tests for _filter_backtrack_relations ---


def test_filter_removes_reverse_of_outgoing_arrival():
    """Arrived via (rel_A, outgoing) → remove (rel_A, incoming) candidates."""
    node = _make_node("rel_A", "outgoing")
    relations = [
        ("rel_A", "parent_entity", "incoming", 1.0),
        ("rel_B", "other_entity", "outgoing", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert ("rel_A", "parent_entity", "incoming", 1.0) not in result
    assert ("rel_B", "other_entity", "outgoing", 1.0) in result


def test_filter_removes_reverse_of_incoming_arrival():
    """Arrived via (rel_B, incoming) → remove (rel_B, outgoing) candidates."""
    node = _make_node("rel_B", "incoming")
    relations = [
        ("rel_B", "parent_entity", "outgoing", 1.0),
        ("rel_C", "other_entity", "incoming", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert ("rel_B", "parent_entity", "outgoing", 1.0) not in result
    assert ("rel_C", "other_entity", "incoming", 1.0) in result


def test_filter_keeps_same_relation_same_direction():
    """Same relation in same direction (parallel forward edge) is NOT removed."""
    node = _make_node("rel_A", "outgoing")
    relations = [
        ("rel_A", "another_entity", "outgoing", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert len(result) == 1


def test_filter_keeps_different_relation_opposite_direction():
    """Different relation in opposite direction is NOT removed."""
    node = _make_node("rel_A", "outgoing")
    relations = [
        ("rel_Z", "some_entity", "incoming", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert len(result) == 1


def test_filter_noop_on_root_node():
    """Root node (relation_from_parent=None) → no filtering."""
    node = ExplorationNode(
        entity_id="root",
        entity_name="Root",
        entity_description="root desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    relations = [
        ("rel_A", "entity_1", "outgoing", 1.0),
        ("rel_B", "entity_2", "incoming", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert result == relations


def test_filter_noop_when_direction_missing():
    """relation_direction_from_parent=None → no filtering (backward compat)."""
    node = _make_node("rel_A", None)
    relations = [
        ("rel_A", "entity_1", "incoming", 1.0),
    ]
    result = ToGSearch._filter_backtrack_relations(node, relations)  # noqa: SLF001
    assert result == relations


# --- Integration tests for _process_node ---


class _CapturingPruning:
    def __init__(self):
        self.scored_relations: list = []

    async def score_relations(self, query, entity_name, relations, **kwargs):
        self.scored_relations.extend(relations)
        return [(*r, 9.0) for r in relations], MagicMock()

    async def score_entities(self, query, current_path, entities, **kwargs):
        return [8.0] * len(entities), MagicMock()


@pytest.mark.asyncio
async def test_process_node_filters_backtrack_before_scoring():
    """Only non-backtrack relations reach the pruning strategy."""
    node = _make_node("rel_A", "outgoing")

    explorer = MagicMock()
    explorer.get_relations.return_value = [
        ("rel_A", "root", "incoming", 1.0),   # backtrack — must be removed
        ("rel_B", "other", "outgoing", 1.0),  # forward — must be kept
    ]
    explorer.get_full_entity_info.return_value = ("other-id", "Other", "other desc")
    explorer.get_full_relation_info.return_value = ("rel_id", "full rel desc")

    pruning = _CapturingPruning()
    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = cast(Any, pruning)  # noqa: TC006
    engine.width = 2
    engine.num_retain_entity = 5

    await engine._process_node("query", node)  # noqa: SLF001

    relation_names = [r[0] for r in pruning.scored_relations]
    assert "rel_B" in relation_names
    assert "rel_A" not in relation_names


@pytest.mark.asyncio
async def test_process_node_returns_empty_when_only_backtrack_relation_exists():
    """If the only available relation is the backtrack, return ([], [])."""
    node = _make_node("rel_A", "outgoing")

    explorer = MagicMock()
    explorer.get_relations.return_value = [
        ("rel_A", "root", "incoming", 1.0),
    ]

    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = explorer
    engine.pruning_strategy = MagicMock()
    engine.width = 2
    engine.num_retain_entity = 5

    new_nodes, metrics = await engine._process_node("query", node)  # noqa: SLF001

    assert new_nodes == []
    assert metrics == []
