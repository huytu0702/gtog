from graphrag.query.structured_search.tog_search.state import ExplorationNode


def _root() -> ExplorationNode:
    return ExplorationNode(
        entity_id="root",
        entity_name="Root",
        entity_description="root desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )


def _child(
    parent: ExplorationNode,
    name: str,
    relation: str,
    direction: str,
) -> ExplorationNode:
    return ExplorationNode(
        entity_id=name.lower(),
        entity_name=name,
        entity_description=f"{name} desc",
        depth=parent.depth + 1,
        score=1.0,
        parent=parent,
        relation_from_parent=relation,
        relation_direction_from_parent=direction,
        relation_source_id=parent.entity_id,
        relation_target_id=name.lower(),
    )


def test_root_node_has_no_relation_history():
    root = _root()

    assert root.get_relation_history() == []
    assert root.get_relation_history_text() == "None"


def test_outgoing_child_history_includes_direction():
    root = _root()
    child = _child(root, "Child", "created", "outgoing")

    assert child.get_relation_history() == [("Root", "created", "Child", "outgoing")]
    assert child.get_relation_history_text() == "Root --[created]--> Child"


def test_incoming_child_history_renders_direction():
    root = _root()
    child = _child(root, "Child", "created_by", "incoming")

    assert child.get_relation_history() == [("Root", "created_by", "Child", "incoming")]
    assert child.get_relation_history_text() == "Root <--[created_by]-- Child"


def test_multihop_history_preserves_root_to_leaf_order():
    root = _root()
    first = _child(root, "First", "rel1", "outgoing")
    second = _child(first, "Second", "rel2", "incoming")

    assert second.get_relation_history() == [
        ("Root", "rel1", "First", "outgoing"),
        ("First", "rel2", "Second", "incoming"),
    ]
    assert second.get_relation_history_text() == (
        "Root --[rel1]--> First\nFirst <--[rel2]-- Second"
    )
