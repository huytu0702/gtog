import pytest
from unittest.mock import MagicMock
from graphrag.query.structured_search.tog_search.reasoning import ToGReasoning
from graphrag.query.structured_search.tog_search.state import ExplorationNode


def _make_node(name: str) -> ExplorationNode:
    return ExplorationNode(
        entity_id="e1",
        entity_name=name,
        entity_description="desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )


@pytest.mark.asyncio
async def test_generate_answer_includes_history_context():
    """History context appears in the prompt sent to the LLM."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "answer"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]
    history_ctx = "-----Conversation History-----\nuser|Who is Entity A?"

    await reasoning.generate_answer(
        "Tell me more", nodes, conversation_history_context=history_ctx
    )

    assert len(captured_prompt) == 1
    assert history_ctx in captured_prompt[0]


@pytest.mark.asyncio
async def test_generate_answer_no_history_unchanged():
    """When no history, prompt is unaffected."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "answer"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]

    await reasoning.generate_answer("Tell me more", nodes)

    assert len(captured_prompt) == 1
    assert "Conversation History" not in captured_prompt[0]


@pytest.mark.asyncio
async def test_check_early_termination_includes_history():
    """History context appears in early termination prompt."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "NO: need more"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]
    history_ctx = "-----Conversation History-----\nuser|Previous question"

    should_terminate, _answer, _ = await reasoning.check_early_termination(
        "Follow-up?", nodes, conversation_history_context=history_ctx
    )

    assert history_ctx in captured_prompt[0]
    assert should_terminate is False


def test_reasoning_path_formats_incoming_and_outgoing_edges():
    root = _make_node("Root")
    outgoing = ExplorationNode(
        entity_id="child",
        entity_name="Child",
        entity_description="child desc",
        depth=1,
        score=1.0,
        parent=root,
        relation_from_parent="rel_out",
        relation_direction_from_parent="outgoing",
    )
    incoming = ExplorationNode(
        entity_id="leaf",
        entity_name="Leaf",
        entity_description="leaf desc",
        depth=2,
        score=1.0,
        parent=outgoing,
        relation_from_parent="rel_in",
        relation_direction_from_parent="incoming",
    )

    reasoning = ToGReasoning(model=MagicMock())

    assert reasoning.get_reasoning_paths([incoming]) == [
        "Root --[rel_out]--> Child | Child <--[rel_in]-- Leaf"
    ]


def test_reasoning_path_keeps_legacy_fallback_when_direction_missing():
    root = _make_node("Root")
    child = ExplorationNode(
        entity_id="child",
        entity_name="Child",
        entity_description="child desc",
        depth=1,
        score=1.0,
        parent=root,
        relation_from_parent="rel",
    )

    reasoning = ToGReasoning(model=MagicMock())

    assert reasoning.get_reasoning_paths([child]) == ["Root --[rel]--> Child"]


def test_format_paths_renders_incoming_relationships():
    root = _make_node("Root")
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

    reasoning = ToGReasoning(model=MagicMock())

    assert "Root <--[rel]-- Child" in reasoning.format_paths([child])
