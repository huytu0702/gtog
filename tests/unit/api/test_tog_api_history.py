import pytest
import inspect
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
    ConversationRole,
)


def test_tog_search_signature_has_required_text_units_and_conversation_history():
    """Verify tog_search() accepts text_units and conversation_history parameters."""
    from graphrag.api.query import tog_search

    sig = inspect.signature(tog_search)
    assert "text_units" in sig.parameters
    assert "conversation_history" in sig.parameters

    text_units_param = sig.parameters["text_units"]
    assert text_units_param.default is inspect._empty

    param = sig.parameters["conversation_history"]
    assert param.default is None


def test_tog_search_streaming_signature_has_required_text_units_and_conversation_history():
    """Verify tog_search_streaming() accepts text_units and conversation_history parameters."""
    from graphrag.api.query import tog_search_streaming

    sig = inspect.signature(tog_search_streaming)
    assert "text_units" in sig.parameters
    assert "conversation_history" in sig.parameters

    text_units_param = sig.parameters["text_units"]
    assert text_units_param.default is inspect._empty

    param = sig.parameters["conversation_history"]
    assert param.default is None


@pytest.mark.asyncio
async def test_conversation_history_can_be_instantiated():
    """Verify ConversationHistory can be created and used."""
    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "Previous question")
    history.add_turn(ConversationRole.ASSISTANT, "Previous answer")

    # Verify build_context works (include_user_turns_only=False to get all turns)
    context, _ = history.build_context(include_user_turns_only=False)
    assert "Previous question" in context
    assert "Previous answer" in context


@pytest.mark.asyncio
async def test_conversation_history_user_turns():
    """Verify get_user_turns extracts user questions."""
    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "First question")
    history.add_turn(ConversationRole.ASSISTANT, "First answer")
    history.add_turn(ConversationRole.USER, "Second question")

    # Use max_user_turns=5 to get all user turns
    user_turns = history.get_user_turns(max_user_turns=5)
    assert len(user_turns) == 2
    assert "First question" in user_turns
    assert "Second question" in user_turns
