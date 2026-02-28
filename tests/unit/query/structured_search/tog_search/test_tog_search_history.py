import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
    engine._debug = False
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

    result = await engine.search("Who directed it?", conversation_history=history)

    assert len(captured) == 1
    assert "Who directed it?" in captured[0]
    assert "Tell me about Inception" in captured[0]


@pytest.mark.asyncio
async def test_search_no_history_uses_original_query():
    """Without history the query is used as-is."""
    captured = []
    engine = _make_engine(captured)

    result = await engine.search("Who directed it?", conversation_history=None)

    assert captured[0] == "Who directed it?"
