"""Unit tests for server-side conversation service."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.models.schemas import ConversationTurn
from backend.app.services.conversation_service import ConversationService

conversation_service_module = importlib.import_module("backend.app.services.conversation_service")


def test_get_prompt_context_returns_summary_and_recent_turns():
    service = ConversationService()
    service.control_plane = MagicMock()
    service.repo = MagicMock()
    service.repo.get_session.return_value = {
        "sessionId": "s1",
        "collectionId": "c1",
        "summary": "topic summary",
        "turnCount": 4,
        "userTurnCount": 2,
    }
    service.repo.list_turns_desc.return_value = [
        {"role": "assistant", "content": "A2", "createdAt": "2026-01-01T00:00:04"},
        {"role": "user", "content": "Q2", "createdAt": "2026-01-01T00:00:03"},
        {"role": "assistant", "content": "A1", "createdAt": "2026-01-01T00:00:02"},
        {"role": "user", "content": "Q1", "createdAt": "2026-01-01T00:00:01"},
    ]

    summary, turns = service.get_prompt_context("c1", "s1")

    assert summary == "topic summary"
    assert turns == [
        ConversationTurn(role="user", content="Q1"),
        ConversationTurn(role="assistant", content="A1"),
        ConversationTurn(role="user", content="Q2"),
        ConversationTurn(role="assistant", content="A2"),
    ]


@pytest.mark.asyncio
async def test_append_exchange_triggers_auto_summary_at_threshold():
    service = ConversationService()
    service.control_plane = MagicMock()
    service.repo = MagicMock()
    service.repo.get_session.return_value = {
        "sessionId": "s1",
        "collectionId": "c1",
        "summary": "old summary",
        "turnCount": 10,
        "userTurnCount": 7,
    }
    service.repo.update_session_after_turns.return_value = {
        "sessionId": "s1",
        "collectionId": "c1",
        "summary": "old summary",
        "turnCount": 12,
        "userTurnCount": 8,
    }
    service.repo.list_turns_desc.return_value = [
        {"role": "assistant", "content": "A2", "createdAt": "2026-01-01T00:00:04"},
        {"role": "user", "content": "Q2", "createdAt": "2026-01-01T00:00:03"},
        {"role": "assistant", "content": "A1", "createdAt": "2026-01-01T00:00:02"},
        {"role": "user", "content": "Q1", "createdAt": "2026-01-01T00:00:01"},
    ]

    with patch.object(
        conversation_service_module.summarization_service,
        "summarize",
        new=AsyncMock(return_value="new summary"),
    ) as mock_summarize:
        await service.append_exchange(
            collection_id="c1",
            session_id="s1",
            user_query="Who directed it?",
            assistant_response="Christopher Nolan.",
            rewritten_query="Who directed Inception?",
            method_used="local",
        )

    service.repo.append_turns.assert_called_once()
    mock_summarize.assert_awaited_once()
    service.repo.update_summary.assert_called_once_with(
        session_id="s1",
        summary="new summary",
    )
