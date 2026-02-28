"""Tests for SummarizationService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.summarization_service import SummarizationService
from backend.app.models.schemas import ConversationTurn


class TestSummarizationService:
    """Test SummarizationService."""

    @pytest.fixture
    def service(self):
        return SummarizationService()

    @pytest.mark.asyncio
    async def test_summarize_returns_string(self, service):
        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "User asked about Inception (2010 film)."
            turns = [
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ]
            result = await service.summarize(turns)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_includes_existing_summary_in_prompt(self, service):
        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Updated summary."
            turns = [
                ConversationTurn(role="user", content="Who directed it?",
                                 rewritten_query="Who directed Inception?", method_used="local"),
                ConversationTurn(role="assistant", content="Christopher Nolan."),
            ]
            await service.summarize(turns, existing_summary="User was asking about Inception.")
            call_prompt = mock_llm.call_args[0][0]
            assert "User was asking about Inception." in call_prompt

    @pytest.mark.asyncio
    async def test_summarize_falls_back_on_llm_error(self, service):
        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")
            turns = [
                ConversationTurn(role="user", content="Tell me about Inception"),
            ]
            result = await service.summarize(turns)
            # Falls back to a basic concatenation rather than crashing
            assert isinstance(result, str)

    def test_get_trimmed_history_keeps_recent_turns(self, service):
        from backend.app.models.schemas import ConversationTurn
        turns = []
        for i in range(5):
            turns.append(ConversationTurn(role="user", content=f"Q{i}",
                                          rewritten_query=f"Q{i}", method_used="local"))
            turns.append(ConversationTurn(role="assistant", content=f"A{i}"))

        trimmed = service.get_trimmed_history(turns, keep_turns=3)
        # Should keep last 3 user turns = 6 messages
        user_turns = [t for t in trimmed if t.role == "user"]
        assert len(user_turns) == 3
        assert user_turns[-1].content == "Q4"
