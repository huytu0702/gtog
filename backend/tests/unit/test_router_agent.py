"""Tests for Router Agent service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.services.router_agent import RouterAgent, RouteDecision
from backend.app.models.schemas import (
    ConversationTurn,
    AgentSearchRequest,
    AgentSearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)


class TestRouterAgent:
    """Test RouterAgent class."""

    @pytest.fixture
    def router_agent(self):
        """Create RouterAgent instance with mocked LLM."""
        return RouterAgent()

    @pytest.mark.asyncio
    async def test_route_returns_route_decision(self, router_agent):
        """route() should return a RouteDecision object."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"method": "local", "confidence": 0.85, "reasoning": "Query asks about specific entity"}'

        with patch.object(
            router_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route(
                "What is chamomile used for?", "herbs collection"
            )

            assert isinstance(result, RouteDecision)
            assert result.method == "local"
            assert result.confidence == 0.85
            assert "specific entity" in result.reasoning

    @pytest.mark.asyncio
    async def test_route_defaults_to_local_on_parse_error(self, router_agent):
        """route() should default to LOCAL if LLM response can't be parsed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "invalid json"

        with patch.object(
            router_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route("test query", "test collection")

            assert result.method == "local"
            assert (
                "default" in result.reasoning.lower()
                or "error" in result.reasoning.lower()
            )

    @pytest.mark.asyncio
    async def test_route_identifies_web_search_query(self, router_agent):
        """route() should identify queries needing web search."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"method": "web", "confidence": 0.92, "reasoning": "Query asks about current FDA regulations"}'

        with patch.object(
            router_agent, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route(
                "What are the latest FDA regulations?", "herbs collection"
            )

            assert result.method == "web"

    def test_format_history_block_empty(self, router_agent):
        """Returns empty string when no history and no summary."""
        result = router_agent._format_history_block([], None)
        assert result == ""

    def test_format_history_block_summary_only(self, router_agent):
        """Shows summary section when only summary provided."""
        result = router_agent._format_history_block([], "User asked about Inception.")
        assert "Past conversation summary" in result
        assert "User asked about Inception." in result

    def test_format_history_block_turns_only(self, router_agent):
        from backend.app.models.schemas import ConversationTurn
        turns = [
            ConversationTurn(role="user", content="Tell me about Inception",
                             rewritten_query="Tell me about Inception", method_used="local"),
            ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
        ]
        result = router_agent._format_history_block(turns, None)
        assert "[User]" in result
        assert "Tell me about Inception" in result
        assert "method: local" in result
        assert "[Assistant]" in result

    def test_format_history_block_summary_and_turns(self, router_agent):
        from backend.app.models.schemas import ConversationTurn
        turns = [
            ConversationTurn(role="user", content="Who starred in it?",
                             rewritten_query="Who starred in Inception?", method_used="local"),
            ConversationTurn(role="assistant", content="Leonardo DiCaprio starred..."),
        ]
        result = router_agent._format_history_block(turns, "User asked about Inception (2010).")
        assert "Past conversation summary" in result
        assert "User asked about Inception (2010)." in result
        assert "Recent conversation" in result
        assert "Who starred in it?" in result

    def test_format_history_block_limits_to_three_recent_turns(self, router_agent):
        from backend.app.models.schemas import ConversationTurn
        turns = []
        for i in range(5):
            turns.append(ConversationTurn(role="user", content=f"Question {i}",
                                          rewritten_query=f"Question {i}", method_used="local"))
            turns.append(ConversationTurn(role="assistant", content=f"Answer {i}"))
        result = router_agent._format_history_block(turns, None)
        assert "Question 1" not in result
        assert "Question 4" in result

    @pytest.mark.asyncio
    async def test_route_returns_rewritten_query(self, router_agent):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"rewritten_query": "Who directed Inception?", '
            '"method": "local", "confidence": 0.9, "reasoning": "entity query"}'
        )
        with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await router_agent.route("Who directed it?", "movies collection")
            assert result.rewritten_query == "Who directed Inception?"

    @pytest.mark.asyncio
    async def test_route_falls_back_to_original_query_when_rewrite_missing(self, router_agent):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"method": "local", "confidence": 0.8, "reasoning": "entity query"}'
        )
        with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await router_agent.route("Who directed it?", "movies collection")
            assert result.rewritten_query == "Who directed it?"

    @pytest.mark.asyncio
    async def test_route_injects_summary_into_prompt(self, router_agent):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"rewritten_query": "Who directed Inception?", '
            '"method": "local", "confidence": 0.9, "reasoning": "entity"}'
        )
        with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            await router_agent.route(
                "Who directed it?", "movies",
                conversation_summary="User asked about Inception (2010)."
            )
            call_prompt = mock_llm.call_args[0][0]
            assert "User asked about Inception (2010)." in call_prompt

    @pytest.mark.asyncio
    async def test_route_injects_history_into_prompt(self, router_agent):
        from backend.app.models.schemas import ConversationTurn
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"rewritten_query": "Who directed Inception?", '
            '"method": "local", "confidence": 0.9, "reasoning": "entity"}'
        )
        history = [
            ConversationTurn(role="user", content="Tell me about Inception",
                             rewritten_query="Tell me about Inception", method_used="local"),
            ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
        ]
        with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            await router_agent.route("Who directed it?", "movies", conversation_history=history)
            call_prompt = mock_llm.call_args[0][0]
            assert "Tell me about Inception" in call_prompt
            assert "Inception is a 2010 film" in call_prompt


class TestSchemaModels:
    """Test new conversation schema models."""

    def test_conversation_turn_user_with_metadata(self):
        turn = ConversationTurn(
            role="user",
            content="Who directed it?",
            rewritten_query="Who directed Inception?",
            method_used="local",
        )
        assert turn.role == "user"
        assert turn.rewritten_query == "Who directed Inception?"
        assert turn.method_used == "local"

    def test_conversation_turn_assistant_no_metadata(self):
        turn = ConversationTurn(role="assistant", content="Christopher Nolan directed Inception.")
        assert turn.rewritten_query is None
        assert turn.method_used is None

    def test_agent_search_request_defaults(self):
        req = AgentSearchRequest(query="hello")
        assert req.conversation_history == []
        assert req.conversation_summary is None

    def test_agent_search_request_accepts_summary_and_history(self):
        req = AgentSearchRequest(
            query="Who directed it?",
            conversation_summary="User asked about Inception.",
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ],
        )
        assert req.conversation_summary == "User asked about Inception."
        assert len(req.conversation_history) == 2

    def test_agent_search_response_has_rewritten_query(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="entity query",
            rewritten_query="Who directed Inception?",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query == "Who directed Inception?"

    def test_agent_search_response_rewritten_query_optional(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="entity query",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query is None

    def test_summarize_request_model(self):
        req = SummarizeRequest(
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ],
            existing_summary="Previous summary.",
        )
        assert len(req.conversation_history) == 2
        assert req.existing_summary == "Previous summary."

    def test_summarize_request_no_existing_summary(self):
        req = SummarizeRequest(
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception"),
            ]
        )
        assert req.existing_summary is None

    def test_summarize_response_model(self):
        resp = SummarizeResponse(
            summary="User asked about Inception.",
            trimmed_history=[
                ConversationTurn(role="user", content="Who directed it?",
                                 rewritten_query="Who directed Inception?", method_used="local"),
                ConversationTurn(role="assistant", content="Christopher Nolan."),
            ],
        )
        assert resp.summary == "User asked about Inception."
        assert len(resp.trimmed_history) == 2
