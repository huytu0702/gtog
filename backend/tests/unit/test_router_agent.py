"""Tests for Router Agent service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.services.router_agent import RouterAgent, RouteDecision


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
