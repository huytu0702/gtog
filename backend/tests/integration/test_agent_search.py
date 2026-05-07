"""Integration tests for agent search."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from backend.app.main import app


class TestAgentSearchIntegration:
    """Integration tests for agent search flow."""

    @pytest.fixture
    async def client(self):
        """Create async test client."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_full_agent_search_flow_with_web_fallback(self, client):
        """GraphRAG runs first, then web fallback when judge marks insufficient."""
        mock_route = MagicMock()
        mock_route.method = "local"
        mock_route.confidence = 0.9
        mock_route.reasoning = "Focused entity query"
        mock_route.rewritten_query = "What are the latest FDA regulations?"

        mock_graphrag_result = MagicMock()
        mock_graphrag_result.response = "I do not have enough indexed information."
        mock_graphrag_result.context_data = {}

        mock_web_result = MagicMock()
        mock_web_result.response = "The FDA regulations..."
        mock_web_result.sources = []

        with patch(
            "backend.app.routers.search.router_agent.route",
            new_callable=AsyncMock,
        ) as mock_router:
            with patch(
                "backend.app.routers.search.query_service.local_search",
                new_callable=AsyncMock,
            ) as mock_local:
                with patch(
                    "backend.app.routers.search._should_trigger_web_fallback",
                    new_callable=AsyncMock,
                ) as mock_should_fallback:
                    with patch(
                        "backend.app.routers.search.web_search_service.search",
                        new_callable=AsyncMock,
                    ) as mock_web:
                        mock_router.return_value = mock_route
                        mock_local.return_value = mock_graphrag_result
                        mock_should_fallback.return_value = True
                        mock_web.return_value = mock_web_result

                        response = await client.post(
                            "/api/collections/test/search/agent",
                            json={"query": "What are latest FDA regulations?", "stream": False},
                        )

                        assert response.status_code == 200
                        data = response.json()
                        assert data["method_used"] == "local"
                        assert data["web_search_triggered"] is True
                        assert "FDA" in data["web_response"]

    @pytest.mark.asyncio
    async def test_summarize_endpoint_returns_summary_and_trimmed_history(self, client):
        """POST /agent/summarize returns summary and trimmed history."""
        mock_summary = "User explored Inception (2010) film."

        with patch(
            "backend.app.routers.search.summarization_service.summarize",
            new_callable=AsyncMock,
        ) as mock_summarize:
            mock_summarize.return_value = mock_summary

            response = await client.post(
                "/api/collections/test/search/agent/summarize",
                json={
                    "conversation_history": [
                        {"role": "user", "content": "Tell me about Inception",
                         "rewritten_query": "Tell me about Inception", "method_used": "local"},
                        {"role": "assistant", "content": "Inception is a 2010 film..."},
                        {"role": "user", "content": "Who directed it?",
                         "rewritten_query": "Who directed Inception?", "method_used": "local"},
                        {"role": "assistant", "content": "Christopher Nolan."},
                    ],
                    "existing_summary": None,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert data["summary"] == mock_summary
            assert "trimmed_history" in data
            assert isinstance(data["trimmed_history"], list)

    @pytest.mark.asyncio
    async def test_agent_search_passes_history_and_summary_to_router(self, client):
        """agent_search endpoint passes both conversation_history and conversation_summary to router."""
        mock_route = MagicMock()
        mock_route.method = "local"
        mock_route.confidence = 0.9
        mock_route.reasoning = "entity query"
        mock_route.rewritten_query = "Who directed Inception?"

        mock_result = MagicMock()
        mock_result.response = "Christopher Nolan."
        mock_result.context_data = {}

        with patch(
            "backend.app.routers.search.router_agent.route",
            new_callable=AsyncMock,
        ) as mock_router:
            with patch(
                "backend.app.routers.search.query_service.local_search",
                new_callable=AsyncMock,
            ) as mock_local:
                with patch(
                    "backend.app.routers.search._should_trigger_web_fallback",
                    new_callable=AsyncMock,
                ) as mock_should_fallback:
                    mock_router.return_value = mock_route
                    mock_local.return_value = mock_result
                    mock_should_fallback.return_value = False

                    response = await client.post(
                        "/api/collections/test/search/agent",
                        json={
                            "query": "Who directed it?",
                            "stream": False,
                            "conversation_summary": "User asked about Inception (2010).",
                            "conversation_history": [
                                {"role": "user", "content": "Who starred in it?",
                                 "rewritten_query": "Who starred in Inception?", "method_used": "local"},
                                {"role": "assistant", "content": "Leonardo DiCaprio."},
                            ],
                        },
                    )

                    assert response.status_code == 200
                    args, kwargs = mock_router.call_args
                    assert kwargs.get("conversation_summary") == "User asked about Inception (2010)."
                    history = kwargs.get("conversation_history") or (args[2] if len(args) > 2 else None)
                    assert history is not None
                    assert len(history) == 2

    @pytest.mark.asyncio
    async def test_agent_search_response_includes_rewritten_query(self, client):
        """AgentSearchResponse includes rewritten_query field."""
        mock_route = MagicMock()
        mock_route.method = "local"
        mock_route.confidence = 0.9
        mock_route.reasoning = "entity query"
        mock_route.rewritten_query = "Who directed Inception?"

        mock_result = MagicMock()
        mock_result.response = "Christopher Nolan."
        mock_result.context_data = {}

        with patch(
            "backend.app.routers.search.router_agent.route",
            new_callable=AsyncMock,
        ) as mock_router:
            with patch(
                "backend.app.routers.search.query_service.local_search",
                new_callable=AsyncMock,
            ) as mock_local:
                with patch(
                    "backend.app.routers.search._should_trigger_web_fallback",
                    new_callable=AsyncMock,
                ) as mock_should_fallback:
                    mock_router.return_value = mock_route
                    mock_local.return_value = mock_result
                    mock_should_fallback.return_value = False

                    response = await client.post(
                        "/api/collections/test/search/agent",
                        json={"query": "Who directed it?", "stream": False},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["rewritten_query"] == "Who directed Inception?"

    @pytest.mark.asyncio
    async def test_agent_search_uses_rewritten_query_for_search(self, client):
        """agent_search calls search methods with rewritten_query, not original query."""
        mock_route = MagicMock()
        mock_route.method = "local"
        mock_route.confidence = 0.9
        mock_route.reasoning = "entity query"
        mock_route.rewritten_query = "Who directed Inception?"

        mock_result = MagicMock()
        mock_result.response = "Christopher Nolan."
        mock_result.context_data = {}

        with patch(
            "backend.app.routers.search.router_agent.route",
            new_callable=AsyncMock,
        ) as mock_router:
            with patch(
                "backend.app.routers.search.query_service.local_search",
                new_callable=AsyncMock,
            ) as mock_local:
                with patch(
                    "backend.app.routers.search._should_trigger_web_fallback",
                    new_callable=AsyncMock,
                ) as mock_should_fallback:
                    mock_router.return_value = mock_route
                    mock_local.return_value = mock_result
                    mock_should_fallback.return_value = False

                    await client.post(
                        "/api/collections/test/search/agent",
                        json={"query": "Who directed it?", "stream": False},
                    )

                    args, kwargs = mock_local.call_args
                    query_used = kwargs.get("query") or args[1]
                    assert query_used == "Who directed Inception?"
