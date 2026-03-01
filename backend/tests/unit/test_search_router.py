"""Tests for search router endpoints."""

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pandas as pd
import pytest

from backend.app.main import app


class TestAgentSearchEndpoint:
    """Test /agent search endpoint."""

    @pytest.mark.asyncio
    async def test_agent_search_returns_200(self):
        """POST /agent should return 200 with valid response."""
        # Mock router agent
        mock_route_decision = MagicMock()
        mock_route_decision.method = "local"
        mock_route_decision.confidence = 0.85
        mock_route_decision.reasoning = "Specific entity query"
        mock_route_decision.rewritten_query = None

        # Mock query service
        mock_search_response = MagicMock()
        mock_search_response.query = "test"
        mock_search_response.response = "Test response"
        mock_search_response.context_data = None
        mock_search_response.method = "local"

        with patch("backend.app.routers.search.router_agent") as mock_router:
            with patch("backend.app.routers.search.query_service") as mock_query:
                mock_router.route = AsyncMock(return_value=mock_route_decision)
                mock_query.local_search = AsyncMock(return_value=mock_search_response)

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/api/collections/test-collection/search/agent",
                        json={"query": "What is chamomile?", "stream": False},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "method_used" in data
                    assert "router_reasoning" in data


class TestWebSearchEndpoint:
    """Test /web search endpoint."""

    @pytest.mark.asyncio
    async def test_web_search_returns_200(self):
        """POST /web should return 200 with valid response."""
        mock_result = MagicMock()
        mock_result.response = "Web search result"
        mock_result.sources = []

        with patch("backend.app.routers.search.web_search_service") as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/web",
                    json={"query": "What are latest FDA regulations?", "stream": False},
                )

                assert response.status_code == 200
                data = response.json()
                assert "response" in data


class TestToGDebugEndpoint:
    """Test /tog/debug endpoint production gating."""

    @pytest.mark.asyncio
    async def test_tog_debug_returns_404_when_disabled(self):
        with patch("backend.app.routers.search.settings.enable_tog_debug_endpoint", False):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/collections/test-collection/search/tog/debug")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_tog_debug_returns_data_when_enabled(self):
        entities_df = pd.DataFrame(
            [{"title": "Entity A", "description": "Entity A description", "type": "org"}]
        )

        with patch("backend.app.routers.search.settings.enable_tog_debug_endpoint", True):
            with patch(
                "backend.app.utils.get_search_data_paths",
                return_value={"entities": "entities.parquet"},
            ):
                with patch("pandas.read_parquet", return_value=entities_df):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        response = await client.get(
                            "/api/collections/test-collection/search/tog/debug"
                        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_entities"] == 1
