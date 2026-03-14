"""Tests for search router endpoints."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from backend.app.errors import ServingContextUnavailableError
from backend.app.main import app
from backend.app.models import AgentSearchRequest
from backend.app.routers.search import (
    SSE_HEARTBEAT_INTERVAL_SECONDS,
    _build_agent_stream_response,
)


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
        with patch("backend.app.routers.search.settings.enable_tog_debug_endpoint", True):
            with patch(
                "backend.app.routers.search.query_service.get_tog_entities_preview",
                return_value={
                    "collection_id": "test-collection",
                    "source": "cosmos:v1",
                    "total_entities": 1,
                    "showing_first": 1,
                    "entities": [
                        {
                            "id": "Entity A",
                            "description": "Entity A description",
                            "type": "org",
                        }
                    ],
                },
            ):
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


class TestSearchErrorMapping:
    """Test HTTP mapping for serving context failures."""

    @pytest.mark.asyncio
    async def test_global_returns_503_when_serving_unavailable(self):
        with patch(
            "backend.app.routers.search.query_service.global_search",
            new=AsyncMock(side_effect=ServingContextUnavailableError("cosmos down")),
        ):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/global",
                    json={"query": "hello"},
                )

        assert response.status_code == 503


class TestAgentStreamEndpoint:
    """Test GET /agent/stream EventSource contract."""

    @pytest.mark.asyncio
    async def test_agent_stream_get_returns_event_stream_headers(self):
        mock_route_decision = MagicMock()
        mock_route_decision.method = "web"
        mock_route_decision.confidence = 0.95
        mock_route_decision.reasoning = "Needs web context"
        mock_route_decision.rewritten_query = None

        async def fake_stream(_query: str):
            yield "stream chunk"

        with patch("backend.app.routers.search.router_agent") as mock_router:
            with patch("backend.app.routers.search.web_search_service") as mock_web:
                mock_router.route = AsyncMock(return_value=mock_route_decision)
                mock_web.search_streaming = fake_stream

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.get(
                        "/api/collections/test-collection/search/agent/stream",
                        params={"query": "What changed?"},
                    )

        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"

    def test_build_agent_stream_response_configures_heartbeat_events(self):
        response = _build_agent_stream_response(
            "test-collection",
            AgentSearchRequest(query="What changed?", stream=True),
        )

        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("X-Accel-Buffering") == "no"
        assert response.ping_interval == SSE_HEARTBEAT_INTERVAL_SECONDS

        heartbeat_event = response.ping_message_factory()
        assert heartbeat_event.event == "heartbeat"
        assert json.loads(heartbeat_event.data) == {"message": "keepalive"}
