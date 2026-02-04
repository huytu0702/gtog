"""Tests for search router endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app


class TestAgentSearchEndpoint:
    """Test /agent search endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_agent_search_returns_200(self, client):
        """POST /agent should return 200 with valid response."""
        # Mock router agent
        mock_route_decision = MagicMock()
        mock_route_decision.method = "local"
        mock_route_decision.confidence = 0.85
        mock_route_decision.reasoning = "Specific entity query"

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

                response = client.post(
                    "/api/collections/test-collection/search/agent",
                    json={"query": "What is chamomile?", "stream": False},
                )

                assert response.status_code == 200
                data = response.json()
                assert "method_used" in data
                assert "router_reasoning" in data


class TestWebSearchEndpoint:
    """Test /web search endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_web_search_returns_200(self, client):
        """POST /web should return 200 with valid response."""
        mock_result = MagicMock()
        mock_result.response = "Web search result"
        mock_result.sources = []

        with patch("backend.app.routers.search.web_search_service") as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)

            response = client.post(
                "/api/collections/test-collection/search/web",
                json={"query": "What are latest FDA regulations?", "stream": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
