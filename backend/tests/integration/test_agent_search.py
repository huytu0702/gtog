"""Integration tests for agent search."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from backend.app.main import app


class TestAgentSearchIntegration:
    """Integration tests for agent search flow."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_full_agent_search_flow(self, client):
        """Test complete agent search from request to response."""
        # This test verifies the full flow works end-to-end
        # with mocked external services

        mock_route = MagicMock()
        mock_route.method = "web"
        mock_route.confidence = 0.9
        mock_route.reasoning = "External information needed"

        mock_web_result = MagicMock()
        mock_web_result.response = "The FDA regulations..."
        mock_web_result.sources = []

        with patch(
            "backend.app.services.router_agent.RouterAgent.route",
            new_callable=AsyncMock,
        ) as mock_router:
            with patch(
                "backend.app.services.web_search.WebSearchService.search",
                new_callable=AsyncMock,
            ) as mock_web:
                mock_router.return_value = mock_route
                mock_web.return_value = mock_web_result

                response = client.post(
                    "/api/collections/test/search/agent",
                    json={"query": "What are latest FDA regulations?", "stream": False},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["method_used"] == "web"
                assert (
                    "FDA" in data["response"]
                    or "regulations" in data["response"].lower()
                )
