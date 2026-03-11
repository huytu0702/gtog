"""Tests for Web Search service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.web_search import WebSearchResult, WebSearchService


class TestWebSearchService:
    """Test WebSearchService class."""

    @pytest.fixture
    def web_search_service(self):
        """Create WebSearchService instance with mocked clients."""
        with patch("backend.app.services.web_search.AsyncTavilyClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.search = AsyncMock()
            mock_client_cls.return_value = mock_client
            return WebSearchService()

    @pytest.mark.asyncio
    async def test_search_returns_web_search_result(self, web_search_service):
        """search() should return a WebSearchResult object."""
        mock_tavily_result = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com",
                    "content": "Test content",
                }
            ]
        }
        web_search_service.tavily.search = AsyncMock(return_value=mock_tavily_result)

        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "Synthesized response [1]"

        with patch.object(
            web_search_service, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_llm_response

            result = await web_search_service.search("test query")

        assert isinstance(result, WebSearchResult)
        assert "Synthesized response" in result.response
        assert len(result.sources) >= 1

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, web_search_service):
        """search() should handle empty Tavily results gracefully."""
        web_search_service.tavily.search = AsyncMock(return_value={"results": []})

        result = await web_search_service.search("obscure query")

        assert isinstance(result, WebSearchResult)
        assert "no relevant results" in result.response.lower() or result.response != ""

    @pytest.mark.asyncio
    async def test_search_handles_tavily_error(self, web_search_service):
        """search() should handle Tavily API errors gracefully."""
        web_search_service.tavily.search = AsyncMock(side_effect=Exception("API Error"))

        result = await web_search_service.search("test query")

        assert isinstance(result, WebSearchResult)
        assert "error" in result.response.lower() or result.sources == []
