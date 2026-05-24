"""Pytest configuration for backend tests."""

from unittest.mock import patch

import pytest


@pytest.fixture
def valid_edge_secret_headers() -> dict[str, str]:
    return {"X-Edge-Secret": "secret-123"}


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("backend.app.config.settings") as mock:
        mock.openai_api_key = "test-key"
        mock.tavily_api_key = "test-tavily-key"
        mock.default_chat_model = "gpt-4o-mini"
        yield mock
