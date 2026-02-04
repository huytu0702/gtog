"""Pytest configuration for backend tests."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("backend.app.config.settings") as mock:
        mock.openai_api_key = "test-key"
        mock.tavily_api_key = "test-tavily-key"
        mock.default_chat_model = "gpt-4o-mini"
        mock.collections_dir = MagicMock()
        yield mock
