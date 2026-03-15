"""Pytest configuration for backend tests."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def valid_easy_auth_principal() -> str:
    principal = {
        "auth_typ": "aad",
        "name_typ": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
        "role_typ": "http://schemas.microsoft.com/ws/2008/06/identity/claims/role",
        "claims": [
            {
                "typ": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier",
                "val": "user-123",
            },
            {
                "typ": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
                "val": "test.user@example.com",
            },
        ],
    }
    return base64.b64encode(json.dumps(principal).encode("utf-8")).decode("ascii")


@pytest.fixture
def valid_easy_auth_headers(valid_easy_auth_principal: str) -> dict[str, str]:
    return {"X-MS-CLIENT-PRINCIPAL": valid_easy_auth_principal}


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("backend.app.config.settings") as mock:
        mock.openai_api_key = "test-key"
        mock.tavily_api_key = "test-tavily-key"
        mock.default_chat_model = "gpt-4o-mini"
        mock.collections_dir = MagicMock()
        yield mock
