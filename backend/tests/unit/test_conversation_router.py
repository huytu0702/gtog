"""Tests for conversation session router endpoints."""

from unittest.mock import patch

import httpx
import pytest

from backend.app.errors import ConversationSessionNotFoundError
from backend.app.main import app


@pytest.mark.asyncio
async def test_create_session_returns_201():
    with patch(
        "backend.app.routers.conversation.conversation_service.create_session",
        return_value={
            "sessionId": "s1",
            "createdAt": "2026-03-03T10:00:00",
        },
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post("/api/collections/c1/sessions")

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == "s1"
    assert body["collection_id"] == "c1"


@pytest.mark.asyncio
async def test_get_session_returns_404_for_missing_session():
    with patch(
        "backend.app.routers.conversation.conversation_service.get_session_view",
        side_effect=ConversationSessionNotFoundError("missing"),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/api/collections/c1/sessions/s-missing")

    assert response.status_code == 404
