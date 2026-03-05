"""Tests for API guards, fallback rate limit, and readiness endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import backend.app.main as main
from backend.app.main import app


@pytest.mark.asyncio
async def test_api_rejects_missing_afd_secret_when_configured():
    with patch.object(main.settings, "afd_origin_secret", "secret-123"):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_api_rejects_missing_principal_when_afd_secret_matches():
    with patch.object(main.settings, "afd_origin_secret", "secret-123"):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"X-AFD-Secret": "secret-123"},
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_api_allows_request_when_guards_present():
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with patch.object(main.settings, "afd_origin_secret", "secret-123"):
        with patch("backend.app.routers.search.web_search_service") as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/web",
                    headers={
                        "X-AFD-Secret": "secret-123",
                        "X-MS-CLIENT-PRINCIPAL": "present",
                    },
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_fallback_rate_limiter_returns_429():
    with patch.object(main.settings, "afd_origin_secret", ""):
        with patch.object(main.settings, "rate_limit_enabled", True):
            with patch("backend.app.main._rate_limiter", main.InMemoryRateLimiter(1)):
                with patch("backend.app.routers.search.web_search_service") as mock_web:
                    mock_result = MagicMock()
                    mock_result.response = "ok"
                    mock_result.sources = []
                    mock_web.search = AsyncMock(return_value=mock_result)

                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        first = await client.post(
                            "/api/collections/test-collection/search/web",
                            json={"query": "hello", "stream": False},
                        )
                        second = await client.post(
                            "/api/collections/test-collection/search/web",
                            json={"query": "hello", "stream": False},
                        )

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_readiness_returns_200_when_all_checks_pass():
    with patch("backend.app.main._check_cosmos_ready", return_value=(True, "ok")):
        with patch("backend.app.main._check_blob_ready", return_value=(True, "ok")):
            with patch("backend.app.main._check_search_ready", return_value=(True, "ok")):
                with patch("backend.app.main._check_key_vault_ready", return_value=(True, "ok")):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        response = await client.get("/health/readiness")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_readiness_returns_503_when_any_check_fails():
    with patch("backend.app.main._check_cosmos_ready", return_value=(False, "cosmos down")):
        with patch("backend.app.main._check_blob_ready", return_value=(True, "ok")):
            with patch("backend.app.main._check_search_ready", return_value=(True, "ok")):
                with patch("backend.app.main._check_key_vault_ready", return_value=(True, "ok")):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        response = await client.get("/health/readiness")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["cosmos"]["ok"] is False
