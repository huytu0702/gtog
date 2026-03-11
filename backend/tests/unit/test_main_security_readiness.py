"""Tests for API guards, fallback rate limit, and readiness endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import backend.app.main as main
from backend.app.main import app


@pytest.mark.asyncio
async def test_api_rejects_missing_edge_secret_when_configured():
    with patch.object(main.settings, "edge_origin_secret", "secret-123"):
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
async def test_api_rejects_missing_principal_when_edge_secret_matches():
    with patch.object(main.settings, "edge_origin_secret", "secret-123"):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"X-Edge-Secret": "secret-123"},
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_api_allows_request_when_guards_present():
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with patch.object(main.settings, "edge_origin_secret", "secret-123"):
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
                        "X-Edge-Secret": "secret-123",
                        "X-MS-CLIENT-PRINCIPAL": "present",
                    },
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_fallback_rate_limiter_returns_429():
    with patch.object(main.settings, "edge_origin_secret", ""):
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
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "cosmos": {"ok": True, "detail": "ok"},
        "blob": {"ok": True, "detail": "ok"},
        "search": {"ok": True, "detail": "ok"},
        "key_vault": {"ok": True, "detail": "ok"},
    }


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
    assert body["checks"]["cosmos"] == {"ok": False, "detail": "cosmos down"}
    assert body["checks"]["blob"] == {"ok": True, "detail": "ok"}
    assert body["checks"]["search"] == {"ok": True, "detail": "ok"}
    assert body["checks"]["key_vault"] == {"ok": True, "detail": "ok"}


@pytest.mark.asyncio
async def test_cors_preflight_allows_configured_origin():
    allowed_origin = main.allowed_origins[0]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.options(
            "/api/collections/test-collection/search/web",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == allowed_origin


@pytest.mark.asyncio
async def test_cors_preflight_allows_configured_origin_when_edge_secret_enabled():
    allowed_origin = main.allowed_origins[0]

    with patch.object(main.settings, "edge_origin_secret", "secret-123"):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.options(
                "/api/collections/test-collection/search/web",
                headers={
                    "Origin": allowed_origin,
                    "Access-Control-Request-Method": "POST",
                },
            )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == allowed_origin


@pytest.mark.asyncio
async def test_cors_preflight_rejects_unknown_origin():
    unknown_origin = "https://evil.example.com"
    assert unknown_origin not in main.allowed_origins

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.options(
            "/api/collections/test-collection/search/web",
            headers={
                "Origin": unknown_origin,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_request_logging_includes_cloudflare_headers(caplog):
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with patch.object(main.settings, "edge_origin_secret", ""):
        with patch.object(main.settings, "rate_limit_enabled", False):
            with patch("backend.app.routers.search.web_search_service") as mock_web:
                mock_web.search = AsyncMock(return_value=mock_result)
                transport = httpx.ASGITransport(app=app)
                with caplog.at_level("INFO"):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        response = await client.post(
                            "/api/collections/test-collection/search/web",
                            headers={
                                "Cf-Ray": "abc123",
                                "CF-Connecting-IP": "203.0.113.10",
                            },
                            json={"query": "hello", "stream": False},
                        )

    assert response.status_code == 200
    payload = None
    for record in caplog.records:
        message = record.getMessage()
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            continue
        if parsed.get("event") == "http_request":
            payload = parsed

    assert payload is not None
    assert payload["cf_ray"] == "abc123"
    assert payload["cf_connecting_ip"] == "203.0.113.10"
    assert payload["client_ip"] == "203.0.113.10"
    assert payload["request_id"]
