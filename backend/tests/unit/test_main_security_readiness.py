"""Tests for edge-secret guard, fallback rate limit, and readiness endpoint."""

import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import backend.app.main as main
from backend.app.main import app


@pytest.mark.asyncio
async def test_api_rejects_missing_edge_secret_when_configured():
    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
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
async def test_api_rejects_wrong_edge_secret_when_configured():
    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"X-Edge-Secret": "wrong-secret"},
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_api_returns_cors_headers_on_403_for_allowed_origin():
    allowed_origin = main.allowed_origins[0]

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"Origin": allowed_origin},
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 403
    assert response.headers["access-control-allow-origin"] == allowed_origin


@pytest.mark.asyncio
async def test_api_does_not_return_cors_headers_for_unknown_origin_on_403():
    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"Origin": "https://evil.example.com"},
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_api_allows_request_with_valid_edge_secret_when_configured(
    valid_edge_secret_headers,
):
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        with patch("backend.app.routers.search.web_search_service") as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/web",
                    headers=valid_edge_secret_headers,
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_allows_request_from_trusted_tunnel_proxy_without_edge_secret():
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
        patch("backend.app.main._connection_ip", return_value="100.100.0.64"),
    ):
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
                        "CF-Ray": "test-ray",
                        "CF-Connecting-IP": "203.0.113.10",
                    },
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_rejects_public_proxy_headers_without_edge_secret():
    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
        patch("backend.app.main._connection_ip", return_value="198.51.100.20"),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/collections/test-collection/search/web",
                headers={
                    "CF-Ray": "test-ray",
                    "CF-Connecting-IP": "203.0.113.10",
                },
                json={"query": "hello", "stream": False},
            )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_api_ignores_unrelated_header_when_edge_secret_matches(
    valid_edge_secret_headers,
):
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
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
                        **valid_edge_secret_headers,
                        "X-Unrelated-Header": "ignored",
                    },
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_allows_request_without_edge_secret_for_local_origins_only():
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with (
        patch.object(main.settings, "edge_origin_secret", ""),
        patch.object(main.settings, "require_edge_auth", False),
    ):
        with patch.object(main, "allowed_origins", ["http://localhost:3000"]):
            with patch("backend.app.routers.search.web_search_service") as mock_web:
                mock_web.search = AsyncMock(return_value=mock_result)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    response = await client.post(
                        "/api/collections/test-collection/search/web",
                        json={"query": "hello", "stream": False},
                    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_returns_503_without_edge_secret_for_non_local_origins():
    with (
        patch.object(main.settings, "edge_origin_secret", ""),
        patch.object(main.settings, "require_edge_auth", False),
    ):
        with patch.object(main, "allowed_origins", ["https://app.example.com"]):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/web",
                    headers={"Origin": "https://app.example.com"},
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 503
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"


@pytest.mark.asyncio
async def test_api_allows_request_when_edge_secret_has_surrounding_whitespace(
    valid_edge_secret_headers,
):
    mock_result = MagicMock()
    mock_result.response = "ok"
    mock_result.sources = []

    with (
        patch.object(main.settings, "edge_origin_secret", "  secret-123  "),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        with patch("backend.app.routers.search.web_search_service") as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/collections/test-collection/search/web",
                    headers=valid_edge_secret_headers,
                    json={"query": "hello", "stream": False},
                )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_fallback_rate_limiter_returns_429_for_local_origins_without_secret():
    with (
        patch.object(main.settings, "edge_origin_secret", ""),
        patch.object(main.settings, "require_edge_auth", False),
    ):
        with patch.object(main, "allowed_origins", ["http://localhost:3000"]):
            with patch.object(main.settings, "rate_limit_enabled", True):
                with patch(
                    "backend.app.main._rate_limiter", main.InMemoryRateLimiter(1)
                ):
                    with patch(
                        "backend.app.routers.search.web_search_service"
                    ) as mock_web:
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


def test_rate_limiter_evicts_stale_keys_after_allow() -> None:
    limiter = main.InMemoryRateLimiter(2)
    limiter._events = {
        "stale": deque([1.0]),
        "fresh": deque([80.0]),
    }
    limiter._next_stale_key_prune_at = 0.0

    with patch("backend.app.main.monotonic", return_value=100.0):
        allowed, retry_after = limiter.allow("active")

    assert allowed is True
    assert retry_after == 0
    assert "stale" not in limiter._events
    assert "fresh" in limiter._events
    assert "active" in limiter._events


@pytest.mark.asyncio
async def test_rate_limiter_returns_cors_headers_on_429_for_allowed_origin():
    allowed_origin = "http://localhost:3000"

    with (
        patch.object(main.settings, "edge_origin_secret", ""),
        patch.object(main.settings, "require_edge_auth", False),
    ):
        with patch.object(main, "allowed_origins", [allowed_origin]):
            with patch.object(main.settings, "rate_limit_enabled", True):
                with patch(
                    "backend.app.main._rate_limiter", main.InMemoryRateLimiter(1)
                ):
                    with patch(
                        "backend.app.routers.search.web_search_service"
                    ) as mock_web:
                        mock_result = MagicMock()
                        mock_result.response = "ok"
                        mock_result.sources = []
                        mock_web.search = AsyncMock(return_value=mock_result)

                        transport = httpx.ASGITransport(app=app)
                        async with httpx.AsyncClient(
                            transport=transport,
                            base_url="http://testserver",
                        ) as client:
                            await client.post(
                                "/api/collections/test-collection/search/web",
                                headers={"Origin": allowed_origin},
                                json={"query": "hello", "stream": False},
                            )
                            response = await client.post(
                                "/api/collections/test-collection/search/web",
                                headers={"Origin": allowed_origin},
                                json={"query": "hello", "stream": False},
                            )

    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == allowed_origin


@pytest.mark.asyncio
async def test_readiness_returns_200_when_all_checks_pass():
    with patch("backend.app.main._check_cosmos_ready", return_value=(True, "ok")):
        with patch("backend.app.main._check_blob_ready", return_value=(True, "ok")):
            with patch(
                "backend.app.main._check_queue_ready", return_value=(True, "ok")
            ):
                with patch(
                    "backend.app.main._check_search_ready", return_value=(True, "ok")
                ):
                    with patch(
                        "backend.app.main._check_key_vault_ready",
                        return_value=(True, "ok"),
                    ):
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
        "queue": {"ok": True, "detail": "ok"},
        "search": {"ok": True, "detail": "ok"},
        "key_vault": {"ok": True, "detail": "ok"},
    }


@pytest.mark.asyncio
async def test_readiness_returns_503_when_any_check_fails():
    with patch(
        "backend.app.main._check_cosmos_ready", return_value=(False, "cosmos down")
    ):
        with patch("backend.app.main._check_blob_ready", return_value=(True, "ok")):
            with patch(
                "backend.app.main._check_queue_ready", return_value=(True, "ok")
            ):
                with patch(
                    "backend.app.main._check_search_ready", return_value=(True, "ok")
                ):
                    with patch(
                        "backend.app.main._check_key_vault_ready",
                        return_value=(True, "ok"),
                    ):
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
    assert body["checks"]["queue"] == {"ok": True, "detail": "ok"}
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

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
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
async def test_cors_preflight_does_not_bypass_api_guards():
    allowed_origin = main.allowed_origins[0]

    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            preflight = await client.options(
                "/api/collections/test-collection/search/web",
                headers={
                    "Origin": allowed_origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            request = await client.post(
                "/api/collections/test-collection/search/web",
                headers={"Origin": allowed_origin},
                json={"query": "hello", "stream": False},
            )

    assert preflight.status_code == 200
    assert request.status_code == 403


@pytest.mark.asyncio
async def test_public_health_endpoint_does_not_open_api_access():
    with (
        patch.object(main.settings, "edge_origin_secret", "secret-123"),
        patch.object(main.settings, "require_edge_auth", True),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            api_request = await client.post(
                "/api/collections/test-collection/search/web",
                json={"query": "hello", "stream": False},
            )

    assert health.status_code == 200
    assert api_request.status_code == 403


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

    with (
        patch.object(main.settings, "edge_origin_secret", ""),
        patch.object(main.settings, "require_edge_auth", False),
    ):
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
    assert payload["client_ip"] == "127.0.0.1"
    assert payload["request_id"]
