"""Tests for indexing endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import backend.app.main as main
from backend.app.main import app
from backend.app.models import IndexJobResponse, IndexStatus, IndexStatusResponse


@pytest.mark.asyncio
async def test_start_indexing_returns_202_and_job_payload(valid_edge_secret_headers):
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        with patch("backend.app.routers.indexing.storage_service") as mock_storage:
            with patch("backend.app.routers.indexing.indexing_service") as mock_indexing:
                mock_storage.get_collection.return_value = SimpleNamespace(document_count=2)
                mock_indexing.start_indexing = AsyncMock(
                    return_value=IndexStatusResponse(
                        collection_id="c1",
                        job_id="job-1",
                        status=IndexStatus.PENDING,
                        progress=0.0,
                        message="Indexing job queued",
                        attempt=0,
                        max_attempts=3,
                    )
                )

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    response = await client.post(
                        "/api/collections/c1/index",
                        headers=valid_edge_secret_headers,
                    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_get_collection_index_status_returns_retrying_state(
    valid_edge_secret_headers,
):
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        with patch("backend.app.routers.indexing.indexing_service") as mock_indexing:
            mock_indexing.get_index_status.return_value = IndexStatusResponse(
                collection_id="c1",
                job_id="job-1",
                status=IndexStatus.RETRYING,
                progress=0.0,
                message="Retry scheduled",
                attempt=1,
                max_attempts=3,
                error="temporary failure",
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get(
                    "/api/collections/c1/index",
                    headers=valid_edge_secret_headers,
                )

    assert response.status_code == 200
    assert response.json()["status"] == "retrying"


@pytest.mark.asyncio
async def test_get_job_status_returns_canonical_payload(valid_edge_secret_headers):
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        with patch("backend.app.routers.indexing.indexing_service") as mock_indexing:
            mock_indexing.get_job_status.return_value = IndexJobResponse(
                job_id="job-1",
                collection_id="c1",
                status="running",
                attempt=1,
                max_attempts=3,
                target_version="v1",
                progress=25.0,
                message="Running indexing pipeline...",
                lease_owner_id="worker-a",
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get(
                    "/api/index-jobs/job-1",
                    headers=valid_edge_secret_headers,
                )

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["lease_owner_id"] == "worker-a"


@pytest.mark.asyncio
async def test_get_job_status_returns_404_when_missing(valid_edge_secret_headers):
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        with patch("backend.app.routers.indexing.indexing_service") as mock_indexing:
            mock_indexing.get_job_status.return_value = None

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get(
                    "/api/index-jobs/missing",
                    headers=valid_edge_secret_headers,
                )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_start_indexing_requires_edge_secret_when_configured():
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post("/api/collections/c1/index")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_start_indexing_returns_503_without_secret_for_non_local_origins():
    with patch.object(main.settings, "edge_origin_secret", ""), patch.object(
        main.settings, "require_edge_auth", False
    ):
        with patch.object(main, "allowed_origins", ["https://app.example.com"]):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/collections/c1/index",
                    headers={"Origin": "https://app.example.com"},
                )

    assert response.status_code == 503
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"


@pytest.mark.asyncio
async def test_get_job_status_rejects_wrong_edge_secret():
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get(
                "/api/index-jobs/job-1",
                headers={"X-Edge-Secret": "wrong-secret"},
            )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_start_indexing_returns_cors_headers_on_403_for_allowed_origin():
    with patch.object(main.settings, "edge_origin_secret", "secret-123"), patch.object(
        main.settings, "require_edge_auth", True
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/collections/c1/index",
                headers={"Origin": main.allowed_origins[0]},
            )

    assert response.status_code == 403
    assert response.headers["access-control-allow-origin"] == main.allowed_origins[0]
