"""Tests for indexing endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.main import app
from backend.app.models import IndexJobResponse, IndexStatus, IndexStatusResponse


@pytest.mark.asyncio
async def test_start_indexing_returns_202_and_job_payload():
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
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/api/collections/c1/index",
                    headers={"X-MS-CLIENT-PRINCIPAL": "present"},
                )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_get_collection_index_status_returns_retrying_state():
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
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                "/api/collections/c1/index",
                headers={"X-MS-CLIENT-PRINCIPAL": "present"},
            )

    assert response.status_code == 200
    assert response.json()["status"] == "retrying"


@pytest.mark.asyncio
async def test_get_job_status_returns_canonical_payload():
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
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                "/api/index-jobs/job-1",
                headers={"X-MS-CLIENT-PRINCIPAL": "present"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-1"
    assert body["lease_owner_id"] == "worker-a"


@pytest.mark.asyncio
async def test_get_job_status_returns_404_when_missing():
    with patch("backend.app.routers.indexing.indexing_service") as mock_indexing:
        mock_indexing.get_job_status.return_value = None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                "/api/index-jobs/missing",
                headers={"X-MS-CLIENT-PRINCIPAL": "present"},
            )

    assert response.status_code == 404
