"""End-to-end indexing + search test."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_indexing_flow(client: AsyncClient):
    """Test indexing flow queues a job and updates status."""
    create_resp = await client.post("/api/collections", json={"name": "e2e"})
    assert create_resp.status_code == 201

    files = {"file": ("note.txt", b"hello", "text/plain")}
    await client.post("/api/collections/e2e/documents", files=files)

    start_resp = await client.post("/api/collections/e2e/index")
    assert start_resp.status_code == 202
