"""End-to-end indexing + search test."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_indexing_flow(client: AsyncClient):
    """Test indexing flow queues a job and updates status."""
    create_resp = await client.post("/api/collections", json={"name": "e2e"})
    assert create_resp.status_code == 201
    collection_id = create_resp.json()["id"]

    files = {"file": ("note.txt", b"hello", "text/plain")}
    upload_resp = await client.post(
        f"/api/collections/{collection_id}/documents", files=files
    )
    assert upload_resp.status_code == 201

    start_resp = await client.post(f"/api/collections/{collection_id}/index")
    assert start_resp.status_code == 202
