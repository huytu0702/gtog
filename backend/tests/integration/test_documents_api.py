"""Integration tests for documents API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_and_list_documents(client: AsyncClient):
    """Test uploading and listing documents."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "docs-test"})
    assert create_resp.status_code == 201

    # Upload document
    files = {"file": ("note.txt", b"hello", "text/plain")}
    upload_resp = await client.post(
        "/api/collections/docs-test/documents",
        files=files,
    )
    assert upload_resp.status_code == 201

    # List documents
    list_resp = await client.get("/api/collections/docs-test/documents")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] == 1
    assert data["documents"][0]["name"] == "note.txt"


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient):
    """Test deleting a document."""
    await client.post("/api/collections", json={"name": "docs-delete"})
    files = {"file": ("delete.txt", b"bye", "text/plain")}
    await client.post("/api/collections/docs-delete/documents", files=files)

    delete_resp = await client.delete(
        "/api/collections/docs-delete/documents/delete.txt"
    )
    assert delete_resp.status_code == 204

    list_resp = await client.get("/api/collections/docs-delete/documents")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 0
