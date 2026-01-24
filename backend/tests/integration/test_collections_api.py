"""Integration tests for collections API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_collection(client: AsyncClient):
    """Test creating a new collection."""
    response = await client.post(
        "/api/collections",
        json={"name": "test-collection", "description": "Test description"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-collection"
    assert data["description"] == "Test description"
    assert data["document_count"] == 0
    assert data["indexed"] is False


@pytest.mark.asyncio
async def test_create_duplicate_collection_returns_409(client: AsyncClient):
    """Test creating duplicate collection returns 409."""
    # Create first
    await client.post("/api/collections", json={"name": "duplicate-test"})

    # Try to create duplicate
    response = await client.post("/api/collections", json={"name": "duplicate-test"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_collections(client: AsyncClient):
    """Test listing collections."""
    # Create some collections
    await client.post("/api/collections", json={"name": "list-test-1"})
    await client.post("/api/collections", json={"name": "list-test-2"})

    response = await client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert data["total"] >= 2


@pytest.mark.asyncio
async def test_get_collection(client: AsyncClient):
    """Test getting a specific collection."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "get-test"})
    collection_id = create_resp.json()["id"]

    # Get by ID
    response = await client.get(f"/api/collections/{collection_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "get-test"


@pytest.mark.asyncio
async def test_get_collection_by_name(client: AsyncClient):
    """Test getting collection by name."""
    await client.post("/api/collections", json={"name": "name-lookup-test"})

    response = await client.get("/api/collections/name-lookup-test")
    assert response.status_code == 200
    assert response.json()["name"] == "name-lookup-test"


@pytest.mark.asyncio
async def test_get_nonexistent_collection_returns_404(client: AsyncClient):
    """Test getting nonexistent collection returns 404."""
    response = await client.get("/api/collections/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_collection(client: AsyncClient):
    """Test deleting a collection."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "delete-test"})
    collection_id = create_resp.json()["id"]

    # Delete
    response = await client.delete(f"/api/collections/{collection_id}")
    assert response.status_code == 204

    # Verify deleted
    get_resp = await client.get(f"/api/collections/{collection_id}")
    assert get_resp.status_code == 404
