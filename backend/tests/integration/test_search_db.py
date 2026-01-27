"""Integration tests for Postgres-backed search."""

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db.models import Entity, IndexRun, IndexRunStatus


@pytest.mark.asyncio
async def test_search_returns_results_from_db(client, db_session):
    """Test search returns results from Postgres database."""
    # Create a collection
    create_resp = await client.post(
        "/api/collections",
        json={"name": "search-test", "description": "Test search from DB"},
    )
    assert create_resp.status_code == 201
    collection_id = create_resp.json()["id"]

    # Create a completed index run with test entities
    index_run_id = uuid4()
    index_run = IndexRun(
        id=index_run_id,
        collection_id=collection_id,
        status=IndexRunStatus.COMPLETED,
    )
    db_session.add(index_run)

    # Add test entities
    test_entities = [
        Entity(
            id=uuid4(),
            collection_id=collection_id,
            index_run_id=index_run_id,
            title="Test Entity 1",
            type="Organization",
            description="A test organization",
            graphrag_id="e1",
        ),
        Entity(
            id=uuid4(),
            collection_id=collection_id,
            index_run_id=index_run_id,
            title="Test Entity 2",
            type="Person",
            description="A test person",
            graphrag_id="e2",
        ),
    ]
    for entity in test_entities:
        db_session.add(entity)

    await db_session.commit()

    # Perform search
    response = await client.get(f"/api/collections/{collection_id}/search?query=test")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test"
    # Response should contain answer and context
    assert "response" in data
