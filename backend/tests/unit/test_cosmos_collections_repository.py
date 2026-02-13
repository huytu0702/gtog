import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from app.repositories.cosmos_collections import CosmosCollectionRepository


@pytest.fixture
def mock_cosmos_container():
    """Mock Cosmos DB container."""
    container = MagicMock()
    container.upsert_item = MagicMock(return_value={
        "id": "collection:demo",
        "kind": "collection",
        "collection_id": "demo",
        "name": "demo",
        "description": "desc",
        "created_at": datetime.utcnow().isoformat(),
    })
    container.read_item = MagicMock(side_effect=Exception("Not found"))
    container.query_items = MagicMock(return_value=[])
    container.delete_item = MagicMock(return_value=True)
    return container


def test_create_collection_upserts_item(mock_cosmos_container):
    repo = CosmosCollectionRepository(mock_cosmos_container)
    rec = repo.create("demo", "desc")
    assert rec.id == "demo"
    assert rec.name == "demo"
    assert rec.description == "desc"
    mock_cosmos_container.upsert_item.assert_called_once()


def test_get_collection_returns_none_when_missing(mock_cosmos_container):
    repo = CosmosCollectionRepository(mock_cosmos_container)
    assert repo.get("missing") is None


def test_get_collection_returns_record_when_found(mock_cosmos_container):
    mock_cosmos_container.read_item = MagicMock(return_value={
        "id": "collection:demo",
        "kind": "collection",
        "collection_id": "demo",
        "name": "demo",
        "description": "desc",
        "created_at": datetime.utcnow().isoformat(),
    })
    repo = CosmosCollectionRepository(mock_cosmos_container)
    rec = repo.get("demo")
    assert rec is not None
    assert rec.id == "demo"


def test_list_collections_returns_list(mock_cosmos_container):
    mock_cosmos_container.query_items = MagicMock(return_value=[
        {
            "id": "collection:demo1",
            "kind": "collection",
            "collection_id": "demo1",
            "name": "demo1",
            "description": "desc1",
            "created_at": datetime.utcnow().isoformat(),
        },
        {
            "id": "collection:demo2",
            "kind": "collection",
            "collection_id": "demo2",
            "name": "demo2",
            "description": "desc2",
            "created_at": datetime.utcnow().isoformat(),
        },
    ])
    repo = CosmosCollectionRepository(mock_cosmos_container)
    recs = repo.list()
    assert len(recs) == 2
    assert recs[0].id == "demo1"
    assert recs[1].id == "demo2"


def test_delete_collection_returns_true(mock_cosmos_container):
    repo = CosmosCollectionRepository(mock_cosmos_container)
    result = repo.delete("demo")
    assert result is True
    mock_cosmos_container.delete_item.assert_called_once_with(item="collection:demo", partition_key="collection:demo")
