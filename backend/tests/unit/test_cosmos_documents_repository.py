import base64
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from app.repositories.cosmos_documents import CosmosDocumentRepository


@pytest.fixture
def mock_cosmos_container():
    """Mock Cosmos DB container."""
    container = MagicMock()
    container.upsert_item = MagicMock(return_value={
        "id": "document:demo:a.txt",
        "kind": "document",
        "collection_id": "demo",
        "name": "a.txt",
        "size": 5,
        "uploaded_at": datetime.utcnow().isoformat(),
        "content_b64": base64.b64encode(b"hello").decode("ascii"),
    })
    container.read_item = MagicMock(side_effect=Exception("Not found"))
    container.query_items = MagicMock(return_value=[])
    container.delete_item = MagicMock(return_value=True)
    return container


def test_put_document_stores_payload(mock_cosmos_container):
    repo = CosmosDocumentRepository(mock_cosmos_container)
    rec = repo.put("demo", "a.txt", b"hello")
    assert rec.size == 5
    assert rec.name == "a.txt"
    assert rec.collection_id == "demo"
    mock_cosmos_container.upsert_item.assert_called_once()


def test_list_documents_filters_collection(mock_cosmos_container):
    mock_cosmos_container.query_items = MagicMock(return_value=[
        {
            "id": "document:demo:a.txt",
            "kind": "document",
            "collection_id": "demo",
            "name": "a.txt",
            "size": 5,
            "uploaded_at": datetime.utcnow().isoformat(),
        },
    ])
    repo = CosmosDocumentRepository(mock_cosmos_container)
    docs = repo.list("demo")
    assert len(docs) == 1
    assert docs[0].name == "a.txt"


def test_get_content_returns_bytes(mock_cosmos_container):
    mock_cosmos_container.read_item = MagicMock(return_value={
        "id": "document:demo:a.txt",
        "kind": "document",
        "collection_id": "demo",
        "name": "a.txt",
        "size": 5,
        "uploaded_at": datetime.utcnow().isoformat(),
        "content_b64": base64.b64encode(b"hello").decode("ascii"),
    })
    repo = CosmosDocumentRepository(mock_cosmos_container)
    content = repo.get_content("demo", "a.txt")
    assert content == b"hello"


def test_get_content_returns_none_when_missing(mock_cosmos_container):
    repo = CosmosDocumentRepository(mock_cosmos_container)
    content = repo.get_content("demo", "missing.txt")
    assert content is None


def test_delete_document_returns_true(mock_cosmos_container):
    repo = CosmosDocumentRepository(mock_cosmos_container)
    result = repo.delete("demo", "a.txt")
    assert result is True
    mock_cosmos_container.delete_item.assert_called_once_with(
        item="document:demo:a.txt",
        partition_key="document:demo:a.txt",
    )
