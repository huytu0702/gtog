"""Tests for worker tasks."""

from uuid import UUID
from unittest.mock import AsyncMock

import pytest

from graphrag.config.models.graph_rag_config import GraphRagConfig
from app.db.models import Collection, Document
from app.repositories import CollectionRepository, DocumentRepository
import app.worker.tasks as tasks


async def _create_collection(db_session: object) -> Collection:
    repo = CollectionRepository(db_session)
    return await repo.create(Collection(name="Test Collection"))


async def _create_document(db_session: object, collection_id: UUID) -> Document:
    repo = DocumentRepository(db_session)
    document = Document(
        collection_id=collection_id,
        title="Doc",
        text=None,
        doc_metadata=None,
        filename="doc.txt",
        content_type="text/plain",
        bytes_content=b"GraphRAG worker contents",
    )
    return await repo.create(document)


@pytest.mark.asyncio
async def test_run_graphrag_pipeline_uses_graphrag_config(monkeypatch, db_session):
    config = GraphRagConfig.model_construct()

    async def fake_build_index(**kwargs):
        assert kwargs.get("config") is config
        return []

    def _fake_load_config(*_args, **_kwargs):
        _ = (_args, _kwargs)
        return config

    monkeypatch.setattr(
        "app.utils.helpers.load_graphrag_config",
        _fake_load_config,
    )
    monkeypatch.setattr(
        "graphrag.api.build_index",
        AsyncMock(side_effect=fake_build_index),
    )

    collection = await _create_collection(db_session)
    await _create_document(db_session, collection.id)

    await tasks._run_graphrag_pipeline(
        db_session,
        UUID(str(collection.id)),
        UUID(str(collection.id)),
    )


@pytest.mark.asyncio
async def test_run_graphrag_pipeline_passes_input_documents(monkeypatch, db_session):
    config = GraphRagConfig.model_construct()

    async def fake_build_index(**kwargs):
        input_documents = kwargs.get("input_documents")
        assert input_documents is not None
        assert "text" in input_documents.columns
        return []

    def _fake_load_config(*_args, **_kwargs):
        _ = (_args, _kwargs)
        return config

    monkeypatch.setattr(
        "app.utils.helpers.load_graphrag_config",
        _fake_load_config,
    )
    monkeypatch.setattr(
        "graphrag.api.build_index",
        AsyncMock(side_effect=fake_build_index),
    )

    collection = await _create_collection(db_session)
    await _create_document(db_session, collection.id)

    await tasks._run_graphrag_pipeline(
        db_session,
        UUID(str(collection.id)),
        UUID(str(collection.id)),
    )
