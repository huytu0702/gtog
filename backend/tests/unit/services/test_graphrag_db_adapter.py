"""Tests for GraphRAGDbAdapter entity ingestion."""

from typing import Sequence
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, IndexRun, Entity
from app.services.graphrag_db_adapter import GraphRAGDbAdapter


async def _create_collection(db_session: AsyncSession) -> Collection:
    collection = Collection(name=f"Test Collection {uuid4()}"[:30])
    db_session.add(collection)
    await db_session.flush()
    await db_session.refresh(collection)
    return collection


async def _create_index_run(
    db_session: AsyncSession, collection_id
) -> IndexRun:
    index_run = IndexRun(collection_id=collection_id)
    db_session.add(index_run)
    await db_session.flush()
    await db_session.refresh(index_run)
    return index_run


@pytest.mark.asyncio
async def test_adapter_inserts_entities(db_session: AsyncSession):
    """Adapter should persist provided entities."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    entities: Sequence[dict] = [
        {"id": uuid4(), "title": "Entity 1", "type": "Person"},
        {"id": uuid4(), "title": "Entity 2", "type": "Org"},
    ]

    await adapter.insert_entities(collection.id, index_run.id, entities)

    result = (await db_session.execute(select(Entity))).scalars().all()
    titles = sorted(entity.title for entity in result)

    assert len(result) == 2
    assert titles == ["Entity 1", "Entity 2"]
