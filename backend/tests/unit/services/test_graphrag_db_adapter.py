"""Tests for GraphRAGDbAdapter ingestion helpers."""

from typing import Sequence
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Collection,
    IndexRun,
    Entity,
    Relationship,
    Community,
    CommunityReport,
    TextUnit,
    Covariate,
)
from app.db.models.embeddings import Embedding, EmbeddingType
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
        {"id": "entity-1", "title": "Entity 1", "type": "Person"},
        {"id": "entity-2", "title": "Entity 2", "type": "Org"},
    ]

    await adapter.insert_entities(collection.id, index_run.id, entities)

    result = (await db_session.execute(select(Entity))).scalars().all()
    titles = sorted(entity.title for entity in result)

    assert len(result) == 2
    assert titles == ["Entity 1", "Entity 2"]


@pytest.mark.asyncio
async def test_adapter_inserts_relationships(db_session: AsyncSession):
    """Adapter should persist relationship outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    relationships: Sequence[dict] = [
        {
            "id": "rel-1",
            "source": "Entity 1",
            "target": "Entity 2",
            "description": "Relates 1 to 2",
        },
        {
            "id": "rel-2",
            "source": "Entity 2",
            "target": "Entity 3",
            "weight": 0.42,
        },
    ]

    await adapter.insert_relationships(collection.id, index_run.id, relationships)

    result = (await db_session.execute(select(Relationship))).scalars().all()
    edges = sorted((row.source, row.target) for row in result)

    assert len(result) == 2
    assert edges == [("Entity 1", "Entity 2"), ("Entity 2", "Entity 3")]


@pytest.mark.asyncio
async def test_adapter_inserts_communities(db_session: AsyncSession):
    """Adapter should persist community outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    communities: Sequence[dict] = [
        {"id": "comm-1", "community": 1, "level": 0, "title": "Top"},
        {"id": "comm-2", "community": 2, "level": 1, "title": "Child"},
    ]

    await adapter.insert_communities(collection.id, index_run.id, communities)

    result = (await db_session.execute(select(Community))).scalars().all()
    levels = sorted(row.level for row in result)

    assert len(result) == 2
    assert levels == [0, 1]


@pytest.mark.asyncio
async def test_adapter_inserts_community_reports(db_session: AsyncSession):
    """Adapter should persist community report outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    reports: Sequence[dict] = [
        {"id": "report-1", "community": 1, "level": 0, "summary": "Summary"},
        {"id": "report-2", "community": 2, "level": 0, "summary": "Other"},
    ]

    await adapter.insert_community_reports(collection.id, index_run.id, reports)

    result = (await db_session.execute(select(CommunityReport))).scalars().all()
    summaries = sorted(row.summary for row in result)

    assert len(result) == 2
    assert summaries == ["Other", "Summary"]


@pytest.mark.asyncio
async def test_adapter_inserts_text_units(db_session: AsyncSession):
    """Adapter should persist text unit outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    text_units: Sequence[dict] = [
        {"id": "text-1", "text": "Paragraph 1", "n_tokens": 10},
        {"id": "text-2", "text": "Paragraph 2", "n_tokens": 12},
    ]

    await adapter.insert_text_units(collection.id, index_run.id, text_units)

    result = (await db_session.execute(select(TextUnit))).scalars().all()
    contents = sorted(row.text for row in result)

    assert len(result) == 2
    assert contents == ["Paragraph 1", "Paragraph 2"]


@pytest.mark.asyncio
async def test_adapter_inserts_covariates(db_session: AsyncSession):
    """Adapter should persist covariate outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    covariates: Sequence[dict] = [
        {
            "id": "cov-1",
            "covariate_type": "claim",
            "description": "First",
            "status": "active",
        },
        {
            "id": "cov-2",
            "covariate_type": "claim",
            "description": "Second",
            "status": "inactive",
        },
    ]

    await adapter.insert_covariates(collection.id, index_run.id, covariates)

    result = (await db_session.execute(select(Covariate))).scalars().all()
    descriptions = sorted(row.description for row in result)

    assert len(result) == 2
    assert descriptions == ["First", "Second"]


@pytest.mark.asyncio
async def test_adapter_inserts_embeddings(db_session: AsyncSession):
    """Adapter should persist embedding outputs."""
    collection = await _create_collection(db_session)
    index_run = await _create_index_run(db_session, collection.id)
    adapter = GraphRAGDbAdapter(db_session)

    embeddings: Sequence[dict] = [
        {
            "embedding_type": EmbeddingType.TEXT_UNIT.value,
            "ref_id": uuid4(),
            "vector": [0.1] * 1536,
        },
        {
            "embedding_type": EmbeddingType.ENTITY.value,
            "ref_id": uuid4(),
            "vector": [0.2] * 1536,
        },
    ]

    await adapter.insert_embeddings(collection.id, index_run.id, embeddings)

    result = (await db_session.execute(select(Embedding))).scalars().all()
    types = sorted(row.embedding_type for row in result)

    assert len(result) == 2
    assert types == [EmbeddingType.ENTITY, EmbeddingType.TEXT_UNIT]
