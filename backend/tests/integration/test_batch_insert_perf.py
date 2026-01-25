"""Performance tests for batch inserts."""

import pytest
import time
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import Collection, Document, Entity, Relationship


@pytest.mark.asyncio
async def test_batch_insert_documents_performance(db_session: AsyncSession):
    """Test batch insert of documents with timing logs."""
    # Create collection
    collection = Collection(id=uuid4(), name="perf-test")
    db_session.add(collection)
    await db_session.flush()

    # Batch insert documents
    batch_size = 100
    start_time = time.perf_counter()

    for i in range(batch_size):
        doc = Document(
            id=uuid4(),
            collection_id=collection.id,
            title=f"Document {i}",
            text=f"Content for document {i}",
            filename=f"doc_{i}.txt",
            content_type="text/plain",
            bytes_content=f"Content for document {i}".encode(),
        )
        db_session.add(doc)

    await db_session.commit()
    elapsed = time.perf_counter() - start_time

    # Verify all documents inserted
    result = await db_session.execute(
        select(Document).where(Document.collection_id == collection.id)
    )
    docs = result.scalars().all()

    assert len(docs) == batch_size
    # Log timing (no strict threshold)
    pytest.log_time(f"Inserted {batch_size} documents in {elapsed:.3f}s")


@pytest.mark.asyncio
async def test_batch_insert_entities_performance(db_session: AsyncSession):
    """Test batch insert of entities with timing logs."""
    # Create collection
    collection = Collection(id=uuid4(), name="perf-test-entities")
    index_run_id = uuid4()
    db_session.add(collection)
    await db_session.flush()

    # Batch insert entities
    batch_size = 100
    start_time = time.perf_counter()

    for i in range(batch_size):
        entity = Entity(
            id=uuid4(),
            collection_id=collection.id,
            index_run_id=index_run_id,
            title=f"Entity_{i}",
            type="person",
            description=f"Entity number {i}",
        )
        db_session.add(entity)

    await db_session.commit()
    elapsed = time.perf_counter() - start_time

    # Verify all entities inserted
    result = await db_session.execute(
        select(Entity).where(Entity.collection_id == collection.id)
    )
    entities = result.scalars().all()

    assert len(entities) == batch_size
    # Log timing (no strict threshold)
    pytest.log_time(f"Inserted {batch_size} entities in {elapsed:.3f}s")


@pytest.mark.asyncio
async def test_batch_insert_relationships_performance(db_session: AsyncSession):
    """Test batch insert of relationships with timing logs."""
    # Create collection
    collection = Collection(id=uuid4(), name="perf-test-relationships")
    index_run_id = uuid4()
    db_session.add(collection)
    await db_session.flush()

    # Batch insert relationships
    batch_size = 100
    start_time = time.perf_counter()

    for i in range(batch_size):
        rel = Relationship(
            id=uuid4(),
            collection_id=collection.id,
            index_run_id=index_run_id,
            source=f"Entity_{i}",
            target=f"Entity_{(i + 1) % batch_size}",
            description=f"Relationship {i}",
        )
        db_session.add(rel)

    await db_session.commit()
    elapsed = time.perf_counter() - start_time

    # Verify all relationships inserted
    result = await db_session.execute(
        select(Relationship).where(Relationship.collection_id == collection.id)
    )
    rels = result.scalars().all()

    assert len(rels) == batch_size
    # Log timing (no strict threshold)
    pytest.log_time(f"Inserted {batch_size} relationships in {elapsed:.3f}s")
