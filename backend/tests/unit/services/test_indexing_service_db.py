"""Tests for database-backed indexing service."""

from datetime import datetime
from uuid import uuid4

import pytest
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, IndexRun, IndexRunStatus
from app.services.indexing_service_db import IndexingServiceDB


@pytest.fixture(autouse=True)
def stub_enqueue_job(monkeypatch):
    """Prevent actual queue interactions during tests."""

    def _fake_enqueue(*args, **kwargs) -> str:  # pragma: no cover - simple stub
        return "job-123"

    monkeypatch.setattr(
        "app.services.indexing_service_db.enqueue_indexing_job",
        _fake_enqueue,
    )


async def _create_collection(db_session: AsyncSession) -> Collection:
    collection = Collection(name=f"Test Collection {uuid4()}"[:30])
    db_session.add(collection)
    await db_session.flush()
    await db_session.refresh(collection)
    return collection


async def _create_index_run(
    db_session: AsyncSession,
    collection_id,
    status: IndexRunStatus,
) -> IndexRun:
    run = IndexRun(
        collection_id=collection_id,
        status=status,
        started_at=datetime.utcnow(),
    )
    db_session.add(run)
    await db_session.flush()
    await db_session.refresh(run)
    return run


@pytest.mark.asyncio
async def test_start_indexing_rejects_if_run_in_progress(db_session: AsyncSession):
    """Queued/running runs should block new indexing attempts."""
    collection = await _create_collection(db_session)
    await _create_index_run(db_session, collection.id, IndexRunStatus.RUNNING)

    service = IndexingServiceDB(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.start_indexing(collection.id)

    assert exc.value.status_code == status.HTTP_409_CONFLICT
    assert "already" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_start_indexing_rejects_if_run_already_queued(db_session: AsyncSession):
    """Existing queued runs should result in conflict."""
    collection = await _create_collection(db_session)
    await _create_index_run(db_session, collection.id, IndexRunStatus.QUEUED)

    service = IndexingServiceDB(db_session)

    with pytest.raises(HTTPException) as exc:
        await service.start_indexing(collection.id)

    assert exc.value.status_code == status.HTTP_409_CONFLICT
