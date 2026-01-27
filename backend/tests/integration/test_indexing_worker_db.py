"""Integration tests for indexing worker GraphRAG persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db.models import (
    Collection,
    Document,
    Entity,
    IndexRunStatus,
    Relationship,
)
from app.repositories import CollectionRepository, DocumentRepository, IndexRunRepository
from app.worker.tasks import run_indexing_task


@dataclass(slots=True)
class _FakePipelineRunResult:
    """Lightweight stub matching graphrag.index.typing.pipeline_run_result."""

    workflow: str
    result: Any
    state: dict[str, Any]
    errors: list[BaseException] | None


@pytest.mark.asyncio
async def test_worker_persists_outputs(monkeypatch, db_session):
    collection_repo = CollectionRepository(db_session)
    document_repo = DocumentRepository(db_session)
    run_repo = IndexRunRepository(db_session)

    collection = await collection_repo.create(Collection(name="Worker Test Collection"))

    doc = Document(
        collection_id=collection.id,
        title="Doc",
        text=None,
        doc_metadata=None,
        filename="doc.txt",
        content_type="text/plain",
        bytes_content=b"GraphRAG worker contents",
    )
    await document_repo.create(doc)

    run = await run_repo.create(
        collection_id=collection.id,
        status=IndexRunStatus.QUEUED,
        job_id=str(uuid4()),
    )

    fake_outputs = [
        _FakePipelineRunResult(
            workflow="entities",
            result=None,
            state={
                "entities": [
                    {"id": "1", "title": "Entity", "type": "Person"},
                ],
                "relationships": [
                    {"id": "1", "source": "Entity", "target": "Entity", "description": "self"},
                ],
                "communities": [
                    {"id": "1", "community": 1, "level": 0, "title": "Root"},
                ],
                "community_reports": [
                    {"id": "1", "community": 1, "level": 0, "summary": "Report"},
                ],
                "text_units": [
                    {"id": "1", "text": "Unit", "n_tokens": 5},
                ],
                "covariates": [
                    {"id": "1", "covariate_type": "claim", "description": "Claim"},
                ],
                "embeddings": [
                    {
                        "embedding_type": "text_unit",
                        "ref_id": uuid4(),
                        "vector": [0.1, 0.2],
                    }
                ],
            },
            errors=None,
        )
    ]

    async def _fake_build_index(*args, **kwargs):
        return fake_outputs

    monkeypatch.setattr("app.worker.tasks.api.build_index", _fake_build_index)

    result = await asyncio.to_thread(
        run_indexing_task,
        str(collection.id),
        str(run.id),
    )

    assert result["status"] == "completed"

    entity_count = await db_session.scalar(select(Entity).count())
    relationship_count = await db_session.scalar(select(Relationship).count())
    assert entity_count and entity_count > 0
    assert relationship_count and relationship_count > 0

    refreshed_run = await run_repo.get_by_id(run.id)
    assert refreshed_run.status == IndexRunStatus.COMPLETED
    assert refreshed_run.finished_at is not None
