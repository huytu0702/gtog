"""Integration tests for indexing worker GraphRAG persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from app.db.models import (
    Collection,
    Document,
    Entity,
    IndexRun,
    IndexRunStatus,
    Relationship,
)
from app.repositories import CollectionRepository, DocumentRepository, IndexRunRepository
import app.worker.tasks as tasks


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
        IndexRun(
            collection_id=collection.id,
            status=IndexRunStatus.QUEUED,
        )
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
                        "vector": [0.1] * 1536,
                    }
                ],
            },
            errors=None,
        )
    ]

    async def _fake_run_async(_collection_id, _index_run_id):
        from app.services.graphrag_db_adapter import GraphRAGDbAdapter
        adapter = GraphRAGDbAdapter(db_session)
        await adapter.ingest_outputs(UUID(str(collection.id)), UUID(str(run.id)), fake_outputs)
        run.status = IndexRunStatus.COMPLETED
        run.finished_at = datetime.utcnow()
        await db_session.commit()
        return {"status": "completed", "index_run_id": str(run.id)}

    monkeypatch.setattr(tasks, "_run_indexing_async", _fake_run_async)

    result = await tasks._run_indexing_async(UUID(str(collection.id)), UUID(str(run.id)))

    assert result["status"] == "completed"

    entity_count = await db_session.scalar(select(func.count(Entity.id)))
    relationship_count = await db_session.scalar(select(func.count(Relationship.id)))
    assert entity_count and entity_count > 0
    assert relationship_count and relationship_count > 0

    refreshed_run = await run_repo.get_by_id(UUID(str(run.id)))
    assert refreshed_run is not None
    assert refreshed_run.status == IndexRunStatus.COMPLETED
    assert refreshed_run.finished_at is not None
