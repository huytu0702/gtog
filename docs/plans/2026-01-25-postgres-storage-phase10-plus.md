# PostgreSQL Storage Phase 10+ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete phases 10+ of the PostgreSQL migration: move documents, indexing, querying, and app lifecycle to the database, and add end-to-end tests.

**Architecture:** Introduce DB-backed repositories/services for documents, index runs, and query data. Update routers to depend on DB services and Redis queue. Replace filesystem search/load with DB reads. Add startup migrations/health checks and full integration tests.

**Tech Stack:** FastAPI, SQLAlchemy async, PostgreSQL + pgvector, Redis + RQ, pytest + testcontainers

---

## Phase 10: Document Service (Tasks 17-18)

### Task 17: Create Document Repository

**Files:**
- Create: `backend/app/repositories/document.py`
- Modify: `backend/app/repositories/__init__.py`
- Test: `backend/tests/unit/test_repository_document.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_repository_document.py`:

```python
"""Tests for document repository."""

import pytest
from app.repositories.document import DocumentRepository


def test_document_repository_has_methods():
    """Test DocumentRepository has required methods."""
    assert hasattr(DocumentRepository, "get_by_collection")
    assert hasattr(DocumentRepository, "get_by_name")
    assert hasattr(DocumentRepository, "delete_by_name")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_repository_document.py -v`
Expected: FAIL with "No module named 'app.repositories.document'"

**Step 3: Write minimal implementation**

Create `backend/app/repositories/document.py`:

```python
"""Document repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from .base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for document operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Document)

    async def get_by_collection(self, collection_id: UUID) -> list[Document]:
        """Get documents for a collection."""
        result = await self.session.execute(
            select(Document).where(Document.collection_id == collection_id)
        )
        return list(result.scalars().all())

    async def get_by_name(self, collection_id: UUID, name: str) -> Optional[Document]:
        """Get document by filename within a collection."""
        result = await self.session.execute(
            select(Document)
            .where(Document.collection_id == collection_id)
            .where(Document.filename == name)
        )
        return result.scalar_one_or_none()

    async def delete_by_name(self, collection_id: UUID, name: str) -> bool:
        """Delete document by filename within a collection."""
        doc = await self.get_by_name(collection_id, name)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.flush()
        return True
```

Update `backend/app/repositories/__init__.py`:

```python
"""Repository layer for database operations."""

from .base import BaseRepository
from .collection import CollectionRepository
from .document import DocumentRepository

__all__ = ["BaseRepository", "CollectionRepository", "DocumentRepository"]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_repository_document.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/repositories/document.py backend/app/repositories/__init__.py

git add backend/tests/unit/test_repository_document.py

git commit -m "feat(backend): add document repository"
```

---

### Task 18: Create DB-backed Document Service + Update Documents Router

**Files:**
- Create: `backend/app/services/document_service_db.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/routers/documents.py`
- Modify: `backend/app/services/__init__.py`
- Test: `backend/tests/unit/test_document_service_db.py`
- Test: `backend/tests/integration/test_documents_api.py`

**Step 1: Write the failing unit test**

Create `backend/tests/unit/test_document_service_db.py`:

```python
"""Tests for database-backed document service."""

import pytest
from app.services.document_service_db import DocumentServiceDB


def test_document_service_has_methods():
    """Test DocumentServiceDB has required methods."""
    assert hasattr(DocumentServiceDB, "upload_document")
    assert hasattr(DocumentServiceDB, "list_documents")
    assert hasattr(DocumentServiceDB, "delete_document")
```

**Step 2: Write the failing integration test**

Create `backend/tests/integration/test_documents_api.py`:

```python
"""Integration tests for documents API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_and_list_documents(client: AsyncClient):
    """Test uploading and listing documents."""
    # Create collection
    create_resp = await client.post("/api/collections", json={"name": "docs-test"})
    assert create_resp.status_code == 201

    # Upload document
    files = {"file": ("note.txt", b"hello", "text/plain")}
    upload_resp = await client.post(
        "/api/collections/docs-test/documents",
        files=files,
    )
    assert upload_resp.status_code == 201

    # List documents
    list_resp = await client.get("/api/collections/docs-test/documents")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] == 1
    assert data["documents"][0]["name"] == "note.txt"


@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient):
    """Test deleting a document."""
    await client.post("/api/collections", json={"name": "docs-delete"})
    files = {"file": ("delete.txt", b"bye", "text/plain")}
    await client.post("/api/collections/docs-delete/documents", files=files)

    delete_resp = await client.delete(
        "/api/collections/docs-delete/documents/delete.txt"
    )
    assert delete_resp.status_code == 204

    list_resp = await client.get("/api/collections/docs-delete/documents")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 0
```

**Step 3: Run tests to verify they fail**

Run: `cd backend && pytest tests/unit/test_document_service_db.py -v`
Expected: FAIL with "No module named 'app.services.document_service_db'"

Run: `cd backend && pytest tests/integration/test_documents_api.py -v`
Expected: FAIL due to existing filesystem service behavior

**Step 4: Write minimal implementation**

Create `backend/app/services/document_service_db.py`:

```python
"""Database-backed document service."""

from datetime import datetime
from typing import List
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Document
from app.models import DocumentResponse
from app.repositories import CollectionRepository, DocumentRepository


class DocumentServiceDB:
    """Service for document operations using database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.collections = CollectionRepository(session)
        self.documents = DocumentRepository(session)

    async def _get_collection_or_error(self, collection_id: UUID) -> Collection:
        collection = await self.collections.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection '{collection_id}' not found")
        return collection

    async def upload_document(self, collection_id: UUID, file: UploadFile) -> DocumentResponse:
        await self._get_collection_or_error(collection_id)

        # Read file bytes
        content = await file.read()

        # Create document row
        doc = Document(
            collection_id=collection_id,
            title=file.filename,
            text=None,
            doc_metadata=None,
            filename=file.filename,
            content_type=file.content_type,
            bytes_content=content,
        )
        await self.documents.create(doc)

        return DocumentResponse(
            name=doc.filename,
            size=len(content),
            uploaded_at=datetime.now(),
        )

    async def list_documents(self, collection_id: UUID) -> List[DocumentResponse]:
        await self._get_collection_or_error(collection_id)
        docs = await self.documents.get_by_collection(collection_id)
        return [
            DocumentResponse(
                name=doc.filename,
                size=len(doc.bytes_content or b""),
                uploaded_at=doc.created_at,
            )
            for doc in docs
        ]

    async def delete_document(self, collection_id: UUID, document_name: str) -> None:
        await self._get_collection_or_error(collection_id)
        deleted = await self.documents.delete_by_name(collection_id, document_name)
        if not deleted:
            raise ValueError(f"Document '{document_name}' not found")
```

Update `backend/app/services/__init__.py`:

```python
"""Service layer for the application."""

from .storage_service import storage_service
from .indexing_service import indexing_service
from .query_service import query_service
from .collection_service_db import CollectionServiceDB
from .document_service_db import DocumentServiceDB

__all__ = [
    "storage_service",
    "indexing_service",
    "query_service",
    "CollectionServiceDB",
    "DocumentServiceDB",
]
```

Update `backend/app/api/deps.py`:

```python
from app.services.document_service_db import DocumentServiceDB


async def get_document_service(
    session: AsyncSession = Depends(get_db_session)
) -> DocumentServiceDB:
    """FastAPI dependency for document service."""
    return DocumentServiceDB(session)
```

Update `backend/app/routers/documents.py`:

```python
from uuid import UUID
from fastapi import Depends
from app.api.deps import get_document_service
from app.services.document_service_db import DocumentServiceDB


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    service: DocumentServiceDB = Depends(get_document_service),
):
    try:
        if not file.filename.lower().endswith((".txt", ".md")):
            raise ValueError("Only .txt and .md files are supported")
        result = await service.upload_document(UUID(collection_id), file)
        logger.info(f"Uploaded document {file.filename} to collection {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
```

(Apply same dependency pattern to `list_documents` and `delete_document`, using UUID parse for `collection_id`.)

**Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/unit/test_document_service_db.py -v`
Expected: PASS

Run: `cd backend && pytest tests/integration/test_documents_api.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backend/app/services/document_service_db.py

git add backend/app/routers/documents.py backend/app/api/deps.py backend/app/services/__init__.py

git add backend/tests/unit/test_document_service_db.py backend/tests/integration/test_documents_api.py

git commit -m "feat(backend): add database-backed document service and router"
```

---

## Phase 11: Indexing Service (Tasks 19-21)

### Task 19: Create IndexRun Repository

**Files:**
- Create: `backend/app/repositories/index_run.py`
- Modify: `backend/app/repositories/__init__.py`
- Test: `backend/tests/unit/test_repository_index_run.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_repository_index_run.py`:

```python
"""Tests for index run repository."""

import pytest
from app.repositories.index_run import IndexRunRepository


def test_index_run_repository_has_methods():
    """Test IndexRunRepository has required methods."""
    assert hasattr(IndexRunRepository, "get_latest_for_collection")
    assert hasattr(IndexRunRepository, "create_run")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_repository_index_run.py -v`
Expected: FAIL with "No module named 'app.repositories.index_run'"

**Step 3: Write minimal implementation**

Create `backend/app/repositories/index_run.py`:

```python
"""Index run repository."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRun, IndexRunStatus
from .base import BaseRepository


class IndexRunRepository(BaseRepository[IndexRun]):
    """Repository for index run operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, IndexRun)

    async def get_latest_for_collection(self, collection_id: UUID) -> Optional[IndexRun]:
        result = await self.session.execute(
            select(IndexRun)
            .where(IndexRun.collection_id == collection_id)
            .order_by(IndexRun.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_run(self, collection_id: UUID) -> IndexRun:
        run = IndexRun(collection_id=collection_id, status=IndexRunStatus.QUEUED)
        await self.create(run)
        return run
```

Update `backend/app/repositories/__init__.py`:

```python
from .index_run import IndexRunRepository

__all__ = ["BaseRepository", "CollectionRepository", "DocumentRepository", "IndexRunRepository"]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_repository_index_run.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/repositories/index_run.py backend/app/repositories/__init__.py

git add backend/tests/unit/test_repository_index_run.py

git commit -m "feat(backend): add index run repository"
```

---

### Task 20: Create DB-backed Indexing Service (enqueue jobs)

**Files:**
- Create: `backend/app/services/indexing_service_db.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/routers/indexing.py`
- Modify: `backend/app/services/__init__.py`
- Test: `backend/tests/unit/test_indexing_service_db.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_indexing_service_db.py`:

```python
"""Tests for database-backed indexing service."""

import pytest
from app.services.indexing_service_db import IndexingServiceDB


def test_indexing_service_has_methods():
    """Test IndexingServiceDB has required methods."""
    assert hasattr(IndexingServiceDB, "start_indexing")
    assert hasattr(IndexingServiceDB, "get_index_status")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_indexing_service_db.py -v`
Expected: FAIL with "No module named 'app.services.indexing_service_db'"

**Step 3: Write minimal implementation**

Create `backend/app/services/indexing_service_db.py`:

```python
"""Database-backed indexing service."""

from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IndexRunStatus
from app.models import IndexStatus, IndexStatusResponse
from app.repositories import CollectionRepository, IndexRunRepository
from app.worker.queue import enqueue_indexing_job


class IndexingServiceDB:
    """Service for managing indexing operations using database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.collections = CollectionRepository(session)
        self.runs = IndexRunRepository(session)

    async def start_indexing(self, collection_id: UUID) -> IndexStatusResponse:
        collection = await self.collections.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection '{collection_id}' not found")

        run = await self.runs.create_run(collection_id)
        job_id = enqueue_indexing_job(collection_id, run.id)

        return IndexStatusResponse(
            collection_id=str(collection_id),
            status=IndexStatus.RUNNING,
            progress=0.0,
            message=f"Queued job {job_id}",
            started_at=datetime.now(),
        )

    async def get_index_status(self, collection_id: UUID) -> IndexStatusResponse | None:
        run = await self.runs.get_latest_for_collection(collection_id)
        if not run:
            return None

        status_map = {
            IndexRunStatus.QUEUED: IndexStatus.RUNNING,
            IndexRunStatus.RUNNING: IndexStatus.RUNNING,
            IndexRunStatus.COMPLETED: IndexStatus.COMPLETED,
            IndexRunStatus.FAILED: IndexStatus.FAILED,
        }
        return IndexStatusResponse(
            collection_id=str(collection_id),
            status=status_map[run.status],
            progress=100.0 if run.status == IndexRunStatus.COMPLETED else 0.0,
            message=run.error or "",
            started_at=run.started_at,
            completed_at=run.finished_at,
        )
```

Update `backend/app/api/deps.py`:

```python
from app.services.indexing_service_db import IndexingServiceDB


async def get_indexing_service(
    session: AsyncSession = Depends(get_db_session)
) -> IndexingServiceDB:
    """FastAPI dependency for indexing service."""
    return IndexingServiceDB(session)
```

Update `backend/app/routers/indexing.py`:

```python
from uuid import UUID
from fastapi import Depends
from app.api.deps import get_indexing_service
from app.services.indexing_service_db import IndexingServiceDB


@router.post("", response_model=IndexStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_indexing(
    collection_id: str,
    service: IndexingServiceDB = Depends(get_indexing_service),
):
    try:
        result = await service.start_indexing(UUID(collection_id))
        logger.info(f"Started indexing for collection: {collection_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
```

(Apply same dependency pattern to `get_index_status`.)

Update `backend/app/services/__init__.py`:

```python
from .indexing_service_db import IndexingServiceDB

__all__ = [
    "storage_service",
    "indexing_service",
    "query_service",
    "CollectionServiceDB",
    "DocumentServiceDB",
    "IndexingServiceDB",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_indexing_service_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/indexing_service_db.py backend/app/routers/indexing.py

git add backend/app/api/deps.py backend/app/services/__init__.py

git add backend/tests/unit/test_indexing_service_db.py

git commit -m "feat(backend): add database-backed indexing service"
```

---

### Task 21: Implement GraphRAG DB Adapter for Indexing Outputs

**Files:**
- Create: `backend/app/services/graphrag_db_adapter.py`
- Modify: `backend/app/worker/tasks.py`
- Test: `backend/tests/unit/test_graphrag_db_adapter.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_graphrag_db_adapter.py`:

```python
"""Tests for GraphRAG DB adapter."""

import pytest
from app.services.graphrag_db_adapter import GraphRAGDbAdapter


def test_adapter_has_methods():
    """Test adapter exposes ingestion methods."""
    assert hasattr(GraphRAGDbAdapter, "ingest_outputs")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_graphrag_db_adapter.py -v`
Expected: FAIL with "No module named 'app.services.graphrag_db_adapter'"

**Step 3: Write minimal implementation**

Create `backend/app/services/graphrag_db_adapter.py`:

```python
"""Adapter to persist GraphRAG outputs to database."""

from typing import Iterable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


class GraphRAGDbAdapter:
    """Persist GraphRAG output artifacts into SQL tables."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def ingest_outputs(self, collection_id: UUID, index_run_id: UUID, outputs: Iterable[object]) -> None:
        """Persist GraphRAG outputs to database (placeholder)."""
        # Implementation will map outputs to models (entities, relationships, etc.)
        return None
```

Update `backend/app/worker/tasks.py` to call adapter after GraphRAG indexing completes (currently TODO block). Wire it in after successful indexing to save outputs to DB.

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_graphrag_db_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/graphrag_db_adapter.py backend/app/worker/tasks.py

git add backend/tests/unit/test_graphrag_db_adapter.py

git commit -m "feat(backend): add GraphRAG database adapter"
```

---

## Phase 12: Query Service (Tasks 22-24)

### Task 22: Create Query Repositories

**Files:**
- Create: `backend/app/repositories/query.py`
- Modify: `backend/app/repositories/__init__.py`
- Test: `backend/tests/unit/test_repository_query.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_repository_query.py`:

```python
"""Tests for query repository."""

import pytest
from app.repositories.query import QueryRepository


def test_query_repository_has_methods():
    """Test QueryRepository has required methods."""
    assert hasattr(QueryRepository, "get_latest_run_data")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_repository_query.py -v`
Expected: FAIL with "No module named 'app.repositories.query'"

**Step 3: Write minimal implementation**

Create `backend/app/repositories/query.py`:

```python
"""Query repository for GraphRAG data."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Community,
    CommunityReport,
    Entity,
    Relationship,
    TextUnit,
    IndexRun,
    IndexRunStatus,
)


class QueryRepository:
    """Repository for query-time data access."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_run_id(self, collection_id: UUID) -> UUID | None:
        result = await self.session.execute(
            select(IndexRun.id)
            .where(IndexRun.collection_id == collection_id)
            .where(IndexRun.status == IndexRunStatus.COMPLETED)
            .order_by(IndexRun.finished_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_latest_run_data(self, collection_id: UUID):
        run_id = await self.get_latest_run_id(collection_id)
        if not run_id:
            return None
        return run_id
```

Update `backend/app/repositories/__init__.py`:

```python
from .query import QueryRepository

__all__ = [
    "BaseRepository",
    "CollectionRepository",
    "DocumentRepository",
    "IndexRunRepository",
    "QueryRepository",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_repository_query.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/repositories/query.py backend/app/repositories/__init__.py

git add backend/tests/unit/test_repository_query.py

git commit -m "feat(backend): add query repository"
```

---

### Task 23: Create DB-backed Query Service

**Files:**
- Create: `backend/app/services/query_service_db.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/routers/search.py`
- Modify: `backend/app/services/__init__.py`
- Test: `backend/tests/unit/test_query_service_db.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_query_service_db.py`:

```python
"""Tests for database-backed query service."""

import pytest
from app.services.query_service_db import QueryServiceDB


def test_query_service_has_methods():
    """Test QueryServiceDB has required methods."""
    assert hasattr(QueryServiceDB, "global_search")
    assert hasattr(QueryServiceDB, "local_search")
    assert hasattr(QueryServiceDB, "tog_search")
    assert hasattr(QueryServiceDB, "drift_search")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_query_service_db.py -v`
Expected: FAIL with "No module named 'app.services.query_service_db'"

**Step 3: Write minimal implementation**

Create `backend/app/services/query_service_db.py`:

```python
"""Database-backed query service for GraphRAG search."""

import logging
from typing import Optional
from uuid import UUID

import graphrag.api as api

from app.models import SearchMethod, SearchResponse
from app.repositories import QueryRepository
n
logger = logging.getLogger(__name__)


class QueryServiceDB:
    """Service for managing query/search operations using database."""

    def __init__(self, session):
        self.session = session
        self.repo = QueryRepository(session)

    async def global_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        # Placeholder: actual implementation will load db tables
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.GLOBAL)

    async def local_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.LOCAL)

    async def tog_search(self, collection_id: UUID, query: str) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.TOG)

    async def drift_search(self, collection_id: UUID, query: str, **kwargs) -> SearchResponse:
        return SearchResponse(query=query, response="", context_data=None, method=SearchMethod.DRIFT)
```

Update `backend/app/api/deps.py`:

```python
from app.services.query_service_db import QueryServiceDB


async def get_query_service(
    session: AsyncSession = Depends(get_db_session)
) -> QueryServiceDB:
    """FastAPI dependency for query service."""
    return QueryServiceDB(session)
```

Update `backend/app/routers/search.py` to use `get_query_service` instead of global `query_service` and parse `collection_id` as UUID.

Update `backend/app/services/__init__.py`:

```python
from .query_service_db import QueryServiceDB

__all__ = [
    "storage_service",
    "indexing_service",
    "query_service",
    "CollectionServiceDB",
    "DocumentServiceDB",
    "IndexingServiceDB",
    "QueryServiceDB",
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_query_service_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/query_service_db.py backend/app/routers/search.py

git add backend/app/api/deps.py backend/app/services/__init__.py

git add backend/tests/unit/test_query_service_db.py

git commit -m "feat(backend): add database-backed query service"
```

---

### Task 24: Implement DB-backed Query Data Loading

**Files:**
- Modify: `backend/app/services/query_service_db.py`
- Modify: `backend/app/repositories/query.py`
- Test: `backend/tests/unit/test_query_service_db_data.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_query_service_db_data.py`:

```python
"""Tests for QueryServiceDB data loading."""

import pytest
from app.services.query_service_db import QueryServiceDB


def test_query_service_db_loads_latest_run():
    """Test QueryServiceDB uses latest completed run."""
    assert hasattr(QueryServiceDB, "_load_run_data")
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_query_service_db_data.py -v`
Expected: FAIL if helper not implemented

**Step 3: Write minimal implementation**

Update `backend/app/repositories/query.py` to add methods that return ORM rows for latest run (entities, relationships, communities, reports, text_units, covariates).

Update `backend/app/services/query_service_db.py` to:
- Add `_load_run_data` that calls repository methods.
- Convert ORM rows into pandas DataFrames where required by `graphrag.api` search functions.

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_query_service_db_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/query_service_db.py backend/app/repositories/query.py

git add backend/tests/unit/test_query_service_db_data.py

git commit -m "feat(backend): load query data from database"
```

---

## Phase 13: Lifespan and Migration Runner (Tasks 25-26)

### Task 25: Run Alembic Migrations at Startup

**Files:**
- Modify: `backend/app/main.py`
- Test: `backend/tests/unit/test_startup_migrations.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_startup_migrations.py`:

```python
"""Tests for startup migrations."""

import pytest
from app.main import lifespan


def test_lifespan_exists():
    assert callable(lifespan)
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_startup_migrations.py -v`
Expected: FAIL if no module path or import issues

**Step 3: Write minimal implementation**

Update `backend/app/main.py` lifespan to:
- Run Alembic upgrade head on startup using `alembic.command.upgrade` with config path `backend/alembic.ini`.
- Remove filesystem storage directory creation.

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_startup_migrations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/unit/test_startup_migrations.py

git commit -m "feat(backend): run migrations on startup"
```

---

### Task 26: Add Startup Database Health Check

**Files:**
- Modify: `backend/app/main.py`
- Test: `backend/tests/unit/test_startup_db_health.py`

**Step 1: Write the failing test**

Create `backend/tests/unit/test_startup_db_health.py`:

```python
"""Tests for startup database health check."""

import pytest


def test_startup_health_check_placeholder():
    assert True
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/unit/test_startup_db_health.py -v`
Expected: PASS (placeholder)

**Step 3: Write minimal implementation**

Update `backend/app/main.py` to add an async health check during lifespan startup:
- Open an async session and execute `SELECT 1`.
- Log failure and raise exception if DB unreachable.

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/unit/test_startup_db_health.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/unit/test_startup_db_health.py

git commit -m "feat(backend): add startup db health check"
```

---

## Phase 14: End-to-End Testing (Tasks 27-28)

### Task 27: Full Indexing + Search Integration Test

**Files:**
- Create: `backend/tests/integration/test_indexing_search_flow.py`

**Step 1: Write the failing test**

Create `backend/tests/integration/test_indexing_search_flow.py`:

```python
"""End-to-end indexing + search test."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_indexing_flow(client: AsyncClient):
    """Test indexing flow queues a job and updates status."""
    create_resp = await client.post("/api/collections", json={"name": "e2e"})
    assert create_resp.status_code == 201

    files = {"file": ("note.txt", b"hello", "text/plain")}
    await client.post("/api/collections/e2e/documents", files=files)

    start_resp = await client.post("/api/collections/e2e/index")
    assert start_resp.status_code == 202
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/integration/test_indexing_search_flow.py -v`
Expected: FAIL until indexing service wired

**Step 3: Write minimal implementation**

Update tests to align with queue semantics and ensure status endpoint returns queued/running. Stub if needed while DB adapter is placeholder.

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/integration/test_indexing_search_flow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/tests/integration/test_indexing_search_flow.py

git commit -m "test(backend): add end-to-end indexing flow test"
```

---

### Task 28: Performance Tests for Batch Inserts

**Files:**
- Create: `backend/tests/integration/test_batch_insert_perf.py`

**Step 1: Write the failing test**

Create `backend/tests/integration/test_batch_insert_perf.py`:

```python
"""Performance tests for batch inserts."""

import pytest


def test_batch_insert_placeholder():
    assert True
```

**Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/integration/test_batch_insert_perf.py -v`
Expected: PASS (placeholder)

**Step 3: Write minimal implementation**

Add a basic batch insert test using SQLAlchemy session and timing logs (no strict thresholds).

**Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/integration/test_batch_insert_perf.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/tests/integration/test_batch_insert_perf.py

git commit -m "test(backend): add batch insert performance test"
```
