 Fix Backend Indexing Config Implementation Plan                                                                                    ││                                                                                                                                    │
│ For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.                               │
│                                                                                                                                    │
│ Goal: Ensure backend indexing passes a valid GraphRagConfig to graphrag.api.build_index so indexing succeeds.                      │
│                                                                                                                                    │
│ Architecture: Use the existing backend helper load_graphrag_config to construct a GraphRagConfig from backend/settings.yaml with   │
│ collection-specific storage overrides. Keep the worker responsible for assembling input documents and invoking build_index with    │
│ the config and optional in-memory documents.                                                                                       │
│                                                                                                                                    │
│ Tech Stack: FastAPI backend, GraphRAG Python API, SQLAlchemy async, RQ worker.                                                     │
│                                                                                                                                    │
│ ---                                                                                                                                │
│ Task 1: Update worker to pass GraphRagConfig                                                                                       │
│                                                                                                                                    │
│ Files:                                                                                                                             │
│ - Modify: backend/app/worker/tasks.py:89-118                                                                                       │
│ - Reference: backend/app/utils/helpers.py:13-41                                                                                    │
│                                                                                                                                    │
│ Step 1: Write the failing test                                                                                                     │
│                                                                                                                                    │
│ Create a unit test that asserts build_index is called with a GraphRagConfig instance (not a string) when _run_graphrag_pipeline is │
│  invoked.                                                                                                                          │
│                                                                                                                                    │
│ # tests/unit/test_worker_tasks.py                                                                                                  │
│ import pytest                                                                                                                      │
│ from unittest.mock import AsyncMock                                                                                                │
│ from uuid import UUID                                                                                                              │
│                                                                                                                                    │
│ from graphrag.config.models.graph_rag_config import GraphRagConfig                                                                 │
│ import app.worker.tasks as tasks                                                                                                   │
│                                                                                                                                    │
│ @pytest.mark.asyncio                                                                                                               │
│ async def test_run_graphrag_pipeline_uses_graphrag_config(monkeypatch, db_session, collection):                                    │
│     async def fake_build_index(*, config, **_kwargs):                                                                              │
│         assert isinstance(config, GraphRagConfig)                                                                                  │
│         return []                                                                                                                  │
│                                                                                                                                    │
│     monkeypatch.setattr("app.worker.tasks.api.build_index", AsyncMock(side_effect=fake_build_index))                               │
│                                                                                                                                    │
│     await tasks._run_graphrag_pipeline(db_session, UUID(str(collection.id)), UUID(str(collection.id)))                             │
│                                                                                                                                    │
│ Step 2: Run test to verify it fails                                                                                                │
│                                                                                                                                    │
│ Run: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_uses_graphrag_config -v                            │
│ Expected: FAIL because build_index is called with a string path (current behavior).                                                │
│                                                                                                                                    │
│ Step 3: Write minimal implementation                                                                                               │
│                                                                                                                                    │
│ Update _run_graphrag_pipeline to load config and pass it explicitly.                                                               │
│                                                                                                                                    │
│ # backend/app/worker/tasks.py                                                                                                      │
│ from app.utils.helpers import load_graphrag_config                                                                                 │
│                                                                                                                                    │
│ # inside _run_graphrag_pipeline                                                                                                    │
│ config = load_graphrag_config(str(collection_id))                                                                                  │
│                                                                                                                                    │
│ return await api.build_index(                                                                                                      │
│     config=config,                                                                                                                 │
│     # Keep current file-based input by ensuring config.input.storage.base_dir                                                      │
│ )                                                                                                                                  │
│                                                                                                                                    │
│ Step 4: Run test to verify it passes                                                                                               │
│                                                                                                                                    │
│ Run: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_uses_graphrag_config -v                            │
│ Expected: PASS.                                                                                                                    │
│                                                                                                                                    │
│ Step 5: Commit                                                                                                                     │
│                                                                                                                                    │
│ git add backend/app/worker/tasks.py backend/tests/unit/test_worker_tasks.py                                                        │
│ git commit -m "fix(backend): pass GraphRagConfig to indexing"                                                                      │
│                                                                                                                                    │
│ Task 2: Ensure input documents are correctly wired (optional refinement)                                                           │
│                                                                                                                                    │
│ Files:                                                                                                                             │
│ - Modify: backend/app/worker/tasks.py:101-118                                                                                      │
│                                                                                                                                    │
│ Step 1: Write the failing test                                                                                                     │
│                                                                                                                                    │
│ Add a test to ensure input_documents are passed when documents exist.                                                              │
│                                                                                                                                    │
│ # backend/tests/unit/test_worker_tasks.py                                                                                          │
│ @pytest.mark.asyncio                                                                                                               │
│ async def test_run_graphrag_pipeline_passes_input_documents(monkeypatch, db_session, collection, document):                        │
│     async def fake_build_index(*, input_documents, **_kwargs):                                                                     │
│         assert input_documents is not None                                                                                         │
│         assert "text" in input_documents.columns                                                                                   │
│         return []                                                                                                                  │
│                                                                                                                                    │
│     monkeypatch.setattr("app.worker.tasks.api.build_index", AsyncMock(side_effect=fake_build_index))                               │
│                                                                                                                                    │
│     await tasks._run_graphrag_pipeline(db_session, UUID(str(collection.id)), UUID(str(collection.id)))                             │
│                                                                                                                                    │
│ Step 2: Run test to verify it fails                                                                                                │
│                                                                                                                                    │
│ Run: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_passes_input_documents -v                          │
│ Expected: FAIL because input_documents are not passed.                                                                             │
│                                                                                                                                    │
│ Step 3: Write minimal implementation                                                                                               │
│                                                                                                                                    │
│ Construct a small DataFrame with id, title, and text from Document.bytes_content and pass it to build_index.                       │
│                                                                                                                                    │
│ import pandas as pd                                                                                                                │
│                                                                                                                                    │
│ rows = []                                                                                                                          │
│ for doc in documents:                                                                                                              │
│     rows.append({                                                                                                                  │
│         "id": str(doc.id),                                                                                                         │
│         "title": doc.filename or str(doc.id),                                                                                      │
│         "text": (doc.bytes_content or b"").decode("utf-8", errors="replace"),                                                      │
│     })                                                                                                                             │
│ input_documents = pd.DataFrame(rows)                                                                                               │
│                                                                                                                                    │
│ return await api.build_index(config=config, input_documents=input_documents)                                                       │
│                                                                                                                                    │
│ Step 4: Run test to verify it passes                                                                                               │
│                                                                                                                                    │
│ Run: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_passes_input_documents -v                          │
│ Expected: PASS.                                                                                                                    │
│                                                                                                                                    │
│ Step 5: Commit                                                                                                                     │
│                                                                                                                                    │
│ git add backend/app/worker/tasks.py backend/tests/unit/test_worker_tasks.py                                                        │
│ git commit -m "fix(backend): pass in-memory docs to GraphRAG"                                                                      │
│                                                                                                                                    │
│ Task 3: Verify end-to-end indexing in Docker                                                                                       │
│                                                                                                                                    │
│ Files:                                                                                                                             │
│ - No code changes                                                                                                                  │
│                                                                                                                                    │
│ Step 1: Run integration test (optional)                                                                                            │
│                                                                                                                                    │
│ Run: pytest backend/tests/integration/test_indexing_search_flow.py -v                                                              │
│ Expected: PASS (job enqueued). Note: may require external services.                                                                │
│                                                                                                                                    │
│ Step 2: Manual verification                                                                                                        │
│                                                                                                                                    │
│ - Start Docker: docker compose up -d                                                                                               │
│ - Upload document and trigger indexing via UI or API.                                                                              │
│ - Check logs: docker logs graphrag-worker should not show AttributeError: 'str' object has no attribute 'reporting'.               │
│                                                                                                                                    │
│ ---                                                                                                                                │
│ Verification                                                                                                                       │
│                                                                                                                                    │
│ - Unit: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_uses_graphrag_config -v                         │
│ - Optional unit: pytest backend/tests/unit/test_worker_tasks.py::test_run_graphrag_pipeline_passes_input_documents -v              │
│ - Optional integration: pytest backend/tests/integration/test_indexing_search_flow.py -v                                           │
│ - Manual: docker logs graphrag-worker after indexing