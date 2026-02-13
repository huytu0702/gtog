# CosmosDB Emulator Phase 2 (Collection Domain Migration to Cosmos) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate collection metadata, documents, and prompts from filesystem-only backend storage to Cosmos-backed repositories while preserving existing API contracts.

**Architecture:** Introduce repository classes for collections/documents/prompts in backend service layer and a storage-mode-aware `StorageService` orchestrator that routes to file or Cosmos repository implementations. Keep routers and response schemas stable so frontend/API clients are unchanged.

**Tech Stack:** FastAPI, Pydantic, Azure Cosmos SDK (`azure-cosmos`), async file handling (`aiofiles`), pytest.

---

### Task 1: Create repository package and shared domain records

**Files:**
- Create: `backend/app/repositories/__init__.py`
- Create: `backend/app/repositories/types.py`
- Test: `backend/tests/unit/test_repository_types.py`

**Step 1: Write failing tests for domain records**

```python
from datetime import datetime
from backend.app.repositories.types import CollectionRecord, DocumentRecord, PromptRecord


def test_collection_record_fields():
    rec = CollectionRecord(id="demo", name="demo", description=None, created_at=datetime.utcnow())
    assert rec.id == "demo"


def test_document_record_fields():
    rec = DocumentRecord(collection_id="demo", name="a.txt", size=12, uploaded_at=datetime.utcnow())
    assert rec.name.endswith(".txt")


def test_prompt_record_fields():
    rec = PromptRecord(collection_id="demo", prompt_name="local_search_system_prompt.txt", content="text")
    assert rec.prompt_name.endswith(".txt")
```

**Step 2: Run tests to verify failure**

Run:

```bash
pytest backend/tests/unit/test_repository_types.py -v
```

Expected: FAIL (`ModuleNotFoundError`).

**Step 3: Add minimal shared types**

Implement dataclasses in `backend/app/repositories/types.py`:

```python
@dataclass
class CollectionRecord:
    id: str
    name: str
    description: str | None
    created_at: datetime

@dataclass
class DocumentRecord:
    collection_id: str
    name: str
    size: int
    uploaded_at: datetime

@dataclass
class PromptRecord:
    collection_id: str
    prompt_name: str
    content: str
```

Export from `backend/app/repositories/__init__.py`.

**Step 4: Run tests to verify pass**

Run:

```bash
pytest backend/tests/unit/test_repository_types.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/repositories/__init__.py backend/app/repositories/types.py backend/tests/unit/test_repository_types.py
git commit -m "feat: add shared repository domain record types"
```

---

### Task 2: Implement Cosmos collection repository

**Files:**
- Create: `backend/app/repositories/cosmos_collections.py`
- Test: `backend/tests/unit/test_cosmos_collections_repository.py`

**Step 1: Write failing tests for collection CRUD behavior**

```python
from backend.app.repositories.cosmos_collections import CosmosCollectionRepository


def test_create_collection_upserts_item(mock_cosmos_container):
    repo = CosmosCollectionRepository(mock_cosmos_container)
    rec = repo.create("demo", "desc")
    assert rec.id == "demo"


def test_get_collection_returns_none_when_missing(mock_cosmos_container):
    repo = CosmosCollectionRepository(mock_cosmos_container)
    assert repo.get("missing") is None
```

**Step 2: Run test to confirm fail**

Run:

```bash
pytest backend/tests/unit/test_cosmos_collections_repository.py -v
```

Expected: FAIL (module missing).

**Step 3: Implement minimal repository**

Repository methods:
- `create(collection_id, description) -> CollectionRecord`
- `list() -> list[CollectionRecord]`
- `get(collection_id) -> CollectionRecord | None`
- `delete(collection_id) -> bool`

Item shape (single container strategy):

```python
{
  "id": f"collection:{collection_id}",
  "kind": "collection",
  "collection_id": collection_id,
  "name": collection_id,
  "description": description,
  "created_at": created_iso,
}
```

Use partition key = `id` (consistent with existing cosmos storage implementation).

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_cosmos_collections_repository.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/repositories/cosmos_collections.py backend/tests/unit/test_cosmos_collections_repository.py
git commit -m "feat: add cosmos-backed collection repository"
```

---

### Task 3: Implement Cosmos document repository

**Files:**
- Create: `backend/app/repositories/cosmos_documents.py`
- Test: `backend/tests/unit/test_cosmos_documents_repository.py`

**Step 1: Write failing tests for upload/list/delete document metadata + payload**

```python
from backend.app.repositories.cosmos_documents import CosmosDocumentRepository


def test_put_document_stores_payload(mock_cosmos_container):
    repo = CosmosDocumentRepository(mock_cosmos_container)
    rec = repo.put("demo", "a.txt", b"hello")
    assert rec.size == 5


def test_list_documents_filters_collection(mock_cosmos_container):
    repo = CosmosDocumentRepository(mock_cosmos_container)
    repo.put("demo", "a.txt", b"hello")
    docs = repo.list("demo")
    assert len(docs) == 1
```

**Step 2: Run test to confirm fail**

Run:

```bash
pytest backend/tests/unit/test_cosmos_documents_repository.py -v
```

Expected: FAIL.

**Step 3: Implement repository with base64 payload encoding**

Cosmos item shape:

```python
{
  "id": f"document:{collection_id}:{filename}",
  "kind": "document",
  "collection_id": collection_id,
  "name": filename,
  "size": len(content),
  "uploaded_at": uploaded_iso,
  "content_b64": base64.b64encode(content).decode("ascii"),
}
```

Methods:
- `put(collection_id, filename, content_bytes) -> DocumentRecord`
- `list(collection_id) -> list[DocumentRecord]`
- `delete(collection_id, filename) -> bool`
- `get_content(collection_id, filename) -> bytes | None`

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_cosmos_documents_repository.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/repositories/cosmos_documents.py backend/tests/unit/test_cosmos_documents_repository.py
git commit -m "feat: add cosmos-backed document repository"
```

---

### Task 4: Implement Cosmos prompt repository + default prompt seed helper

**Files:**
- Create: `backend/app/repositories/cosmos_prompts.py`
- Create: `backend/app/repositories/default_prompts.py`
- Test: `backend/tests/unit/test_cosmos_prompts_repository.py`

**Step 1: Write failing tests for storing and fetching prompts**

```python
from backend.app.repositories.cosmos_prompts import CosmosPromptRepository


def test_set_and_get_prompt(mock_cosmos_container):
    repo = CosmosPromptRepository(mock_cosmos_container)
    repo.set_prompt("demo", "local_search_system_prompt.txt", "prompt-text")
    value = repo.get_prompt("demo", "local_search_system_prompt.txt")
    assert value == "prompt-text"


def test_seed_defaults_writes_required_prompt_keys(mock_cosmos_container):
    repo = CosmosPromptRepository(mock_cosmos_container)
    repo.seed_defaults("demo")
    names = repo.list_prompt_names("demo")
    assert "extract_graph.txt" in names
```

**Step 2: Run failing tests**

Run:

```bash
pytest backend/tests/unit/test_cosmos_prompts_repository.py -v
```

Expected: FAIL.

**Step 3: Implement repository + reusable default prompt map**

- Move existing prompt imports currently embedded in `StorageService.create_collection` into `default_prompts.py` function:

```python
def load_default_prompt_texts() -> dict[str, str]:
    ...
```

- Implement Cosmos prompt item shape:

```python
{
  "id": f"prompt:{collection_id}:{prompt_name}",
  "kind": "prompt",
  "collection_id": collection_id,
  "prompt_name": prompt_name,
  "content": text,
}
```

Methods:
- `set_prompt`, `get_prompt`, `list_prompt_names`, `seed_defaults`

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_cosmos_prompts_repository.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/repositories/cosmos_prompts.py backend/app/repositories/default_prompts.py backend/tests/unit/test_cosmos_prompts_repository.py
git commit -m "feat: add cosmos-backed prompt repository and default prompt seeding"
```

---

### Task 5: Add storage mode + Cosmos client settings in backend config

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/unit/test_config.py`

**Step 1: Write failing tests for new settings fields**

```python
from backend.app.config import Settings


def test_storage_mode_defaults_to_file():
    s = Settings()
    assert s.storage_mode == "file"


def test_cosmos_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081")
    monkeypatch.setenv("COSMOS_KEY", "abc")
    s = Settings()
    assert s.cosmos_endpoint.startswith("https://")
```

**Step 2: Run test to see failure**

Run:

```bash
pytest backend/tests/unit/test_config.py -v
```

Expected: FAIL for missing fields.

**Step 3: Implement config additions**

Add fields to `Settings`:

```python
storage_mode: str = "file"
cosmos_endpoint: str = ""
cosmos_key: str = ""
cosmos_database: str = ""
cosmos_container: str = ""
```

Optional helper:

```python
@property
def is_cosmos_mode(self) -> bool:
    return self.storage_mode.strip().lower() == "cosmos"
```

**Step 4: Run tests**

Run:

```bash
pytest backend/tests/unit/test_config.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/unit/test_config.py
git commit -m "feat: add cosmos connection settings and storage mode config"
```

---

### Task 6: Refactor StorageService to use repositories (file + cosmos paths)

**Files:**
- Modify: `backend/app/services/storage_service.py`
- Modify: `backend/app/services/__init__.py` (if exports change)
- Test: `backend/tests/unit/test_storage_service_modes.py`

**Step 1: Write failing tests for mode-specific behavior**

```python
from backend.app.services.storage_service import StorageService


def test_create_collection_cosmos_mode_does_not_create_local_dirs(mock_settings_cosmos):
    svc = StorageService()
    result = svc.create_collection("demo")
    assert result.id == "demo"


def test_upload_and_list_documents_cosmos_mode(mock_settings_cosmos, upload_file_factory):
    svc = StorageService()
    svc.create_collection("demo")
    doc = svc.upload_document_sync_for_test("demo", "a.txt", b"hello")
    docs = svc.list_documents("demo")
    assert len(docs) == 1
```

(Use async wrapper pattern compatible with current upload signature in real tests.)

**Step 2: Run tests and confirm fail**

Run:

```bash
pytest backend/tests/unit/test_storage_service_modes.py -v
```

Expected: FAIL.

**Step 3: Implement minimal refactor**

In `StorageService`:
- Keep current file behavior for `storage_mode=file`.
- Add Cosmos path for:
  - `create_collection`, `delete_collection`, `list_collections`, `get_collection`
  - `upload_document`, `list_documents`, `delete_document`
- Ensure prompt seeding in both modes:
  - file mode: write prompt files to local `prompts/`
  - cosmos mode: use `CosmosPromptRepository.seed_defaults`

Add internal composition:

```python
self._collection_repo
self._document_repo
self._prompt_repo
```

Instantiate Cosmos client once in service init when in cosmos mode.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_storage_service_modes.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/services/storage_service.py backend/tests/unit/test_storage_service_modes.py backend/app/services/__init__.py
git commit -m "refactor: route storage service through file or cosmos repositories"
```

---

### Task 7: Keep indexing/search compatibility with Cosmos-backed documents

**Files:**
- Modify: `backend/app/services/indexing_service.py`
- Modify: `backend/app/utils/helpers.py`
- Test: `backend/tests/unit/test_indexing_service_cosmos_input_sync.py`

**Step 1: Write failing tests for cosmos document materialization to GraphRAG input**

```python
from backend.app.services.indexing_service import IndexingService


def test_indexing_service_materializes_cosmos_documents_to_input_dir(mock_cosmos_mode, mock_docs_repo):
    svc = IndexingService()
    svc._prepare_input_files_for_indexing("demo")
    # assert local input files now exist for GraphRAG input scanning
```

**Step 2: Run tests to confirm failure**

Run:

```bash
pytest backend/tests/unit/test_indexing_service_cosmos_input_sync.py -v
```

Expected: FAIL.

**Step 3: Implement minimal compatibility layer**

Because GraphRAG indexing still reads `input.storage.type=file` in current backend flow for collection docs:
- Add pre-index step in `IndexingService` for cosmos mode:
  - fetch all docs from cosmos repository
  - write bytes into `collections/{id}/input/*.txt|*.md`
- Keep operation idempotent (overwrite local mirror each run).

Do **not** change router contracts.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_indexing_service_cosmos_input_sync.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/services/indexing_service.py backend/tests/unit/test_indexing_service_cosmos_input_sync.py backend/app/utils/helpers.py
git commit -m "feat: sync cosmos documents into indexing input workspace"
```

---

### Task 8: Add API-level integration test for collection/document lifecycle in Cosmos mode

**Files:**
- Create: `backend/tests/integration/test_collections_documents_cosmos.py`
- Test: integration pytest command

**Step 1: Write integration test (red phase)**

Test flow:
1. Set `STORAGE_MODE=cosmos`
2. POST collection
3. POST upload document
4. GET documents
5. DELETE document
6. DELETE collection

Assertions: response codes and payload contracts unchanged from existing API models.

**Step 2: Run test and confirm fail**

Run:

```bash
pytest backend/tests/integration/test_collections_documents_cosmos.py -v
```

Expected: FAIL initially.

**Step 3: Fix minimal gaps**

Adjust service/repositories only if this integration test reveals API mismatch.

**Step 4: Re-run integration test**

Run:

```bash
pytest backend/tests/integration/test_collections_documents_cosmos.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/integration/test_collections_documents_cosmos.py backend/app/services/storage_service.py
git commit -m "test: add cosmos integration coverage for collection and document lifecycle"
```

---

## Phase 2 Completion Checklist

- [ ] Cosmos repositories exist for collections/documents/prompts.
- [ ] StorageService supports `file` and `cosmos` modes with same API contracts.
- [ ] Default prompts are persisted in Cosmos for cosmos mode collections.
- [ ] Indexing service can consume cosmos-mode documents via local input materialization.
- [ ] Integration test covers collection/document CRUD in cosmos mode.
- [ ] Existing file-mode behavior remains unchanged.

---

## Notes for Executor

- Use @superpowers:test-driven-development for each repository/service change.
- Avoid introducing new API fields; keep request/response schemas unchanged.
- Keep repository abstractions simple (YAGNI): no generic ORM layer.
- Prefer one container strategy with `kind` discriminator in this phase unless test evidence requires split containers.
