# Azurite Migration Plan: Remove local_file Mode

## Tóm tắt yêu cầu

1. **Cài Azurite** (Docker) làm emulator local cho Azure Blob Storage và Azure Storage Queue.
2. **Xoá `local_file` mode** — chỉ còn một profile `cosmos_pipeline` chạy thống nhất.
3. **Logs indexing** → Azure Blob Storage (Azurite local, Azure thật trên cloud).
4. **Cosmos DB** → đang dùng Cosmos DB Emulator (giữ nguyên, ngoài scope).

---

## Quyết định đã xác nhận

| Câu hỏi | Quyết định |
|---------|-----------|
| Cách cài Azurite | **Docker** |
| `INDEX_OUTPUT_MODE` config | **Xoá hẳn** |
| Cosmos local | **Cosmos DB Emulator** (đang chạy) |
| `STORAGE_ROOT_DIR` + `backend/storage/` | **Xoá luôn** |
| Lưu indexing logs | **Azure Blob Storage** (`reporting.type: blob`) |

---

## Hiện trạng

- `backend/settings.yaml`: `output: cosmosdb`, `vector_store: cosmosdb`, `input.storage: file`, `reporting: file`
- `backend/app/config.py:92`: `index_output_mode`, `query_context_mode`, `storage_root_dir` fields còn tồn tại
- `backend/app/utils/helpers.py`: `load_graphrag_config` có 2 nhánh `cosmos_pipeline` / `local_file`
- `backend/app/utils/helpers.py`: `validate_collection_indexed` có 2 nhánh
- `backend/app/utils/helpers.py`: `get_search_data_paths` — chỉ dùng cho `local_file`, sẽ xoá
- `.gitignore`: đã có entry cho Azurite (`temp_azurite/`, `__azurite*.json`, v.v.)

---

## Tasks

### Task 1 — Phase 1: Setup Azurite Docker scripts
**Status:** pending

Tạo:
- `scripts/azurite/docker-compose.azurite.yml`
- `scripts/azurite/start-azurite.ps1`
- `scripts/azurite/start-azurite.sh`
- Thêm `.azurite/` vào `.gitignore`

Azurite well-known connection string:
```
DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=<AZURITE_DEFAULT_ACCOUNT_KEY>;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;
```

---

### Task 2 — Phase 2: Update settings.yaml + .env.example
**Status:** pending

**`backend/settings.yaml`:**
- `input.storage.type: file` → `blob`
- `reporting.type: file` → `blob`
- Thêm `reporting.container_name: pipeline-logs`
- Xoá comment `# or blob`

**`backend/.env.example`:**
- Thêm block `# === LOCAL DEV (Azurite) ===` với connection string Azurite
- Xoá `INDEX_OUTPUT_MODE`, `QUERY_CONTEXT_MODE`, `STORAGE_ROOT_DIR`

---

### Task 3 — Phase 3: Remove local_file mode from config.py
**Status:** pending

**`backend/app/config.py`:**
- Xoá fields: `storage_root_dir`, `index_output_mode`, `query_context_mode`
- Xoá property `collections_dir`
- Xoá `serving_dataset_cache_max_entries`, `serving_cache_warm_on_index_complete` nếu chỉ dùng cho local_file (kiểm tra trước)

---

### Task 4 — Phase 4: Remove local_file mode from helpers.py
**Status:** pending

**`backend/app/utils/helpers.py`:**
- `load_graphrag_config`: xoá nhánh `local_file` (~lines 384-389, 422-435), xoá `mode` variable, hard-code `cosmos_pipeline` flow
- `validate_collection_indexed`: xoá nhánh `else` đọc parquet local (~lines 508-516)
- `get_search_data_paths`: **xoá toàn bộ hàm**
- `_input_storage_cli_overrides`: xoá nhánh `if not use_blob_input` (local file path), chỉ giữ blob path
- Cập nhật `use_blob_input` — luôn `True` khi blob client available

**`backend/app/utils/__init__.py`:**
- Xoá export `get_search_data_paths` nếu có

---

### Task 5 — Phase 5: Clean local_file refs in services + main.py
**Status:** pending

Files cần kiểm tra và sửa:
- `backend/app/services/indexing_service.py`
- `backend/app/services/query_service.py`
- `backend/app/services/storage_service.py`
- `backend/app/services/serving_materialization_service.py`
- `backend/app/main.py`

Với mỗi file: xoá `if settings.index_output_mode == "local_file"` branches, xoá `settings.collections_dir` usages.

---

### Task 6 — Phase 6: Update tests
**Status:** pending

Files cần sửa:
- `backend/tests/unit/test_config_compatibility.py`
- `backend/tests/unit/test_helpers_runtime_config.py`
- `backend/tests/conftest.py`
- `backend/tests/integration/test_cosmos_pipeline_emulator.py`

Actions:
- Xoá test cases cho `local_file` mode
- Cập nhật fixtures dùng Azurite connection string
- Đảm bảo integration tests point vào Azurite

---

### Task 7 — Phase 7: Documentation + verification
**Status:** pending

- Update `CLAUDE.md`: thêm step `docker compose -f scripts/azurite/docker-compose.azurite.yml up -d` trước `uvicorn`
- Update `docs/architecture.md`: xoá references `local_file`
- Xoá `backend/storage/` directory (sau khi verify không còn dùng)
- Xoá `backend/logs/` directory (logs giờ vào blob)
- Smoke test: upload file → trigger index → check blob container `pipeline-logs`

---

## Logs Architecture (sau migration)

```
App runtime logs  →  stdout/stderr  →  ACA Log Analytics (prod) / terminal (local)
GraphRAG indexing →  Azure Blob     →  container: pipeline-logs/{collection_id}/{version}/
Job events        →  Cosmos DB      →  container: jobEvents (đã có)
```

---

## Risks

| Risk | Level | Mitigation |
|------|-------|-----------|
| Xoá `local_file` break dev không có Azure | HIGH | Azurite thay thế hoàn toàn |
| Azurite dùng HTTP (không HTTPS) | MEDIUM | Connection string có `DefaultEndpointsProtocol=http` |
| Tests dùng filesystem fixtures | MEDIUM | Migrate sang Azurite hoặc mock |
| Port conflict 10000/10001/10002 | LOW | Document trong README |

---

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1 — Azurite Docker scripts | ~30min |
| Phase 2 — settings.yaml + .env.example | ~30min |
| Phase 3 — config.py cleanup | ~1h |
| Phase 4 — helpers.py cleanup | ~2h |
| Phase 5 — services + main.py | ~2h |
| Phase 6 — tests | ~1.5h |
| Phase 7 — docs + verification | ~1h |
| **Total** | **~8.5h** |
