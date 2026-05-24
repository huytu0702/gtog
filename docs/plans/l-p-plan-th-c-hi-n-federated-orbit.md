# Plan: Sửa delete collection để dọn Cosmos pipeline containers và vectors

## Context

Hiện tại khi xóa collection, backend chỉ xóa control-plane metadata, blob container `col-{collection_id}`, Azure AI Search indexes, conversations và cache. Với cấu hình runtime đang dùng Cosmos, dữ liệu liên quan vẫn còn sót ở:

- Cosmos pipeline output containers dạng `pipeline-{collection_id}-{version}`.
- Shared Cosmos vector container `vectors`, trong đó mỗi item có `collectionId`, `version`, `collectionVersion`, `embeddingKind`, `partitionKey`.

Vấn đề lớn nhất là `StorageService.delete_collection()` đang gọi `control_plane.delete_collection()` trước, làm mất `activeVersion`, `artifact_manifest.version`, và `indexing_jobs.targetVersion` trước khi có cơ hội xác định container/version cần xóa. Mục tiêu là dọn đủ artifact của collection nhưng không xóa nhầm tài nguyên dùng chung.

## Recommended approach

### 1. Thêm discovery version trước khi xóa metadata

Sửa `backend/app/repositories/control_plane_repository.py`:

- Thêm method typed:
  - `list_collection_versions(self, collection_id: str) -> list[str]`
- Method này sẽ:
  - Gọi `get_collection(collection_id)` để validate collection tồn tại; nếu không có thì raise `ValueError` như `delete_collection()` hiện tại.
  - Thu thập version từ:
    - `collections.activeVersion`
    - `artifact_manifest.version`
    - `indexing_jobs.targetVersion`
  - Bỏ giá trị rỗng, deduplicate, trả về sorted list ổn định.

Lý do lấy thêm `indexing_jobs.targetVersion`: job failed/partial có thể đã tạo pipeline/vector artifact trước khi ghi đủ `artifact_manifest`.

### 2. Thêm API xóa Cosmos pipeline output containers

Sửa `backend/app/repositories/pipeline_output_repository.py`:

- Reuse function hiện có:
  - `build_pipeline_container_name(collection_id, version)`
- Thêm method:
  - `delete_collection_outputs(self, *, collection_id: str, versions: list[str]) -> int`
- Method này sẽ:
  - Dùng cấu hình Cosmos hiện có từ repository/runtime helpers (`resolve_cosmos_connection_string`, `cosmos_account_url`, `cosmos_client_kwargs`, `settings.azure_cosmos_database_name`).
  - Tạo Cosmos client theo pattern sẵn có trong `_storage_for()` / indexing service.
  - Với mỗi version unique/non-empty, build container name và gọi delete container trên database.
  - Ignore `CosmosResourceNotFoundError` nếu container đã không tồn tại.
  - Return số container đã xóa.

Không dùng `CosmosDBPipelineStorage.clear()` vì implementation hiện tại có thể xóa cả database.

### 3. Thêm helper xóa vector documents theo collection

Sửa `backend/app/vector_stores/scoped_cosmosdb.py`:

- File này đang sở hữu schema/constants của shared vector container:
  - `_FIXED_VECTOR_CONTAINER_NAME = "vectors"`
  - `_COLLECTION_ID_FIELD = "collectionId"`
  - `_PARTITION_KEY_FIELD = "partitionKey"`
- Thêm function module-level:
  - `delete_collection_vector_documents(collection_id: str) -> int`
- Function này sẽ:
  - Kết nối Cosmos bằng cùng pattern auth hiện có.
  - Mở database `settings.azure_cosmos_database_name` và container `vectors`.
  - Query cross-partition:
    - `SELECT c.id, c.partitionKey FROM c WHERE c.collectionId = @collectionId`
  - Xóa từng item bằng chính `id` và `partitionKey` lưu trên item.
  - Return số document đã xóa.
  - Nếu database/container không tồn tại thì return `0`.

Không xóa container `vectors` vì đây là container dùng chung cho mọi collection.

### 4. Reorder `StorageService.delete_collection()`

Sửa `backend/app/services/storage_service.py`:

- Import thêm:
  - `logging`
  - `get_pipeline_output_repository`
  - `delete_collection_vector_documents`
- Thêm logger module-level.
- Đổi thứ tự delete thành:
  1. `control_plane = self._require_control_plane()`
  2. `blob_client = self._require_blob_client()`
  3. `versions = control_plane.list_collection_versions(collection_id)`
  4. `get_pipeline_output_repository().delete_collection_outputs(collection_id=collection_id, versions=versions)`
  5. `delete_collection_vector_documents(collection_id)`
  6. Xóa blob container `col-{collection_id}` nếu tồn tại.
  7. Best-effort xóa Azure Search indexes bằng `delete_search_indexes_for_collection(collection_id)`.
  8. Best-effort purge conversations.
  9. Best-effort invalidate query cache.
  10. Gọi `control_plane.delete_collection(collection_id)` cuối cùng.

Cleanup Cosmos pipeline/vector và blob nên là mandatory trước metadata deletion. Nếu các bước này fail, không xóa control-plane metadata để tránh orphan artifact không còn metadata discovery path. Search/conversation/cache giữ best-effort như hành vi hiện tại, nhưng thay `except Exception: pass` bằng logging để dễ debug.

## Critical files to modify

- `backend/app/services/storage_service.py`
- `backend/app/repositories/control_plane_repository.py`
- `backend/app/repositories/pipeline_output_repository.py`
- `backend/app/vector_stores/scoped_cosmosdb.py`
- Tests:
  - `backend/tests/unit/repositories/test_pipeline_output_repository.py`
  - `backend/tests/unit/vector_stores/test_scoped_cosmosdb.py`
  - `backend/tests/unit/services/test_storage_service.py` nếu chưa có thì tạo mới
  - repository test hiện có hoặc test mới cho `list_collection_versions()`

## Tests to add/update

### Control-plane version discovery

Test `list_collection_versions()`:

- Include `activeVersion`.
- Include `artifact_manifest.version`.
- Include `indexing_jobs.targetVersion`.
- Deduplicate và ignore blank values.
- Raise `ValueError` khi collection không tồn tại.

### Pipeline output cleanup

Test `PipelineOutputRepository.delete_collection_outputs()`:

- Delete đúng container names từ `build_pipeline_container_name()`.
- Deduplicate versions.
- Ignore blank versions.
- Ignore missing containers (`CosmosResourceNotFoundError`).
- Không gọi `CosmosDBPipelineStorage.clear()`.

### Vector cleanup

Test `delete_collection_vector_documents()`:

- Query bằng `collectionId` cross-partition.
- Delete từng item với returned `partitionKey`.
- Return đúng deleted count.
- Return `0` nếu database/container missing.
- Assert không gọi delete container cho `vectors`.

### StorageService delete ordering

Test `StorageService.delete_collection()`:

- Assert thứ tự quan trọng:
  1. `list_collection_versions()`
  2. pipeline cleanup
  3. vector cleanup
  4. blob cleanup
  5. `control_plane.delete_collection()` cuối cùng
- Nếu pipeline cleanup fail thì không gọi `control_plane.delete_collection()`.
- Nếu vector cleanup fail thì không gọi `control_plane.delete_collection()`.
- Nếu `delete_search_indexes_for_collection()` fail thì vẫn tiếp tục xóa metadata, vì đây là best-effort hiện có.

## Verification

Chạy targeted tests trước:

```powershell
pytest backend/tests/unit/repositories/test_pipeline_output_repository.py
pytest backend/tests/unit/vector_stores/test_scoped_cosmosdb.py
pytest backend/tests/unit/services/test_storage_service.py
```

Sau đó chạy broader backend unit tests:

```powershell
pytest backend/tests/unit
```

Nếu Cosmos emulator đang khả dụng, có thể chạy thêm integration liên quan pipeline Cosmos:

```powershell
pytest backend/tests/integration/test_cosmos_pipeline_emulator.py
```

Manual verification sau khi implement:

1. Tạo/index một collection test.
2. Xác nhận có blob container `col-{collection_id}`.
3. Xác nhận có Cosmos pipeline container `pipeline-{collection_id}-{version}`.
4. Xác nhận shared `vectors` có documents với `collectionId = collection_id`.
5. Gọi DELETE collection.
6. Xác nhận:
   - blob container bị xóa.
   - pipeline container theo version bị xóa.
   - vector documents của collection bị xóa.
   - container `vectors` vẫn còn.
   - vector documents của collection khác vẫn còn.
