# Cosmos Direct Indexing and Direct Query Plan

## Mục tiêu
Chuyển backend từ luồng hiện tại:

- GraphRAG output -> Blob/File parquet
- backend đọc parquet -> materialize sang Cosmos serving containers
- backend query đọc từ `serving_repository`

sang luồng mới:

- **runtime chính / cloud / worker**: GraphRAG output -> Cosmos DB trực tiếp
- backend query đọc **trực tiếp** từ Cosmos pipeline storage
- **local/dev execution**: vẫn giữ nhánh `output.type=file` để chạy local không phụ thuộc Azure Cosmos

Đồng thời:
- **loại bỏ hoàn toàn nhánh Blob output cũ**
- **loại bỏ production flow materialize sang serving Cosmos containers**
- **vẫn giữ nhánh File output dành riêng cho local execution**

---

## Kết luận sau khi xem `graphrag/query`

### GraphRAG gốc query như thế nào
Sau khi rà `graphrag/query` và `graphrag/api/query.py`, điểm quan trọng là query layer của GraphRAG **không gắn với file path hay serving repository**. Nó nhận `pandas.DataFrame` đầu vào rồi tự adapter sang object model nội bộ.

### Entry points chính
- `graphrag/api/query.py`:
  - `global_search(...)`
  - `local_search(...)`
  - `drift_search(...)`
  - `basic_search(...)`
  - `tog_search(...)`
- `graphrag/query/indexer_adapters.py`:
  - chuyển `DataFrame` -> `Entity`, `Relationship`, `TextUnit`, `Community`, `CommunityReport`, `Covariate`
- `graphrag/query/factory.py`:
  - build search engine cho từng mode

### Hệ quả cho backend
Backend **không cần serving materialization** nếu nó có thể cấp đúng `DataFrame` cho các API query của GraphRAG.

Nói cách khác, thay đổi cần làm nằm ở **loader boundary** của backend:
- hiện tại loader đọc từ `serving_repository` hoặc local parquet
- sau khi refactor, loader sẽ đọc từ:
  - `CosmosDBPipelineStorage` trong runtime chính
  - local parquet trong local mode

### Dataset contract cần giữ nguyên
Dựa trên `graphrag/api/query.py` và backend hiện tại, các mode query cần:

- `global`: `entities`, `communities`, `community_reports`
- `local`: `entities`, `communities`, `community_reports`, `text_units`, `relationships`, `covariates` nếu có
- `drift`: `entities`, `communities`, `community_reports`, `text_units`, `relationships`
- `tog`: `entities`, `relationships`, `text_units`

### Backend hiện đang phụ thuộc gì
- `backend/app/services/query_service.py` hiện load context từ `serving_repository`
- fallback local thì đọc `output/*.parquet`
- `backend/app/services/query_service_base.py` có normalize `community_reports.full_content`

### Quyết định mới
- **không đổi GraphRAG query layer**
- **đổi backend loader/query service** để cấp `DataFrame` từ pipeline storage
- **bỏ production serving repository/query path**

---

## Quyết định kiến trúc

### Giữ
- Control-plane trong Cosmos DB (`collections`, `indexingJobs`, `artifactManifest`, ...)
- `activeVersion` là contract chính để publish version query được
- nhánh local `output.type=file` cho dev/local execution
- GraphRAG query APIs và `indexer_adapters` nguyên bản

### Bỏ khỏi runtime chính
- `output.type=blob`
- đọc Blob parquet để materialize
- production flow `serving_materialization_service`
- production query path đọc từ `serving_repository`
- mọi validation/check phụ thuộc Blob artifacts hoặc serving datasets materialized

### Luồng đích

#### Cloud / worker / production-like runtime
1. backend load GraphRAG config với `output.type=cosmosdb`
2. `api.build_index(...)` ghi trực tiếp datasets vào Cosmos pipeline storage
3. worker verify required datasets đã hiện diện trong pipeline storage
4. worker set `activeVersion`
5. query service load datasets trực tiếp từ pipeline Cosmos theo `collectionId + activeVersion`
6. query service gọi `graphrag.api.global_search/local_search/drift_search/tog_search`

#### Local execution
1. backend load GraphRAG config với `output.type=file`
2. `api.build_index(...)` ghi parquet vào local `output/`
3. local query path đọc local parquet
4. local query service vẫn gọi cùng GraphRAG APIs như production

---

## Lý do chọn kiến trúc này
- Gần với cách GraphRAG gốc tổ chức query hơn
- Bỏ một tầng duplicate data model không cần thiết
- Giảm write amplification: không còn Cosmos pipeline -> Cosmos serving copy
- Giữ local dev loop đơn giản nhờ file output branch
- Tập trung refactor đúng chỗ: backend loader/query layer, không đụng sâu GraphRAG internals

## Phạm vi thay đổi

### Trong scope
- đổi output storage của indexing pipeline sang Cosmos DB cho runtime chính
- bỏ hẳn legacy Blob output path
- refactor backend query loader để đọc trực tiếp từ pipeline storage
- bỏ production materialization/service-repository query path
- cập nhật validation, manifest, worker flow
- giữ lại file output branch dành riêng cho local execution
- thêm bài test/spike với Azure Cosmos DB Emulator trước khi refactor chính

### Ngoài scope
- không đổi input document storage trừ khi phát sinh bắt buộc
- không đổi GraphRAG query factory hoặc query API signatures
- không đổi vector store path trừ khi phát sinh dependency kỹ thuật
- không loại bỏ local file flow

---

## Runtime modes sau khi hoàn tất

### Mode 1 — `cosmos_pipeline` cho cloud/worker
Dùng cho:
- background indexing worker
- production / staging runtime
- mọi flow cần publish `activeVersion`

Đặc điểm:
- output GraphRAG luôn vào Cosmos DB
- query service load trực tiếp từ pipeline Cosmos
- không có Blob fallback
- không có serving materialization step

### Mode 2 — `local_file` cho local/dev
Dùng cho:
- chạy local trên máy dev
- debug nhanh indexing mà chưa cấu hình Azure Cosmos
- smoke test local cơ bản

Đặc điểm:
- output GraphRAG ghi local filesystem
- query service load local parquet trực tiếp
- không dùng Blob
- không cần Cosmos để dev loop cơ bản chạy được

---

## Phase 0 — Spike với Azure Cosmos DB Emulator

### Mục tiêu
Xác nhận assumptions về Cosmos pipeline output và direct-query path hoạt động được trong local environment trước khi sửa flow chính.

### Cách chạy emulator
Ưu tiên chạy bằng Docker theo tài liệu official.

#### Option A — Docker Linux container
```bash
docker pull mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest

docker run \
  --publish 8081:8081 \
  --publish 10250-10255:10250-10255 \
  --name linux-emulator \
  --detach \
  mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest
```

#### Option B — Docker Windows container
```powershell
docker pull mcr.microsoft.com/cosmosdb/windows/azure-cosmos-emulator

$parameters = @(
  "--publish", "8081:8081",
  "--publish", "10250-10255:10250-10255",
  "--name", "windows-emulator",
  "--detach"
)
docker run @parameters mcr.microsoft.com/cosmosdb/windows/azure-cosmos-emulator
```

#### Option C — Windows local emulator
Fallback khi Docker image hoặc TLS setup gây block.

### TLS certificate
Nếu dùng container variant, phải import certificate trước khi chạy backend thật bằng Python SDK.

#### Linux container certificate
```powershell
$parameters = @{
  Uri = 'https://localhost:8081/_explorer/emulator.pem'
  Method = 'GET'
  OutFile = 'emulatorcert.crt'
  SkipCertificateCheck = $True
}
Invoke-WebRequest @parameters

$importParams = @{
  FilePath = 'emulatorcert.crt'
  CertStoreLocation = 'Cert:\CurrentUser\Root'
}
Import-Certificate @importParams
```

#### Windows container certificate
```powershell
docker cp windows-emulator:C:\CosmosDB.Emulator\bind-mount .
.\bind-mount\importcert.ps1
```

### Local backend config cho emulator
Tạo cấu hình local riêng cho emulator:

- `AZURE_COSMOS_ENDPOINT=https://localhost:8081/`
- `AZURE_COSMOS_KEY=<emulator key>`
- `AZURE_COSMOS_DATABASE_NAME=gtog-emulator`
- database/container naming cho pipeline output phải tách biệt với data thật

### Việc phải làm trong spike
1. xác nhận backend `.venv` kết nối được tới emulator bằng `azure-cosmos`
2. xác nhận `CosmosClient.create_database_if_not_exists(...)` chạy được
3. xác nhận `create_container_if_not_exists(...)` chạy được
4. mô phỏng `CosmosDBPipelineStorage.set(...)` với ít nhất các datasets:
   - `entities`
   - `relationships`
   - `text_units`
   - `communities`
   - `community_reports`
5. đọc ngược dữ liệu qua `CosmosDBPipelineStorage.get(..., as_bytes=True)` và convert lại thành `DataFrame`
6. feed trực tiếp các `DataFrame` đó vào ít nhất một GraphRAG query API
7. xác nhận backend-side loader có thể nạp đủ dataset contract cho `global`, `local`, `drift`, `tog`
8. ghi lại vấn đề về TLS, partitioning, hiệu năng đọc prefix, schema fidelity, và query latency warm/cold cache

### Deliverables của spike
- checklist pass/fail
- script hoặc test nhỏ để verify write/read/query round-trip
- số liệu cơ bản về write/read/query latency và certificate issues

### Success criteria của Phase 0
- backend Python kết nối emulator qua HTTPS thành công
- round-trip parquet bytes <-> Cosmos items <-> DataFrame hoạt động
- có thể chạy ít nhất 1 query mode end-to-end bằng DataFrame load từ pipeline storage
- direct-query path khả thi mà không cần serving materialization

---

## Phase 1 — Chuẩn hóa config cho 2 mode: `cosmos_pipeline` và `local_file`

### File cần sửa
- `backend/app/config.py`
- `backend/app/utils/helpers.py`
- `backend/app/utils/config_compatibility.py`

### Mục tiêu
Tách rõ nhánh runtime thật và nhánh local, bỏ hẳn Blob path.

### Việc cần làm
1. thêm setting điều khiển storage mode, ví dụ:
   - `INDEX_OUTPUT_MODE=cosmos_pipeline|local_file`
2. refactor `load_graphrag_config(...)` trong `backend/app/utils/helpers.py`
3. xóa helper `_blob_storage_cli_overrides(...)` khỏi indexing output flow
4. giữ helper file output nhưng giới hạn rõ cho local mode
5. tạo helper mới, ví dụ `_cosmos_output_cli_overrides(...)`, để set:
   - `output.type=cosmosdb`
   - `output.base_dir=<pipeline_database_name>`
   - `output.container_name=<container_name_theo_collection_version>`
   - `output.connection_string` hoặc `output.cosmosdb_account_url`
6. trong `local_file` mode:
   - giữ `output.type=file`
   - giữ `output.base_dir=<collection_dir/output>`
7. update config validation:
   - `cosmos_pipeline` mode fail fast nếu thiếu Cosmos config
   - `local_file` mode cho phép chạy không có Cosmos

### Kết quả mong đợi
- cloud/worker luôn dùng `output.type=cosmosdb`
- local vẫn chạy được với `output.type=file`
- không còn nhánh output `blob`

---

## Phase 2 — Tạo repository đọc pipeline output từ Cosmos

### File mới
- `backend/app/repositories/pipeline_output_repository.py`

### Mục tiêu
Đóng gói logic đọc dữ liệu từ `CosmosDBPipelineStorage` để query/indexing service không phụ thuộc trực tiếp vào GraphRAG storage internals.

### API đề xuất
- `load_dataframe(collection_id: str, version: str, dataset: str) -> pd.DataFrame`
- `dataset_exists(collection_id: str, version: str, dataset: str) -> bool`
- `count_rows(collection_id: str, version: str, dataset: str) -> int`
- `load_required_frames(collection_id: str, version: str, datasets: list[str]) -> dict[str, pd.DataFrame]`

### Việc cần làm
1. resolve đúng database/container theo `collectionId + version`
2. wrap `CosmosDBPipelineStorage.get(...)`
3. chuyển parquet bytes thành `DataFrame`
4. chuẩn hóa exception cho missing dataset, malformed dataset, empty dataset
5. thêm logging để trace load time, row counts, cache behavior

### Kết quả mong đợi
- production services chỉ gọi repository API, không query raw Cosmos pipeline storage trực tiếp

---

## Phase 3 — Refactor backend query layer theo `graphrag/query`

### File cần sửa
- `backend/app/services/query_service.py`
- `backend/app/services/query_service_base.py`
- `backend/app/services/query_service_global.py`
- `backend/app/services/query_service_local.py`
- `backend/app/services/query_service_drift.py`
- `backend/app/services/query_service_tog.py`

### Mục tiêu
Đổi backend query layer từ `serving_repository` sang pipeline-storage loader, nhưng vẫn cấp đúng `DataFrame` contract mà `graphrag/api/query.py` cần.

### Việc cần làm
1. thay `self.serving_repo = get_serving_repository()` bằng dependency đọc pipeline output repository cho production path
2. đổi `_load_dataset_frame(...)` để load từ pipeline repository trong `cosmos_pipeline`
3. giữ `_load_context_from_local(...)` cho `local_file`
4. thay `_load_context_from_serving(...)` bằng loader mới, ví dụ `_load_context_from_pipeline(...)`
5. giữ nguyên `_REQUIRED_DATASETS` của backend cho từng mode để match với GraphRAG API contract
6. tiếp tục normalize `community_reports` qua `_normalize_community_reports_frame(...)`
7. giữ caching layer `serving_context_cache` hoặc đổi tên phù hợp hơn nếu cần, nhưng logic cache vẫn nên tồn tại vì direct read từ Cosmos pipeline có thể tốn chi phí
8. không đổi cách gọi `api.global_search(...)`, `api.local_search(...)`, `api.drift_search(...)`, `api.tog_search(...)`

### Ghi chú quan trọng
Phần phải thay là **backend data loader**, không phải GraphRAG query engines. `graphrag/api/query.py` đã nhận `DataFrame` đúng kiểu cần dùng.

### Kết quả mong đợi
- production query path load trực tiếp từ pipeline Cosmos
- local query path load trực tiếp từ local parquet
- không còn query path dùng `serving_repository`

---

## Phase 4 — Đổi indexing worker sang direct Cosmos output và direct publish

### File cần sửa
- `backend/app/services/indexing_service.py`

### Mục tiêu
Worker runtime thật không còn phụ thuộc vào parquet artifacts trong Blob/File và không còn materialization step.

### Việc cần làm
1. giữ bước load config và chạy `api.build_index(...)`
2. đảm bảo worker chỉ chạy với mode `cosmos_pipeline`
3. sau khi build xong, verify required datasets đã tồn tại trong Cosmos pipeline output
4. ghi artifact manifest với metadata mới:
   - `storageMode=cosmos_pipeline`
   - pipeline database/container
   - row counts từng dataset
   - checksum/version metadata
5. chỉ set `activeVersion` sau khi verification hoàn tất
6. nếu verify thất bại thì fail job ngay, không fallback Blob
7. xóa call tới `serving_materialization_service.materialize_collection_version(...)`

### Required datasets để verify
- `entities`
- `relationships`
- `text_units`
- `communities`
- `community_reports`
- `covariates` nếu có

### Kết quả mong đợi
- indexing job production thành công khi pipeline Cosmos output hoàn chỉnh
- không còn materialization step ở worker flow

---

## Phase 5 — Dọn sạch serving-materialization path khỏi production

### File cần sửa hoặc xóa
- `backend/app/services/serving_materialization_service.py`
- `backend/app/repositories/serving_repository.py`
- `backend/app/services/__init__.py`
- các import, wiring, validation còn tham chiếu serving materialization/runtime serving repository

### Mục tiêu
Loại bỏ tầng serving read model khỏi production query path vì không còn cần nữa.

### Việc cần làm
1. xóa production usage của `serving_materialization_service`
2. xóa production usage của `serving_repository` trong query path
3. nếu `serving_repository` không còn consumer hợp lệ nào khác, xóa hẳn
4. cập nhật container/config docs để không còn coi serving containers là bắt buộc cho query runtime

### Kết quả mong đợi
- không còn duplicate storage layer cho query runtime production
- codebase gọn hơn và gần GraphRAG gốc hơn

---

## Phase 6 — Dọn validation và helper legacy

### File cần sửa
- `backend/app/utils/helpers.py`
- `backend/app/services/storage_service.py`
- các service/router còn check `output/` hoặc parquet artifacts theo assumption cũ

### Mục tiêu
Loại bỏ assumption production indexed state = blob/file artifacts hoặc serving datasets, nhưng vẫn giữ local file logic tối thiểu cần thiết.

### Việc cần làm
1. refactor `validate_collection_indexed(...)`
   - production path: indexed = có `activeVersion` và pipeline datasets đầy đủ
   - local path: vẫn có thể check file output nếu local execution đang dùng file mode
2. review `get_search_data_paths(...)`
   - nếu chỉ phục vụ local execution thì ghi rõ phạm vi local-only
   - nếu production code đang dùng thì chuyển sang pipeline repository abstraction
3. review `storage_service.py` và các nơi khác đang check `entities.parquet`
4. xóa helper và checks chỉ còn phục vụ Blob output hoặc serving materialization cũ

### Kết quả mong đợi
- không còn logic legacy Blob-output
- không còn logic production phụ thuộc serving datasets materialized
- file-output logic chỉ còn ở local branch

---

## Test Plan

### Test cấp spike/emulator
- connect tới emulator bằng backend `.venv`
- create database/container
- pipeline write/read round-trip
- chạy direct query bằng DataFrame load từ pipeline Cosmos

### Test cấp service
- `load_graphrag_config(...)` trả đúng `output.type=cosmosdb` trong `cosmos_pipeline`
- `load_graphrag_config(...)` trả đúng `output.type=file` trong `local_file`
- pipeline output repository load được từng dataset
- query loader nạp đúng dataset contract cho `global`, `local`, `drift`, `tog`
- `community_reports` vẫn được normalize đúng
- artifact manifest ghi đúng counts
- `activeVersion` chỉ update sau khi pipeline verification xong

### Test cấp integration — production-like path
- tạo collection
- upload documents
- chạy indexing job end-to-end với `cosmos_pipeline`
- chạy query `global`, `local`, `drift`, `tog`
- verify không có phụ thuộc vào Blob output
- verify không cần serving materialization/serving repository

### Test cấp integration — local path
- chạy local indexing với `local_file`
- verify output files được tạo đúng
- chạy local query `global`, `local`, `drift`, `tog`
- verify local query/dev loop vẫn hoạt động

### Test cấp regression
- xóa hoàn toàn Blob output nhưng cloud path vẫn chạy
- bỏ production serving path nhưng query vẫn chạy
- local file mode vẫn hoạt động sau cleanup
- worker retry/failure path vẫn hoạt động khi Cosmos pipeline output lỗi

---

## Risk Register

### 1. `CosmosDBPipelineStorage` partition theo `/id`
Rủi ro:
- query prefix `STARTSWITH(c.id, 'entities')` có thể tốn RU

Giảm thiểu:
- container riêng theo `collectionId + version`
- benchmark bằng emulator trước, rồi validate tiếp trên Azure thật với dataset thật

### 2. Dataset lớn làm direct query load chậm
Rủi ro:
- query runtime phải rebuild `DataFrame` từ Cosmos items

Giảm thiểu:
- giữ cache trong query service
- đo warm/cold latency theo dataset trong spike
- tối ưu ở repository layer, không quay lại serving duplication nếu chưa thật sự cần

### 3. TLS/certificate issues ở local dev
Rủi ro:
- Python `.venv` không trust emulator cert

Giảm thiểu:
- import cert đúng theo docs
- chốt một runbook local dev sau khi spike pass

### 4. Breaking change cho production runtime
Rủi ro:
- env nào chưa có Cosmos config sẽ không index được ở `cosmos_pipeline`

Giảm thiểu:
- fail fast lúc startup/config validation
- giữ `local_file` mode cho dev, không giữ Blob fallback cho production

### 5. Backend phụ thuộc trực tiếp hơn vào pipeline storage contract
Rủi ro:
- nếu `CosmosDBPipelineStorage` đổi key/prefix/schema behavior, backend query loader có thể bị ảnh hưởng

Giảm thiểu:
- bọc toàn bộ access qua `pipeline_output_repository`
- tránh để `query_service.py` đụng trực tiếp raw storage API

---

## Thứ tự thực hiện đề xuất
1. chạy Phase 0 với emulator
2. sửa `backend/app/config.py`
3. sửa `backend/app/utils/helpers.py`
4. thêm `backend/app/repositories/pipeline_output_repository.py`
5. sửa `backend/app/services/query_service.py` và các query services liên quan
6. sửa `backend/app/services/indexing_service.py`
7. xóa production materialization/serving path
8. dọn validation/helper legacy Blob
9. verify local file mode vẫn sống
10. chạy integration tests
11. cập nhật docs/runbook nếu cần

---

## Definition of Done
- production/worker indexing luôn dùng `output.type=cosmosdb`
- không còn luồng Blob output cho indexing
- không còn production materialization sang serving Cosmos containers
- production query đọc trực tiếp từ pipeline Cosmos output
- local/dev vẫn chạy được với `output.type=file`
- query `global`, `local`, `drift`, `tog` chạy được trên cả `cosmos_pipeline` và `local_file`
- `community_reports` vẫn được normalize đúng trước khi đưa vào GraphRAG query APIs
- `activeVersion` chỉ publish sau khi pipeline datasets hoàn chỉnh
- emulator spike pass
- integration flow pass end-to-end cho cả `cosmos_pipeline` và `local_file`

---

## Tài liệu tham chiếu
- `docs/cosmosdb-indexing-flow.md`
- `docs/cosmosdb-indexing-gaps.md`
- `graphrag/api/query.py`
- `graphrag/query/indexer_adapters.py`
- `graphrag/query/factory.py`
- `backend/app/services/query_service.py`
- `backend/app/services/query_service_base.py`
- https://github.com/Azure/azure-cosmos-db-emulator-docker
- https://learn.microsoft.com/en-us/azure/cosmos-db/how-to-develop-emulator?tabs=docker-linux%2Ccsharp&pivots=api-nosql
