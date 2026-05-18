# GraphRAG Indexing Flow khi dùng CosmosDB cho Output Storage

## Mục tiêu
Tài liệu này mô tả chi tiết luồng indexing trong `graphrag/index` khi cấu hình `output.type: cosmosdb`, và chỉ ra các file source liên quan để debug/mở rộng.

---

## 1) Cấu hình đầu vào cần có

Ví dụ trong `settings.yaml`:

```yaml
output:
  type: cosmosdb
  base_dir: <database_name>
  container_name: <container_name>
  connection_string: ${COSMOS_CONNECTION_STRING}
  # hoặc cosmosdb_account_url: https://<account>.documents.azure.com:443/
```

Ý nghĩa trường:
- `output.type=cosmosdb`: chọn backend lưu output của pipeline là CosmosDB.
- `output.base_dir`: được dùng như **database name** trong CosmosDB pipeline storage.
- `output.container_name`: container chứa dữ liệu output.
- `connection_string` hoặc `cosmosdb_account_url`: thông tin kết nối CosmosDB.

Tham chiếu:
- `graphrag/config/models/storage_config.py`
- `graphrag/settings.yaml`

---

## 2) Luồng khởi tạo storage trong pipeline

1. `run_pipeline(...)` tạo storage:
   - `input_storage = create_storage_from_config(config.input.storage)`
   - `output_storage = create_storage_from_config(config.output)`

2. `create_storage_from_config(...)` gọi `StorageFactory.create_storage(...)`.

3. Với `type=cosmosdb`, factory trả về `CosmosDBPipelineStorage`.

Tham chiếu:
- `graphrag/index/run/run_pipeline.py` (khởi tạo input/output storage)
- `graphrag/utils/api.py` (`create_storage_from_config`)
- `graphrag/storage/factory.py` (map `cosmosdb -> CosmosDBPipelineStorage`)

---

## 3) Workflow logic không đổi, chỉ đổi backend I/O

Pipeline standard vẫn chạy các workflow như cũ (load documents, chunking, extract graph, communities, reports, embeddings...).

Điểm khác là tất cả thao tác đọc/ghi bảng đều đi qua `context.output_storage`, và với cấu hình này backend là CosmosDB.

Ví dụ:
- `generate_text_embeddings` đọc các bảng từ `context.output_storage`.
- Nếu bật snapshots, workflow ghi bảng embeddings về lại `context.output_storage`.

Tham chiếu:
- `graphrag/index/run/run_pipeline.py`
- `graphrag/index/workflows/generate_text_embeddings.py`

---

## 4) Cách CosmosDBPipelineStorage ghi dữ liệu

Trong `CosmosDBPipelineStorage.set(key, value)`:

### Trường hợp bảng parquet
- Khi `value` là `bytes` (parquet bytes), storage sẽ:
  1. Parse parquet bytes thành DataFrame.
  2. Convert DataFrame thành danh sách JSON records.
  3. Upsert từng record thành một item CosmosDB.
  4. Gán `id` theo dạng `"<prefix>:<row_id>"`, với `prefix = key.split('.') [0]` (vd `documents`, `entities`, `relationships`).

Kết quả: dữ liệu bảng nằm trong CosmosDB theo từng item, không bắt buộc tạo file `.parquet` vật lý trên disk.

### Trường hợp JSON metadata
- Với dữ liệu như `context.json`, `stats.json`, storage ghi item dạng:
  - `id = key`
  - `body = parsed json`

Tham chiếu:
- `graphrag/storage/cosmosdb_pipeline_storage.py` (`set`, `_get_prefix`)

---

## 5) Cách đọc lại dữ liệu từ CosmosDB

Trong `CosmosDBPipelineStorage.get(key, as_bytes=True)`:

1. Query các item có `id` bắt đầu bằng `prefix`.
2. Ghép thành DataFrame.
3. Serialize ngược lại thành parquet bytes.
4. Trả về cho pipeline consumer (đọc như đang đọc bảng parquet).

Với key JSON thường (`as_bytes=False`), đọc item theo `id` và trả body JSON.

Tham chiếu:
- `graphrag/storage/cosmosdb_pipeline_storage.py` (`get`)

---

## 6) Kết luận ngắn

- Pipeline vẫn dùng **parquet format** để trao đổi dữ liệu bảng nội bộ.
- Khi output là CosmosDB, dữ liệu bảng được map thành các item CosmosDB, không cần file parquet vật lý trung gian.
- Logic nghiệp vụ của workflow không đổi; chỉ thay tầng lưu trữ I/O.

---

## 7) Danh sách file liên quan (tra cứu nhanh)

### Entry và orchestration
- `graphrag/index/run/run_pipeline.py`
- `graphrag/index/run/utils.py`

### Workflow ví dụ dùng output storage
- `graphrag/index/workflows/generate_text_embeddings.py`

### Factory/API tạo storage
- `graphrag/utils/api.py`
- `graphrag/storage/factory.py`

### CosmosDB pipeline storage implementation
- `graphrag/storage/cosmosdb_pipeline_storage.py`

### Cấu hình
- `graphrag/config/models/storage_config.py`
- `graphrag/config/defaults.py`
- `graphrag/settings.yaml`

### (Liên quan nhưng khác vai trò) Vector store CosmosDB
- `graphrag/vector_stores/cosmosdb.py`
- `graphrag/vector_stores/factory.py`

Ghi chú: `graphrag/vector_stores/cosmosdb.py` là cho vector search store, không phải pipeline output table storage.