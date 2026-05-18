# Gaps giữa tài liệu `cosmosdb-indexing-flow` và backend hiện tại

## Phạm vi đối chiếu
- Tài liệu chuẩn: `docs/cosmosdb-indexing-flow.md`
- Implementation đối chiếu: `backend/app/**`

---

## Tóm tắt ngắn
Backend hiện tại **chưa chạy theo luồng `output.type: cosmosdb` của GraphRAG pipeline** như tài liệu mô tả. Luồng thực tế là:
1. GraphRAG index ghi output ra **Blob** hoặc **File** (parquet).
2. Sau đó backend đọc parquet và **materialize** sang Cosmos serving containers.

---

## Gap chi tiết

### 1) Output storage của GraphRAG không phải CosmosDB
**Theo tài liệu kỳ vọng**
- `output.type=cosmosdb` để `StorageFactory` tạo `CosmosDBPipelineStorage`.

**Thực tế backend**
- Runtime override cấu hình output thành `blob` hoặc `file`, không có nhánh `cosmosdb`:
  - `backend/app/utils/helpers.py:51` (`"output.type": "blob"` trong `_blob_storage_cli_overrides`)
  - `backend/app/utils/helpers.py:365` (`"output.type": "file"` khi local)

**Tác động**
- Pipeline không sử dụng `CosmosDBPipelineStorage` cho output table storage.

---

### 2) Luồng hiện tại vẫn phụ thuộc parquet artifacts trung gian
**Theo tài liệu kỳ vọng**
- Khi output là CosmosDB, dữ liệu bảng được map trực tiếp thành item CosmosDB; không bắt buộc file parquet vật lý trung gian.

**Thực tế backend**
- `serving_materialization_service` yêu cầu và đọc các file parquet (`entities.parquet`, `relationships.parquet`, ...):
  - `backend/app/services/serving_materialization_service.py:14`
  - `backend/app/services/serving_materialization_service.py:41`
  - `backend/app/services/serving_materialization_service.py:47`

**Tác động**
- Hệ thống vẫn phụ thuộc output parquet ở `output/` (Blob/File) trước khi có dữ liệu serving trong Cosmos.

---

### 3) Cosmos trong backend chủ yếu đang dùng cho control-plane + serving, không phải pipeline output
**Theo tài liệu kỳ vọng**
- Cosmos đóng vai trò backend output storage của GraphRAG indexing pipeline.

**Thực tế backend**
- Cosmos đang được dùng cho:
  - Control-plane metadata (collections, jobs, trạng thái)
  - Serving datasets đã materialize
- Điểm khởi chạy index vẫn gọi `api.build_index(...)` với config override Blob/File:
  - `backend/app/services/indexing_service.py:313`
  - `backend/app/utils/helpers.py:381` (load config với `cli_overrides`)

**Tác động**
- Có hai tầng lưu trữ nối tiếp (Blob/File -> Cosmos serving), chưa phải một tầng output Cosmos trực tiếp từ pipeline.

---

## Kết luận
Backend `backend/app` hiện tại **chưa khớp** với luồng trong `docs/cosmosdb-indexing-flow.md` nếu hiểu tài liệu là “GraphRAG output trực tiếp vào CosmosDB (`output.type=cosmosdb`)”.

Luồng đang chạy là mô hình trung gian parquet (Blob/File) rồi materialization sang Cosmos.

---

## Gợi ý bước tiếp theo (chưa triển khai)
1. Quyết định rõ target architecture:
   - A) Giữ mô hình hiện tại (Blob/File -> Cosmos serving), hoặc
   - B) Chuyển sang pipeline output trực tiếp Cosmos (`output.type=cosmosdb`).
2. Nếu chọn B, cần rà soát lại `load_graphrag_config` và `serving_materialization_service` để bỏ phụ thuộc bắt buộc vào parquet artifacts.