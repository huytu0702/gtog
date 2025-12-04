# GraphRAG Backend API Documentation

## Tổng quan hệ thống

GraphRAG Backend là một hệ thống API RESTful được xây dựng bằng FastAPI, cho phép người dùng tạo collections, upload documents, thực hiện indexing và tìm kiếm thông tin thông qua 4 phương pháp search khác nhau của GraphRAG.

### Kiến trúc hệ thống

```
┌─────────────────┐
│   Frontend      │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (Port 8000)    │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Storage │ │Indexing│ │ Query  │ │GraphRAG│
│Service │ │Service │ │Service │ │ Engine │
└────────┘ └────────┘ └────────┘ └────────┘
```

### Công nghệ sử dụng

- **Framework**: FastAPI (Python)
- **Storage**: File system (local storage)
- **Database**: PostgreSQL với pgvector extension
- **AI Models**: OpenAI GPT-4o-mini, text-embedding-3-small
- **GraphRAG**: Microsoft GraphRAG

### Base URL

```
http://127.0.0.1:8000
```

---

## 1. Collections API

Collections là các workspace độc lập để quản lý documents và thực hiện indexing/search.

### 1.1 Tạo Collection

**Endpoint**: `POST /api/collections`

**Description**: Tạo một collection mới

**Request Body**:
```json
{
  "name": "my-collection",
  "description": "Optional description"
}
```

**Validation**:
- `name`: Required, 1-100 ký tự, chỉ chấp nhận `a-zA-Z0-9_-`
- `description`: Optional, tối đa 500 ký tự

**Response**: `201 Created`
```json
{
  "id": "my-collection",
  "name": "my-collection",
  "description": "Optional description",
  "created_at": "2025-12-04T12:00:00Z",
  "document_count": 0,
  "indexed": false
}
```

**Error Responses**:
- `400 Bad Request`: Tên collection không hợp lệ hoặc đã tồn tại
- `500 Internal Server Error`: Lỗi server

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-collection",
    "description": "My first collection"
  }'
```

---

### 1.2 Lấy danh sách Collections

**Endpoint**: `GET /api/collections`

**Description**: Lấy danh sách tất cả collections

**Response**: `200 OK`
```json
{
  "collections": [
    {
      "id": "collection-1",
      "name": "collection-1",
      "description": "First collection",
      "created_at": "2025-12-04T12:00:00Z",
      "document_count": 5,
      "indexed": true
    },
    {
      "id": "collection-2",
      "name": "collection-2",
      "description": null,
      "created_at": "2025-12-04T13:00:00Z",
      "document_count": 0,
      "indexed": false
    }
  ],
  "total": 2
}
```

**Example cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/api/collections"
```

---

### 1.3 Lấy thông tin Collection

**Endpoint**: `GET /api/collections/{collection_id}`

**Description**: Lấy thông tin chi tiết của một collection

**Path Parameters**:
- `collection_id`: ID của collection

**Response**: `200 OK`
```json
{
  "id": "my-collection",
  "name": "my-collection",
  "description": "My collection",
  "created_at": "2025-12-04T12:00:00Z",
  "document_count": 3,
  "indexed": true
}
```

**Error Responses**:
- `404 Not Found`: Collection không tồn tại

**Example cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/api/collections/my-collection"
```

---

### 1.4 Xóa Collection

**Endpoint**: `DELETE /api/collections/{collection_id}`

**Description**: Xóa collection và tất cả documents, index data liên quan

**Path Parameters**:
- `collection_id`: ID của collection

**Response**: `204 No Content`

**Error Responses**:
- `404 Not Found`: Collection không tồn tại

**Example cURL**:
```bash
curl -X DELETE "http://127.0.0.1:8000/api/collections/my-collection"
```

---

## 2. Documents API

Quản lý documents trong một collection.

### 2.1 Upload Document

**Endpoint**: `POST /api/collections/{collection_id}/documents`

**Description**: Upload một document vào collection

**Path Parameters**:
- `collection_id`: ID của collection

**Request**: `multipart/form-data`
- `file`: File cần upload (text, markdown, pdf, docx, etc.)

**Response**: `201 Created`
```json
{
  "name": "document.txt",
  "size": 1024,
  "uploaded_at": "2025-12-04T12:00:00Z"
}
```

**Error Responses**:
- `404 Not Found`: Collection không tồn tại
- `500 Internal Server Error`: Lỗi khi upload file

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/documents" \
  -F "file=@/path/to/document.txt"
```

**Example JavaScript (Fetch)**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch(
  'http://127.0.0.1:8000/api/collections/my-collection/documents',
  {
    method: 'POST',
    body: formData
  }
);

const result = await response.json();
```

---

### 2.2 Lấy danh sách Documents

**Endpoint**: `GET /api/collections/{collection_id}/documents`

**Description**: Lấy danh sách tất cả documents trong collection

**Path Parameters**:
- `collection_id`: ID của collection

**Response**: `200 OK`
```json
{
  "documents": [
    {
      "name": "document1.txt",
      "size": 1024,
      "uploaded_at": "2025-12-04T12:00:00Z"
    },
    {
      "name": "document2.md",
      "size": 2048,
      "uploaded_at": "2025-12-04T12:30:00Z"
    }
  ],
  "total": 2
}
```

**Error Responses**:
- `404 Not Found`: Collection không tồn tại

**Example cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/api/collections/my-collection/documents"
```

---

### 2.3 Xóa Document

**Endpoint**: `DELETE /api/collections/{collection_id}/documents/{document_name}`

**Description**: Xóa một document khỏi collection

**Path Parameters**:
- `collection_id`: ID của collection
- `document_name`: Tên của document

**Response**: `204 No Content`

**Error Responses**:
- `404 Not Found`: Collection hoặc document không tồn tại

**Example cURL**:
```bash
curl -X DELETE "http://127.0.0.1:8000/api/collections/my-collection/documents/document.txt"
```

---

## 3. Indexing API

Quản lý quá trình indexing documents bằng GraphRAG.

### 3.1 Bắt đầu Indexing

**Endpoint**: `POST /api/collections/{collection_id}/index`

**Description**: Bắt đầu quá trình indexing cho collection

**Path Parameters**:
- `collection_id`: ID của collection

**Response**: `202 Accepted`
```json
{
  "collection_id": "my-collection",
  "status": "running",
  "progress": 0.0,
  "message": "Indexing started",
  "started_at": "2025-12-04T12:00:00Z",
  "completed_at": null,
  "error": null
}
```

**Error Responses**:
- `404 Not Found`: Collection không tồn tại
- `400 Bad Request`: Collection chưa có document nào
- `500 Internal Server Error`: Lỗi khi bắt đầu indexing

**Notes**:
- Indexing chạy background, không block request
- Collection phải có ít nhất 1 document
- Quá trình indexing có thể mất vài phút tùy thuộc vào số lượng documents

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/index"
```

---

### 3.2 Kiểm tra trạng thái Indexing

**Endpoint**: `GET /api/collections/{collection_id}/index`

**Description**: Lấy trạng thái hiện tại của quá trình indexing

**Path Parameters**:
- `collection_id`: ID của collection

**Response**: `200 OK`
```json
{
  "collection_id": "my-collection",
  "status": "completed",
  "progress": 100.0,
  "message": "Indexing completed successfully",
  "started_at": "2025-12-04T12:00:00Z",
  "completed_at": "2025-12-04T12:05:00Z",
  "error": null
}
```

**Status Values**:
- `pending`: Chưa bắt đầu
- `running`: Đang chạy
- `completed`: Hoàn thành
- `failed`: Thất bại

**Error Responses**:
- `404 Not Found`: Không tìm thấy trạng thái indexing cho collection

**Example cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/api/collections/my-collection/index"
```

**Polling Example (JavaScript)**:
```javascript
async function pollIndexingStatus(collectionId) {
  const checkStatus = async () => {
    const response = await fetch(
      `http://127.0.0.1:8000/api/collections/${collectionId}/index`
    );
    const status = await response.json();
    
    console.log(`Progress: ${status.progress}%`);
    
    if (status.status === 'completed') {
      console.log('Indexing completed!');
      return status;
    } else if (status.status === 'failed') {
      console.error('Indexing failed:', status.error);
      throw new Error(status.error);
    } else {
      // Poll lại sau 2 giây
      await new Promise(resolve => setTimeout(resolve, 2000));
      return checkStatus();
    }
  };
  
  return checkStatus();
}
```

---

## 4. Search API

GraphRAG cung cấp 4 phương pháp search khác nhau, mỗi phương pháp phù hợp với các use case khác nhau.

### 4.1 Global Search

**Endpoint**: `POST /api/collections/{collection_id}/search/global`

**Description**: Tìm kiếm toàn cục, phù hợp cho câu hỏi tổng quan về toàn bộ dataset

**Path Parameters**:
- `collection_id`: ID của collection

**Request Body**:
```json
{
  "query": "What are the main themes in the documents?",
  "community_level": null,
  "dynamic_community_selection": false,
  "response_type": "Multiple Paragraphs",
  "streaming": false
}
```

**Request Fields**:
- `query`: Câu hỏi (required, 1-1000 ký tự)
- `community_level`: Level của community graph (optional, 0-10, null = auto)
- `dynamic_community_selection`: Tự động chọn community level (default: false)
- `response_type`: Định dạng response (xem bên dưới)
- `streaming`: Enable streaming response (default: false)

**Response Types**:
- `"Single Paragraph"`: Một đoạn văn ngắn gọn
- `"Single Sentence"`: Một câu tóm tắt
- `"Multiple Paragraphs"`: Nhiều đoạn văn chi tiết (default)
- `"List of 3-7 Points"`: Danh sách 3-7 điểm
- `"List of 5-10 Points"`: Danh sách 5-10 điểm

**Response**: `200 OK`
```json
{
  "query": "What are the main themes in the documents?",
  "response": "The main themes include...",
  "context_data": {
    "reports": [...],
    "entities": [...],
    "relationships": [...]
  },
  "method": "global"
}
```

**Use Cases**:
- Câu hỏi tổng quan về toàn bộ dataset
- Tìm kiếm patterns, themes chung
- Phân tích high-level

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/search/global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main themes?",
    "response_type": "Multiple Paragraphs"
  }'
```

---

### 4.2 Local Search

**Endpoint**: `POST /api/collections/{collection_id}/search/local`

**Description**: Tìm kiếm cục bộ, phù hợp cho câu hỏi cụ thể về entities hoặc concepts

**Path Parameters**:
- `collection_id`: ID của collection

**Request Body**:
```json
{
  "query": "What is the relationship between entity A and entity B?",
  "community_level": 2,
  "response_type": "Multiple Paragraphs",
  "streaming": false
}
```

**Request Fields**:
- `query`: Câu hỏi (required, 1-1000 ký tự)
- `community_level`: Level của community graph (default: 2, range: 0-10)
- `response_type`: Định dạng response (như Global Search)
- `streaming`: Enable streaming response (default: false)

**Response**: `200 OK`
```json
{
  "query": "What is the relationship between entity A and entity B?",
  "response": "Entity A and Entity B are related through...",
  "context_data": {
    "entities": [...],
    "relationships": [...],
    "sources": [...]
  },
  "method": "local"
}
```

**Use Cases**:
- Câu hỏi cụ thể về entities
- Tìm kiếm relationships giữa các entities
- Phân tích chi tiết về một topic cụ thể

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/search/local" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is entity A?",
    "community_level": 2
  }'
```

---

### 4.3 ToG Search (Tree-of-Graph)

**Endpoint**: `POST /api/collections/{collection_id}/search/tog`

**Description**: Tìm kiếm dạng cây, kết hợp global và local search

**Path Parameters**:
- `collection_id`: ID của collection

**Request Body**:
```json
{
  "query": "Explain the complete picture of topic X",
  "streaming": false
}
```

**Request Fields**:
- `query`: Câu hỏi (required, 1-1000 ký tự)
- `streaming`: Enable streaming response (default: false)

**Response**: `200 OK`
```json
{
  "query": "Explain the complete picture of topic X",
  "response": "Topic X encompasses...",
  "context_data": {
    "tree_structure": [...],
    "nodes": [...],
    "edges": [...]
  },
  "method": "tog"
}
```

**Use Cases**:
- Câu hỏi phức tạp cần cả overview và details
- Phân tích đa chiều
- Kết hợp nhiều perspectives

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/search/tog" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me a complete analysis"
  }'
```

---

### 4.4 DRIFT Search

**Endpoint**: `POST /api/collections/{collection_id}/search/drift`

**Description**: Dynamic Reasoning and Inference with Flexible Traversal - tìm kiếm động với reasoning

**Path Parameters**:
- `collection_id`: ID của collection

**Request Body**:
```json
{
  "query": "What are the implications of event X?",
  "community_level": 2,
  "response_type": "Multiple Paragraphs",
  "streaming": false
}
```

**Request Fields**:
- `query`: Câu hỏi (required, 1-1000 ký tự)
- `community_level`: Level của community graph (default: 2, range: 0-10)
- `response_type`: Định dạng response (như Global Search)
- `streaming`: Enable streaming response (default: false)

**Response**: `200 OK`
```json
{
  "query": "What are the implications of event X?",
  "response": "The implications include...",
  "context_data": {
    "reasoning_path": [...],
    "inferences": [...],
    "evidence": [...]
  },
  "method": "drift"
}
```

**Use Cases**:
- Câu hỏi cần reasoning và inference
- Phân tích nhân quả
- Dự đoán và implications

**Example cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/api/collections/my-collection/search/drift" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What might happen if...?",
    "community_level": 2
  }'
```

---

## 5. So sánh các phương pháp Search

| Phương pháp | Use Case | Tốc độ | Chi tiết | Phù hợp với |
|-------------|----------|--------|----------|-------------|
| **Global** | Câu hỏi tổng quan | Nhanh | Thấp | Overview, themes, patterns |
| **Local** | Câu hỏi cụ thể | Trung bình | Cao | Specific entities, relationships |
| **ToG** | Câu hỏi phức tạp | Chậm | Rất cao | Complex analysis, multi-perspective |
| **DRIFT** | Câu hỏi reasoning | Chậm | Cao | Causality, implications, predictions |

---

## 6. Error Handling

Tất cả API endpoints đều trả về error theo format chuẩn:

```json
{
  "detail": "Error message description"
}
```

### Common HTTP Status Codes

- `200 OK`: Request thành công
- `201 Created`: Resource được tạo thành công
- `202 Accepted`: Request được chấp nhận (async processing)
- `204 No Content`: Request thành công, không có response body
- `400 Bad Request`: Request không hợp lệ (validation error)
- `404 Not Found`: Resource không tồn tại
- `500 Internal Server Error`: Lỗi server

### Error Handling Example (JavaScript)

```javascript
async function searchCollection(collectionId, query) {
  try {
    const response = await fetch(
      `http://127.0.0.1:8000/api/collections/${collectionId}/search/global`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          response_type: 'Multiple Paragraphs'
        })
      }
    );
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Search failed:', error.message);
    throw error;
  }
}
```

---

## 7. Workflow Examples

### 7.1 Complete Workflow: Từ tạo Collection đến Search

```javascript
// 1. Tạo collection
const createResponse = await fetch('http://127.0.0.1:8000/api/collections', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'my-docs',
    description: 'My document collection'
  })
});
const collection = await createResponse.json();
console.log('Collection created:', collection.id);

// 2. Upload documents
const formData = new FormData();
formData.append('file', document1);
await fetch(`http://127.0.0.1:8000/api/collections/${collection.id}/documents`, {
  method: 'POST',
  body: formData
});
console.log('Document uploaded');

// 3. Bắt đầu indexing
await fetch(`http://127.0.0.1:8000/api/collections/${collection.id}/index`, {
  method: 'POST'
});
console.log('Indexing started');

// 4. Poll indexing status
let indexing = true;
while (indexing) {
  const statusResponse = await fetch(
    `http://127.0.0.1:8000/api/collections/${collection.id}/index`
  );
  const status = await statusResponse.json();
  
  if (status.status === 'completed') {
    indexing = false;
    console.log('Indexing completed!');
  } else if (status.status === 'failed') {
    throw new Error('Indexing failed: ' + status.error);
  } else {
    console.log(`Progress: ${status.progress}%`);
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

// 5. Thực hiện search
const searchResponse = await fetch(
  `http://127.0.0.1:8000/api/collections/${collection.id}/search/global`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'What are the main topics?',
      response_type: 'Multiple Paragraphs'
    })
  }
);
const result = await searchResponse.json();
console.log('Search result:', result.response);
```

---

## 8. Best Practices

### 8.1 Collection Naming

- Sử dụng lowercase với dashes hoặc underscores
- Tên ngắn gọn, mô tả rõ ràng
- Ví dụ: `product-docs`, `customer_feedback`, `research-papers`

### 8.2 Document Upload

- Kiểm tra file size trước khi upload
- Hỗ trợ các format: `.txt`, `.md`, `.pdf`, `.docx`
- Upload từng file một để tracking tốt hơn

### 8.3 Indexing

- Đợi indexing hoàn thành trước khi search
- Implement polling với reasonable interval (2-5 giây)
- Handle timeout cho indexing quá lâu
- Hiển thị progress cho user

### 8.4 Search

- Chọn phương pháp search phù hợp với use case
- Bắt đầu với Global search cho overview
- Sử dụng Local search cho câu hỏi cụ thể
- Cache kết quả search nếu có thể
- Implement debouncing cho search input

### 8.5 Error Handling

- Luôn check HTTP status code
- Parse error message từ response
- Hiển thị error message thân thiện cho user
- Implement retry logic cho network errors

---

## 9. Configuration

### 9.1 Environment Variables

Backend sử dụng các environment variables sau (trong file `.env`):

```env
# API Keys
OPENAI_API_KEY=your-openai-api-key
GRAPHRAG_API_KEY=your-graphrag-api-key

# Storage
STORAGE_ROOT_DIR=./storage

# Models
DEFAULT_CHAT_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# Server
HOST=0.0.0.0
PORT=8000
```

### 9.2 CORS Configuration

Nếu frontend chạy trên domain khác, cần configure CORS trong backend.

---

## 10. Rate Limits & Quotas

**Lưu ý**: Hệ thống sử dụng OpenAI API, cần chú ý:

- Rate limits của OpenAI API
- Token quotas
- Cost implications cho indexing và search
- Recommend implement caching và rate limiting ở frontend

---

## 11. Support & Resources

### GraphRAG Documentation
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)

### API Testing Tools
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### Contact
- Backend Developer: [Your Contact Info]
- Issues: [GitHub Issues URL]

---

## Changelog

### Version 1.0.0 (2025-12-04)
- Initial API release
- Support 4 search methods: Global, Local, ToG, DRIFT
- Collection and document management
- Background indexing with status tracking
