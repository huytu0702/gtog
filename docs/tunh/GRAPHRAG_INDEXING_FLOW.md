# GraphRAG Document Indexing Flow

## Tổng quan

Hệ thống GraphRAG sử dụng một pipeline indexing phức tạp để chuyển đổi các document thô thành một knowledge graph có cấu trúc, được tối ưu hóa cho việc truy vấn và tìm kiếm thông tin. Quá trình indexing diễn ra qua nhiều workflow tuần tự, mỗi workflow thực hiện một nhiệm vụ cụ thể trong việc xử lý và làm giàu dữ liệu.

## Kiến trúc Pipeline

Hệ thống hỗ trợ 4 loại pipeline indexing:

1. **Standard Indexing** - Pipeline đầy đủ với LLM-based entity extraction
2. **Fast Indexing** - Pipeline nhanh hơn sử dụng NLP-based extraction
3. **Standard Update** - Cập nhật incremental cho pipeline standard
4. **Fast Update** - Cập nhật incremental cho pipeline fast

## Flow Chart - Standard Indexing Pipeline

```mermaid
graph TB
    Start([Bắt đầu Indexing]) --> LoadDocs[1. Load Input Documents]
    
    LoadDocs --> ChunkText[2. Create Base Text Units]
    ChunkText --> ExtractGraph[3. Extract Graph]
    
    ExtractGraph --> FinalizeGraph[4. Finalize Graph]
    FinalizeGraph --> ExtractCov[5. Extract Covariates]
    
    ExtractCov --> CreateComm[6. Create Communities]
    CreateComm --> FinalTextUnits[7. Create Final Text Units]
    
    FinalTextUnits --> CreateReports[8. Create Community Reports]
    CreateReports --> GenEmbeddings[9. Generate Text Embeddings]
    
    GenEmbeddings --> End([Hoàn thành])
    
    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style LoadDocs fill:#fff4e6
    style ChunkText fill:#fff4e6
    style ExtractGraph fill:#e3f2fd
    style FinalizeGraph fill:#e3f2fd
    style ExtractCov fill:#f3e5f5
    style CreateComm fill:#fce4ec
    style FinalTextUnits fill:#fce4ec
    style CreateReports fill:#fff9c4
    style GenEmbeddings fill:#e0f2f1
```

## Flow Chart - Fast Indexing Pipeline

```mermaid
graph TB
    Start([Bắt đầu Indexing]) --> LoadDocs[1. Load Input Documents]
    
    LoadDocs --> ChunkText[2. Create Base Text Units]
    ChunkText --> ExtractGraphNLP[3. Extract Graph NLP]
    
    ExtractGraphNLP --> PruneGraph[4. Prune Graph]
    PruneGraph --> FinalizeGraph[5. Finalize Graph]
    FinalizeGraph --> CreateComm[6. Create Communities]
    
    CreateComm --> FinalTextUnits[7. Create Final Text Units]
    FinalTextUnits --> GenEmbeddings[8. Generate Text Embeddings]
    
    GenEmbeddings --> End([Hoàn thành])
    
    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style LoadDocs fill:#fff4e6
    style ChunkText fill:#fff4e6
    style ExtractGraphNLP fill:#e3f2fd
    style PruneGraph fill:#ffecb3
    style FinalizeGraph fill:#e3f2fd
    style CreateComm fill:#fce4ec
    style FinalTextUnits fill:#fce4ec
    style GenEmbeddings fill:#e0f2f1
```

## Chi tiết từng Workflow

### 1. Load Input Documents
**File**: `graphrag/index/workflows/load_input_documents.py`

**Mục đích**: Đọc và phân tích các document đầu vào từ storage (file system, blob storage, etc.)

**Quy trình**:
```mermaid
graph LR
    A[Input Storage] --> B[Create Input Handler]
    B --> C[Parse Documents]
    C --> D[Validate Format]
    D --> E[Output: documents.parquet]
    
    style A fill:#e8f5e9
    style E fill:#c8e6c9
```

**Input**: 
- Raw documents từ `input/` directory
- Supported formats: text, csv, json

**Output**: 
- DataFrame với schema:
  - `id`: Document identifier
  - `text`: Document content
  - `metadata`: Optional metadata (JSON string)

**Ví dụ thực tế**:
```python
# Input: input/document1.txt
"""
Công ty ABC được thành lập năm 2020 bởi Nguyễn Văn A. 
Công ty chuyên về phát triển phần mềm AI.
"""

# Output: documents.parquet
{
    "id": "doc_001",
    "text": "Công ty ABC được thành lập năm 2020...",
    "metadata": "{\"source\": \"document1.txt\", \"type\": \"text\"}"
}
```

---

### 2. Create Base Text Units
**File**: `graphrag/index/workflows/create_base_text_units.py`

**Mục đích**: Chia nhỏ documents thành các text chunks có kích thước phù hợp để xử lý

**Quy trình**:
```mermaid
graph TB
    A[Documents DataFrame] --> B[Group by Columns]
    B --> C{Chunking Strategy}
    C -->|Tokens| D[Token-based Chunking]
    C -->|Sentence| E[Sentence-based Chunking]
    D --> F[Apply Overlap]
    E --> F
    F --> G[Generate Chunk IDs]
    G --> H[Output: text_units.parquet]
    
    style A fill:#e8f5e9
    style H fill:#c8e6c9
```

**Thuật toán chunking**:
1. Group documents theo `group_by_columns` (default: `[id]`)
2. Aggregate text từ tất cả documents trong group
3. Split text thành chunks với:
   - `size`: Maximum tokens per chunk (default: 1200)
   - `overlap`: Tokens overlap giữa các chunks (default: 100)
4. Prepend metadata nếu được config
5. Tạo unique SHA512 hash cho mỗi chunk

**Output**:
- DataFrame với schema:
  - `id`: Chunk identifier (SHA512 hash)
  - `text`: Chunk content
  - `document_ids`: List of source document IDs
  - `n_tokens`: Number of tokens in chunk

**Ví dụ thực tế**:
```python
# Input: 1 document với 2500 tokens
# Config: size=1200, overlap=100

# Output: 3 text units
[
    {
        "id": "chunk_001_hash",
        "text": "Công ty ABC được thành lập...",  # tokens 0-1200
        "document_ids": ["doc_001"],
        "n_tokens": 1200
    },
    {
        "id": "chunk_002_hash",
        "text": "...phát triển phần mềm AI...",  # tokens 1100-2300 (overlap 100)
        "document_ids": ["doc_001"],
        "n_tokens": 1200
    },
    {
        "id": "chunk_003_hash",
        "text": "...và mở rộng thị trường.",  # tokens 2200-2500
        "document_ids": ["doc_001"],
        "n_tokens": 300
    }
]
```

---

### 3. Extract Graph (LLM-based)
**File**: `graphrag/index/workflows/extract_graph.py`

**Mục đích**: Trích xuất entities và relationships từ text units bằng LLM

**Quy trình**:
```mermaid
graph TB
    A[Text Units] --> B[LLM Graph Extraction]
    B --> C[Extract Entities]
    B --> D[Extract Relationships]
    C --> E[Merge Duplicate Entities]
    D --> F[Merge Duplicate Relationships]
    E --> G[Summarize Descriptions]
    F --> G
    G --> H[Output: entities.parquet]
    G --> I[Output: relationships.parquet]
    
    style A fill:#e8f5e9
    style H fill:#c8e6c9
    style I fill:#c8e6c9
```

**LLM Strategy**:
1. Sử dụng prompt `extract_graph.txt` để hướng dẫn LLM
2. Trích xuất các entity types được định nghĩa (organization, person, geo, event)
3. Xác định relationships giữa các entities
4. Gleanings: Lặp lại extraction để thu thập thêm thông tin (max_gleanings)
5. Async processing với concurrent requests

**Entity Schema**:
- `id`: Entity identifier
- `title`: Entity name
- `type`: Entity type (organization, person, etc.)
- `description`: Summarized description from all mentions
- `text_unit_ids`: List of chunks where entity appears

**Relationship Schema**:
- `id`: Relationship identifier
- `source`: Source entity title
- `target`: Target entity title
- `description`: Relationship description
- `weight`: Relationship strength (optional)
- `text_unit_ids`: List of chunks where relationship appears

**Ví dụ thực tế**:
```python
# Input text unit:
"""
Công ty ABC được thành lập năm 2020 bởi Nguyễn Văn A ở Hà Nội. 
Công ty chuyên về phát triển phần mềm AI.
"""

# Output entities:
[
    {
        "id": "entity_001",
        "title": "Công ty ABC",
        "type": "organization",
        "description": "Công ty chuyên về phát triển phần mềm AI, được thành lập năm 2020",
        "text_unit_ids": ["chunk_001_hash"]
    },
    {
        "id": "entity_002",
        "title": "Nguyễn Văn A",
        "type": "person",
        "description": "Nhà sáng lập Công ty ABC",
        "text_unit_ids": ["chunk_001_hash"]
    },
    {
        "id": "entity_003",
        "title": "Hà Nội",
        "type": "geo",
        "description": "Địa điểm thành lập Công ty ABC",
        "text_unit_ids": ["chunk_001_hash"]
    }
]

# Output relationships:
[
    {
        "id": "rel_001",
        "source": "Nguyễn Văn A",
        "target": "Công ty ABC",
        "description": "Nguyễn Văn A là người thành lập Công ty ABC",
        "weight": 1.0,
        "text_unit_ids": ["chunk_001_hash"]
    },
    {
        "id": "rel_002",
        "source": "Công ty ABC",
        "target": "Hà Nội",
        "description": "Công ty ABC được thành lập tại Hà Nội",
        "weight": 0.8,
        "text_unit_ids": ["chunk_001_hash"]
    }
]
```

---

### 4. Finalize Graph
**File**: `graphrag/index/workflows/finalize_graph.py`

**Mục đích**: Làm sạch và chuẩn hóa graph, tính toán các thuộc tính bổ sung

**Quy trình**:
```mermaid
graph TB
    A[Entities + Relationships] --> B{Extraction Type}
    B -->|NLP-based| C[PMI Normalization]
    B -->|LLM-based| D[Sum Weights]
    C --> E[Finalize Entities]
    D --> E
    E --> F[Finalize Relationships]
    F --> G[Compute Node Degree]
    G --> H[Compute Edge Combined Degree]
    H --> I[Remove Duplicates]
    I --> J[Output: Finalized Graph]
    
    style A fill:#e8f5e9
    style J fill:#c8e6c9
```

**Weight Processing khác nhau theo extraction method**:

#### NLP-based Extraction (Fast Pipeline)
- **PMI Normalization**: `normalize_edge_weights = True` (default)
- **Công thức PMI**: `pmi(x,y) = p(x,y) * log2(p(x,y) / (p(x) * p(y))`
- **Bias correction**: Loại bỏ bias đối với low-frequency events
- **Statistical significance**: Tính toán significance thay vì raw counts

#### LLM-based Extraction (Standard Pipeline)
- **No normalization**: Không có statistical normalization
- **LLM weights**: Sử dụng `relationship_strength` scores từ LLM
- **Sum aggregation**: `weight = sum(all_occurrence_strengths)`
- **Subjective scoring**: Weights dựa trên judgment của LLM

**Common Processing**:
- Chuẩn hóa entity names (trim, lowercase for matching)
- Tính `degree` cho mỗi entity (số lượng relationships)
- Tính `combined_degree` cho mỗi relationship
- **Final Deduplication**:
  - **Relationships**: `drop_duplicates(subset=["source", "target"])` - giữ lại 1 record duy nhất cho mỗi directed edge
  - **Entities**: `drop_duplicates(subset="title")` - giữ lại 1 record duy nhất cho mỗi entity name
- Remove orphan entities (entities không có relationships)
- Generate unique IDs (UUID) và human_readable_ids

---

### 5. Extract Covariates (Optional)
**File**: `graphrag/index/workflows/extract_covariates.py`

**Mục đích**: Trích xuất claims và facts từ text units (nếu enabled)

**Output**: `covariates.parquet` chứa các claims/facts được trích xuất

---

### 6. Create Communities
**File**: `graphrag/index/workflows/create_communities.py`

**Mục đích**: Phát hiện communities (nhóm entities liên kết chặt chẽ) trong graph

**Thuật toán**: Hierarchical Leiden Clustering

```mermaid
graph TB
    A[Graph] --> B[Build NetworkX Graph]
    B --> C{Use LCC?}
    C -->|Yes| D[Extract Largest Connected Component]
    C -->|No| E[Use Full Graph]
    D --> F[Hierarchical Leiden]
    E --> F
    F --> G[Multiple Levels of Communities]
    G --> H[Aggregate Entity/Relationship IDs]
    H --> I[Build Community Tree]
    I --> J[Output: communities.parquet]
    
    style A fill:#e8f5e9
    style J fill:#c8e6c9
```

**Parameters**:
- `max_cluster_size`: Maximum size per community (default: 10)
- `use_lcc`: Only use largest connected component
- `seed`: Random seed for reproducibility

**Community Schema**:
- `id`: Community identifier
- `level`: Hierarchy level (0 = lowest, higher = more aggregated)
- `title`: Community title
- `entity_ids`: List of entity IDs in community
- `relationship_ids`: List of relationship IDs within community
- `text_unit_ids`: List of relevant text units
- `parent`: Parent community ID
- `children`: List of child community IDs
- `size`: Number of entities

**Ví dụ thực tế**:
```python
# Input: Graph với 10 entities và 15 relationships
# Communities discovered at 2 levels:

# Level 0 (fine-grained):
[
    {
        "id": "comm_001",
        "level": 0,
        "title": "Community 1",
        "entity_ids": ["entity_001", "entity_002", "entity_003"],
        "relationship_ids": ["rel_001", "rel_002"],
        "parent": 1,  # belongs to community 1 at level 1
        "children": [],
        "size": 3
    },
    {
        "id": "comm_002",
        "level": 0,
        "title": "Community 2",
        "entity_ids": ["entity_004", "entity_005"],
        "relationship_ids": ["rel_003"],
        "parent": 1,
        "children": [],
        "size": 2
    }
]

# Level 1 (coarse-grained):
[
    {
        "id": "comm_003",
        "level": 1,
        "title": "Community 1",
        "entity_ids": ["entity_001", "entity_002", "entity_003", "entity_004", "entity_005"],
        "relationship_ids": ["rel_001", "rel_002", "rel_003"],
        "parent": -1,  # top level
        "children": [1, 2],  # communities 1 and 2 from level 0
        "size": 5
    }
]
```

---

### 7. Create Final Text Units
**File**: `graphrag/index/workflows/create_final_text_units.py`

**Mục đích**: Enrichment text units với entity và relationship information

**Output**: Text units được bổ sung với:
- List of entity IDs mentioned
- List of relationship IDs present
- Community assignments

---

### 8. Create Community Reports
**File**: `graphrag/index/workflows/create_community_reports.py`

**Mục đích**: Tạo summaries cho mỗi community bằng LLM

**Quy trình**:
```mermaid
graph TB
    A[Communities + Entities + Relationships] --> B[Build Local Context]
    B --> C[Filter by Token Limit]
    C --> D[LLM Summarization]
    D --> E[Generate Report Fields]
    E --> F[Output: community_reports.parquet]
    
    subgraph "Report Fields"
        G[Title]
        H[Summary]
        I[Full Content]
        J[Findings]
        K[Rating]
    end
    
    E --> G
    E --> H
    E --> I
    E --> J
    E --> K
    
    style A fill:#e8f5e9
    style F fill:#c8e6c9
```

**LLM Strategy**:
1. Collect entities, relationships, và claims trong community
2. Build local context với token budget (max_input_length: 8000)
3. Sử dụng prompts:
   - `community_report_graph.txt` - cho graph-based communities
   - `community_report_text.txt` - cho text-based communities
4. Generate structured report với các sections

**Report Schema**:
- `id`: Report identifier
- `community`: Community ID
- `level`: Hierarchy level
- `title`: Report title
- `summary`: Executive summary (short)
- `full_content`: Detailed report content
- `findings`: List of key findings (JSON)
- `rating`: Importance rating (float)
- `rating_explanation`: Explanation for rating

**Ví dụ thực tế**:
```json
{
    "id": "report_001",
    "community": 1,
    "level": 0,
    "title": "Hệ sinh thái Công ty ABC và Nhà sáng lập",
    "summary": "Community này tập trung vào Công ty ABC, nhà sáng lập Nguyễn Văn A, và các hoạt động kinh doanh phần mềm AI tại Hà Nội.",
    "full_content": "# Tổng quan\n\nCông ty ABC là một tổ chức...",
    "findings": [
        {
            "summary": "Công ty ABC được thành lập năm 2020",
            "explanation": "Nguyễn Văn A là người sáng lập công ty này"
        },
        {
            "summary": "Chuyên môn về AI",
            "explanation": "Công ty tập trung vào phát triển phần mềm AI"
        }
    ],
    "rating": 8.5,
    "rating_explanation": "Community có tầm quan trọng cao do liên quan đến startup công nghệ và nhân vật chủ chốt"
}
```

---

### 9. Generate Text Embeddings
**File**: `graphrag/index/workflows/generate_text_embeddings.py`

**Mục đích**: Tạo vector embeddings cho các text fields để hỗ trợ semantic search

**Các trường được embed** (configurable):
1. `document.text` - Document embeddings
2. `entity.title` - Entity name embeddings
3. `entity.description` - Entity description embeddings
4. `relationship.description` - Relationship embeddings
5. `text_unit.text` - Text unit embeddings
6. `community_report.title` - Report title embeddings
7. `community_report.summary` - Report summary embeddings
8. `community_report.full_content` - Full report embeddings

**Quy trình**:
```mermaid
graph TB
    A[Multiple DataFrames] --> B{For each field}
    B --> C[Batch Text]
    C --> D[Embedding Model]
    D --> E[Generate Vectors]
    E --> F[Store in Vector Store]
    F --> G[Output: embeddings.<field>.parquet]
    
    style A fill:#e8f5e9
    style G fill:#c8e6c9
```

**Embedding Strategy**:
- Model: `text-embedding-3-small` (configurable)
- Batch processing với:
  - `batch_size`: 16 items per batch
  - `batch_max_tokens`: 8191 tokens per batch
- Concurrent requests: 25
- Vector dimension: 1536 (model-dependent)

**Output**: Separate parquet files cho mỗi embedded field
```python
# embeddings.entity.description.parquet
[
    {
        "id": "entity_001",
        "embedding": [0.123, -0.456, 0.789, ...]  # 1536-dim vector
    },
    ...
]
```

**Storage**: Embeddings được lưu vào Vector Store (LanceDB)
- Table per entity type
- Supports similarity search
- Optimized for retrieval

---

## Pipeline Execution Context

### Storage Architecture
```mermaid
graph TB
    A[Input Storage] --> B[Pipeline Run]
    B --> C[Output Storage]
    B --> D[Cache Storage]
    
    C --> E[documents.parquet]
    C --> F[text_units.parquet]
    C --> G[entities.parquet]
    C --> H[relationships.parquet]
    C --> I[communities.parquet]
    C --> J[community_reports.parquet]
    C --> K[embeddings.*]
    
    D --> L[LLM Response Cache]
    D --> M[Embedding Cache]
    
    style A fill:#e8f5e9
    style C fill:#fff9c4
    style D fill:#e1f5fe
```

### Runtime Context
- **Callbacks**: Progress tracking, logging, monitoring
- **Cache**: LLM response caching để tránh duplicate calls
- **Stats**: Performance metrics cho mỗi workflow
- **State**: Shared state between workflows

### Incremental Update Mode

Khi chạy incremental update (`is_update_run=True`):

```mermaid
graph TB
    A[Previous Index] --> B[Backup to previous/]
    C[New Documents] --> D[Run Pipeline]
    D --> E[Delta Index in delta/]
    B --> F[Merge Strategy]
    E --> F
    F --> G[Updated Index]
    
    style A fill:#e8f5e9
    style C fill:#e8f5e9
    style G fill:#c8e6c9
```

**Update Workflows** (chạy sau standard workflows):
1. `update_entities_relationships` - Merge new entities/relationships
2. `update_communities` - Recompute communities
3. `update_community_reports` - Update reports
4. `update_text_embeddings` - Generate embeddings cho new content
5. `update_clean_state` - Cleanup và finalization

---

## Configuration Example

**File**: `settings.yaml`

```yaml
# Chunking
chunks:
  size: 1200          # tokens per chunk
  overlap: 100        # overlap between chunks
  group_by_columns: [id]

# Entity Extraction (LLM-based)
extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 1    # số lần iteration để extract thêm

# Entity Extraction (NLP-based - Fast Pipeline)
extract_graph_nlp:
  normalize_edge_weights: true  # PMI normalization cho weights
  text_analyzer:
    model_name: "en_core_web_md"

# Community Detection
cluster_graph:
  max_cluster_size: 10
  use_lcc: true       # chỉ dùng largest connected component
  seed: 42            # reproducibility

# Community Reports
community_reports:
  model_id: default_chat_model
  max_length: 2000         # max report length
  max_input_length: 8000   # max context for LLM

# Embeddings
embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store
  # Names of fields to embed:
  names:
    - entity.description
    - text_unit.text
    - community_report.summary
```

---

## Performance Considerations

### Bottlenecks
1. **LLM Calls** - Extract graph và community reports
   - Mitigation: Concurrent requests, caching
2. **Embeddings** - Large number of texts
   - Mitigation: Batching, async processing
3. **Graph Clustering** - Large graphs
   - Mitigation: LCC filtering, max_cluster_size

### Optimization Tips
1. **Use Fast Pipeline** - NLP-based extraction thay vì LLM (nhanh hơn 10-20x)
2. **Weight normalization trade-offs**:
   - NLP-based: Statistical accuracy với PMI nhưng có thể miss complex relationships
   - LLM-based: Semantic richness với subjective weights nhưng không statistical
3. **Adjust chunk size** - Balance giữa context và processing speed
4. **Limit entity types** - Giảm complexity của extraction
5. **Cache aggressively** - Reuse LLM responses
6. **Tune concurrency** - Balance giữa speed và rate limits

### Monitoring
```python
# stats.json output after indexing
{
    "total_runtime": 1234.56,
    "num_documents": 100,
    "workflows": {
        "load_input_documents": {"overall": 5.2},
        "create_base_text_units": {"overall": 12.8},
        "extract_graph": {"overall": 856.3},
        "create_communities": {"overall": 45.1},
        "create_community_reports": {"overall": 267.4},
        "generate_text_embeddings": {"overall": 47.6}
    }
}
```

---

## Ví dụ thực tế End-to-End

### Input
**File**: `input/company_profile.txt`
```
Công ty ABC Technology được thành lập vào tháng 3 năm 2020 bởi hai nhà đồng sáng lập 
Nguyễn Văn A và Trần Thị B tại Hà Nội. Công ty chuyên về phát triển các giải pháp 
trí tuệ nhân tạo cho ngành tài chính.

Năm 2021, công ty đã huy động được 5 triệu USD từ quỹ đầu tư XYZ Ventures. 
CEO Nguyễn Văn A cho biết, nguồn vốn này sẽ được sử dụng để mở rộng đội ngũ 
và phát triển sản phẩm mới.

Hiện tại, ABC Technology có văn phòng tại Hà Nội và TP.HCM, với hơn 50 nhân viên.
```

### Processing Steps

**Step 1: Load Documents**
```python
documents = [{
    "id": "doc_001",
    "text": "Công ty ABC Technology được thành lập...",
    "metadata": '{"source": "company_profile.txt"}'
}]
```

**Step 2: Create Text Units** (với chunk_size=200 tokens)
```python
text_units = [
    {
        "id": "chunk_001",
        "text": "Công ty ABC Technology được thành lập vào tháng 3 năm 2020...",
        "n_tokens": 200
    },
    {
        "id": "chunk_002", 
        "text": "Năm 2021, công ty đã huy động được 5 triệu USD...",
        "n_tokens": 180
    }
]
```

**Step 3: Extract Graph**
```python
entities = [
    {"title": "ABC Technology", "type": "organization"},
    {"title": "Nguyễn Văn A", "type": "person"},
    {"title": "Trần Thị B", "type": "person"},
    {"title": "XYZ Ventures", "type": "organization"},
    {"title": "Hà Nội", "type": "geo"},
    {"title": "TP.HCM", "type": "geo"}
]

relationships = [
    {"source": "Nguyễn Văn A", "target": "ABC Technology", "description": "đồng sáng lập"},
    {"source": "Trần Thị B", "target": "ABC Technology", "description": "đồng sáng lập"},
    {"source": "ABC Technology", "target": "Hà Nội", "description": "có trụ sở tại"},
    {"source": "XYZ Ventures", "target": "ABC Technology", "description": "đầu tư vào"}
]
```

**Step 4: Create Communities**
```python
communities_level_0 = [
    {
        "id": 1,
        "title": "Community 1",
        "entity_ids": ["ABC Technology", "Nguyễn Văn A", "Trần Thị B"],
        "level": 0
    },
    {
        "id": 2,
        "title": "Community 2",
        "entity_ids": ["XYZ Ventures", "ABC Technology"],
        "level": 0
    }
]
```

**Step 5: Community Reports**
```json
{
    "title": "Hệ sinh thái ABC Technology",
    "summary": "Công ty công nghệ AI cho tài chính được thành lập bởi 2 founders và nhận đầu tư từ XYZ Ventures",
    "findings": [
        "ABC Technology được thành lập năm 2020 tại Hà Nội",
        "Nhận đầu tư 5 triệu USD năm 2021",
        "Có 2 văn phòng và 50+ nhân viên"
    ],
    "rating": 8.5
}
```

**Step 6: Generate Embeddings**
```python
# Entity embeddings
entity_embeddings = {
    "ABC Technology": [0.123, -0.456, 0.789, ...],  # 1536-dim
    "Nguyễn Văn A": [0.234, -0.567, 0.890, ...],
    ...
}

# Report embeddings
report_embeddings = {
    "community_1": [0.345, -0.678, 0.901, ...]
}
```

### Output Structure
```
output/
├── documents.parquet           # 1 document
├── text_units.parquet          # 2 text units
├── entities.parquet            # 6 entities
├── relationships.parquet       # 4 relationships
├── communities.parquet         # 2 communities (level 0)
├── community_reports.parquet   # 2 reports
├── embeddings.entity.description.parquet
├── embeddings.community_report.summary.parquet
└── stats.json                  # Performance metrics
```

---

## Summary

Pipeline indexing của GraphRAG chuyển đổi raw documents thành một knowledge graph được cấu trúc hóa cao thông qua 9 workflows chính:

1. **Load Documents** → Parse input
2. **Chunk Text** → Split into processable units
3. **Extract Graph** → LLM extracts entities & relationships
4. **Finalize Graph** → Clean and compute metrics
5. **Extract Covariates** → Extract claims/facts (optional)
6. **Create Communities** → Detect entity clusters
7. **Finalize Text Units** → Enrich with graph data
8. **Create Reports** → LLM summarizes communities
9. **Generate Embeddings** → Vector representations for search

Output là một rich knowledge graph với:
- **Entities** với types và descriptions
- **Relationships** với weights
- **Communities** với hierarchical structure
- **Reports** với summaries và findings
- **Embeddings** cho semantic search

Hệ thống hỗ trợ cả full indexing và incremental updates, với extensive caching và monitoring capabilities.

---

## 📤 Chi Tiết Output Schema

Pipeline tạo ra các bảng output dưới dạng **Parquet files**. Tất cả các bảng đều có 2 trường ID chung:

| Trường | Type | Mô tả |
|--------|------|-------|
| `id` | str | UUID được tạo tự động, đảm bảo tính unique toàn cục |
| `human_readable_id` | int | ID ngắn được increment theo run, dễ đọc cho citations |

---

### 📁 **documents.parquet** - Danh sách Documents

| Trường | Type | Mô tả |
|--------|------|-------|
| `title` | str | Tên file hoặc title được cấu hình |
| `text` | str | Nội dung đầy đủ của document |
| `text_unit_ids` | str[] | Danh sách text units (chunks) đã parse từ document |
| `metadata` | dict | Metadata tùy chọn nếu cấu hình khi import CSV |

---

### 📁 **text_units.parquet** - Danh sách Text Chunks

| Trường | Type | Mô tả |
|--------|------|-------|
| `text` | str | Nội dung đầy đủ của chunk |
| `n_tokens` | int | Số tokens trong chunk (thường = `chunk_size`, trừ chunk cuối) |
| `document_ids` | str[] | Danh sách document IDs mà chunk được parse từ đó |
| `entity_ids` | str[] | Danh sách entities được tìm thấy trong text unit |
| `relationship_ids` | str[] | Danh sách relationships được tìm thấy trong text unit |
| `covariate_ids` | str[] | (Optional) Danh sách covariates trong text unit |

---

### 📁 **entities.parquet** - Danh sách Entities

| Trường | Type | Mô tả |
|--------|------|-------|
| `title` | str | Tên của entity |
| `type` | str | Loại entity: "organization", "person", "geo", "event" |
| `description` | str | Mô tả của entity, được LLM tổng hợp từ nhiều text units |
| `text_unit_ids` | str[] | Danh sách text units chứa entity này |
| `frequency` | int | Số lần entity xuất hiện trong các text units |
| `degree` | int | Node degree (số connections trong graph) |
| `x` | float | Vị trí X cho visualization (0 nếu không bật UMAP) |
| `y` | float | Vị trí Y cho visualization (0 nếu không bật UMAP) |

---

### 📁 **relationships.parquet** - Danh sách Relationships (Edge List)

| Trường | Type | Mô tả |
|--------|------|-------|
| `source` | str | Tên source entity |
| `target` | str | Tên target entity |
| `description` | str | Mô tả relationship, được LLM tổng hợp |
| `weight` | float | Trọng số edge: **NLP**=PMI-normalized, **LLM**=sum of strengths |
| `combined_degree` | int | Tổng degree của source và target nodes |
| `text_unit_ids` | str[] | Danh sách text units chứa relationship này |

---

### 📁 **communities.parquet** - Danh sách Communities (Leiden)

| Trường | Type | Mô tả |
|--------|------|-------|
| `community` | int | Leiden community ID (unique qua tất cả levels) |
| `parent` | int | Parent community ID |
| `children` | int[] | Danh sách child community IDs |
| `level` | int | Độ sâu trong hierarchy (0 = chi tiết nhất) |
| `title` | str | Tên thân thiện của community |
| `entity_ids` | str[] | Danh sách entity members |
| `relationship_ids` | str[] | Danh sách relationships hoàn toàn nằm trong community |
| `text_unit_ids` | str[] | Danh sách text units represented trong community |
| `period` | str | Ngày ingest (ISO8601), dùng cho incremental updates |
| `size` | int | Kích thước community (số entities) |

---

### 📁 **community_reports.parquet** - Báo cáo Community

| Trường | Type | Mô tả |
|--------|------|-------|
| `community` | int | Community ID mà report này áp dụng |
| `parent` | int | Parent community ID |
| `children` | int[] | Danh sách child community IDs |
| `level` | int | Level của community |
| `title` | str | LLM-generated title cho report |
| `summary` | str | LLM-generated summary |
| `full_content` | str | LLM-generated full report |
| `rank` | float | LLM-derived relevance ranking dựa trên entity salience |
| `rating_explanation` | str | LLM-derived giải thích về rank |
| `findings` | dict | LLM-derived list của top 5-10 insights (summary + explanation) |
| `full_content_json` | json | Full JSON output từ LLM, cho phép prompt tuning |
| `period` | str | Ngày ingest (ISO8601) |
| `size` | int | Kích thước community |

---

### 📁 **covariates.parquet** - Claims/Covariates (Optional)

*Chỉ được tạo khi `extract_claims.enabled = true`*

| Trường | Type | Mô tả |
|--------|------|-------|
| `covariate_type` | str | Luôn là "claim" với default config |
| `type` | str | Loại claim |
| `description` | str | LLM-generated description của behavior |
| `subject_id` | str | Tên source entity (thực hiện claimed behavior) |
| `object_id` | str | Tên target entity (nhận claimed behavior) |
| `status` | str | LLM-derived assessment: TRUE, FALSE, hoặc SUSPECTED |
| `start_date` | str | LLM-derived ngày bắt đầu hành vi (ISO8601) |
| `end_date` | str | LLM-derived ngày kết thúc hành vi (ISO8601) |
| `source_text` | str | Đoạn text ngắn chứa claimed behavior |
| `text_unit_id` | str | ID của text unit mà claim được extract từ đó |

---

### 📁 **Cấu trúc Output Directory**

```
output/
├── documents.parquet           # Tài liệu gốc với metadata
├── text_units.parquet          # Text chunks với references
├── entities.parquet            # Entities được trích xuất
├── relationships.parquet       # Relationships giữa entities
├── communities.parquet         # Community assignments
├── community_reports.parquet   # LLM-generated summaries
├── covariates.parquet          # (Optional) Claims/Covariates
├── context.json                # Pipeline state
├── stats.json                  # Execution statistics
└── embeddings/                 # Vector embeddings (nếu enabled)
    ├── entity.description.parquet
    ├── text_unit.text.parquet
    └── community_report.summary.parquet
```

---

## Remove Duplicates - Chi tiết

### Stage 1: Merge trong Extract Graph (Aggregation)
Trong bước Extract Graph, duplicates đã được **aggregated**:

**Entities**:
```python
# groupby(["title", "type"]) và aggregate
entities.groupby(["title", "type"]).agg({
    description=("description", list),        # Gộp tất cả descriptions
    text_unit_ids=("source_id", list),         # Gộp tất cả source IDs
    frequency=("source_id", "count")           # Đếm số lần xuất hiện
})
```

**Relationships**:
```python
# groupby(["source", "target"]) và aggregate  
relationships.groupby(["source", "target"]).agg({
    description=("description", list),          # Gộp tất cả descriptions
    text_unit_ids=("source_id", list),         # Gộp tất cả source IDs
    weight=("weight", "sum")                    # SUM tất cả weights
})
```

### Stage 2: Final Deduplication (Cleanup)
Trong Finalize Graph, **final cleanup** được thực hiện:

**Relationships**:
```python
final_relationships = relationships.drop_duplicates(subset=["source", "target"])
# Chỉ giữ lại một dòng cho mỗi (source, target) pair
# Đã có aggregated weights, descriptions, text_unit_ids từ stage 1
```

**Entities**:
```python
final_entities = entities.drop_duplicates(subset="title")
# Chỉ giữ lại một dòng cho mỗi entity title
# Đã có aggregated info từ stage 1
```

### Tại sao cần 2 stages?

#### Standard Pipeline (LLM-based):
1. **Stage 1 (Aggregation)**: Gộp thông tin từ multiple occurrences
2. **Stage 2 (Deduplication)**: **Safety net cho concurrent processing edge cases**

**Nguyên nhân cần deduplication trong Standard Pipeline:**
```python
# extract_graph.py:63-70 - Concurrent processing
results = await derive_from_rows(
    text_units, run_strategy, async_type=async_mode, num_threads=num_threads
)
# → Multiple text units processed in parallel
# → Rare race conditions or edge cases có thể create duplicates

# extract_graph.py:76-77 - Collect results from parallel workers
for result in results:
    entity_dfs.append(pd.DataFrame(result[0]))      # Each worker returns DataFrame
    relationship_dfs.append(pd.DataFrame(result[1]))
```

#### Fast Pipeline (NLP-based):
1. **Stage 1 (Aggregation)**: Gộp thông tin từ multiple occurrences
2. **Intermediate Processing**: `prune_graph` với merge operations có thể introduce duplicates
3. **Stage 2 (Deduplication)**: Final cleanup sau tất cả transforms

**Root cause duplicates trong Fast Pipeline:**
```python
# prune_graph.py:77-80 - Merge operations
subset_entities = pruned_nodes.merge(entities, on="title", how="inner")
subset_relationships = pruned_edges.merge(relationships, on=["source", "target"], how="inner")
# → Merge có thể create duplicates nếu multiple matches
```

#### Summary:
- **Standard Pipeline**: Concurrent processing edge cases → Need safety net
- **Fast Pipeline**: Prune graph merge operations → Can introduce duplicates  
- **Both pipelines**: `drop_duplicates()` là **defensive programming** đảm bảo data integrity

---
