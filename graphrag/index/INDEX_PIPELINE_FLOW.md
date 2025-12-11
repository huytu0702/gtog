# GraphRAG Indexing Pipeline - Flow Documentation

## 📋 Tổng Quan

Module `graphrag/index` chịu trách nhiệm xây dựng **Knowledge Graph** từ dữ liệu văn bản thô. Pipeline này biến đổi documents thành một đồ thị tri thức có cấu trúc, bao gồm entities, relationships, communities và community reports.

---

## 🗂️ Cấu Trúc Thư Mục

```
graphrag/index/
├── __init__.py              # Package root
├── validate_config.py       # Validation cấu hình
├── input/                   # Input loaders (CSV, JSON, Text)
├── operations/              # Các thao tác xử lý dữ liệu
├── run/                     # Pipeline execution
├── text_splitting/          # Text chunking utilities
├── typing/                  # Type definitions
├── update/                  # Incremental update logic
├── utils/                   # Utility functions
└── workflows/               # Workflow definitions
```

---

## 🔄 Luồng Indexing - Standard Pipeline

### Mermaid Diagram - Tổng Quan Luồng

```mermaid
flowchart TB
    subgraph "1. INPUT PHASE"
        A[📄 Raw Documents<br/>CSV/JSON/TXT] --> B[load_input_documents]
        B --> C[(documents.parquet)]
    end

    subgraph "2. TEXT PROCESSING PHASE"
        C --> D[create_base_text_units]
        D --> E[(text_units.parquet)]
        E --> F[create_final_documents]
        F --> G[(documents.parquet<br/>with text_unit_ids)]
    end

    subgraph "3. GRAPH EXTRACTION PHASE"
        E --> H[extract_graph]
        H --> I{LLM Entity<br/>Extraction}
        I --> J[(entities.parquet)]
        I --> K[(relationships.parquet)]
        
        J --> L[finalize_graph]
        K --> L
        L --> M[(entities.parquet<br/>finalized)]
        L --> N[(relationships.parquet<br/>finalized)]
    end

    subgraph "4. COVARIATES PHASE"
        E --> O[extract_covariates]
        O --> P{LLM Claim<br/>Extraction}
        P --> Q[(covariates.parquet)]
    end

    subgraph "5. COMMUNITY DETECTION PHASE"
        M --> R[create_communities]
        N --> R
        R --> S{Hierarchical<br/>Leiden Algorithm}
        S --> T[(communities.parquet)]
    end

    subgraph "6. TEXT UNITS FINALIZATION"
        E --> U[create_final_text_units]
        M --> U
        N --> U
        Q --> U
        U --> V[(text_units.parquet<br/>finalized)]
    end

    subgraph "7. COMMUNITY REPORTS PHASE"
        M --> W[create_community_reports]
        N --> W
        T --> W
        Q --> W
        W --> X{LLM Summary<br/>Generation}
        X --> Y[(community_reports.parquet)]
    end

    subgraph "8. EMBEDDING PHASE"
        G --> Z[generate_text_embeddings]
        V --> Z
        M --> Z
        N --> Z
        Y --> Z
        Z --> AA{Embedding<br/>Model}
        AA --> AB[(Vector Embeddings<br/>for Search)]
    end

    style A fill:#e1f5fe
    style AB fill:#c8e6c9
    style I fill:#fff3e0
    style P fill:#fff3e0
    style S fill:#f3e5f5
    style X fill:#fff3e0
    style AA fill:#fce4ec
```

---

## 📊 Chi Tiết Từng Workflow

### Mermaid Diagram - Pipeline Workflows

```mermaid
graph LR
    subgraph "Standard Pipeline"
        W1[load_input_documents] --> W2[create_base_text_units]
        W2 --> W3[create_final_documents]
        W3 --> W4[extract_graph]
        W4 --> W5[finalize_graph]
        W5 --> W6[extract_covariates]
        W6 --> W7[create_communities]
        W7 --> W8[create_final_text_units]
        W8 --> W9[create_community_reports]
        W9 --> W10[generate_text_embeddings]
    end

    style W1 fill:#bbdefb
    style W4 fill:#ffe0b2
    style W6 fill:#ffe0b2
    style W7 fill:#e1bee7
    style W9 fill:#ffe0b2
    style W10 fill:#f8bbd9
```

---

## 🔍 Mô Tả Chi Tiết Từng Bước

### 1️⃣ Load Input Documents

**File:** `workflows/load_input_documents.py`

**Mục đích:** Đọc và parse dữ liệu đầu vào từ các nguồn khác nhau.

**Input Factory hỗ trợ:**
- `text` - Plain text files (`.txt`)
- `csv` - CSV files with text columns
- `json` - JSON files

```mermaid
flowchart LR
    subgraph "Input Sources"
        TXT[📄 .txt files]
        CSV[📊 .csv files]
        JSON[📋 .json files]
    end
    
    TXT --> Factory[Input Factory]
    CSV --> Factory
    JSON --> Factory
    
    Factory --> DF["pd.DataFrame<br/>[id, text, title, metadata]"]
    DF --> Storage[(documents.parquet)]
```

**Ví dụ cấu trúc output:**

```python
# documents DataFrame
{
    "id": "doc_001",
    "text": "Nội dung văn bản đầy đủ của tài liệu...",
    "title": "Tên tài liệu",
    "metadata": {"author": "John Doe", "date": "2024-01-01"}
}
```

---

### 2️⃣ Create Base Text Units

**File:** `workflows/create_base_text_units.py`

**Mục đích:** Chia nhỏ documents thành các text chunks (text units) để xử lý hiệu quả hơn.

```mermaid
flowchart TB
    subgraph "Chunking Process"
        DOC[📄 Long Document<br/>~10,000 tokens] --> CHUNK[Text Chunking]
        CHUNK --> U1[Chunk 1<br/>~1200 tokens]
        CHUNK --> U2[Chunk 2<br/>~1200 tokens]
        CHUNK --> U3[Chunk 3<br/>~1200 tokens]
        CHUNK --> UN[Chunk N<br/>...]
    end
    
    subgraph "Chunking Strategies"
        S1[tokens - Token-based splitting]
        S2[sentence - Sentence-based splitting]
    end
    
    CHUNK -.-> S1
    CHUNK -.-> S2
```

**Cấu hình chunking:**

```yaml
chunks:
  size: 1200        # Số tokens mỗi chunk
  overlap: 100      # Số tokens overlap giữa các chunks
  strategy: tokens  # tokens hoặc sentence
  encoding_model: cl100k_base
```

**Ví dụ output:**

```python
# text_units DataFrame
{
    "id": "tu_hash_001",
    "text": "Đây là nội dung của text unit đầu tiên...",
    "document_ids": ["doc_001"],
    "n_tokens": 1150
}
```

---

### 3️⃣ Create Final Documents

**File:** `workflows/create_final_documents.py`

**Mục đích:** Cập nhật documents với danh sách text_unit_ids liên quan.

```mermaid
flowchart LR
    D1[Documents] --> JOIN[Join Operation]
    TU[Text Units] --> JOIN
    JOIN --> FD[Final Documents<br/>with text_unit_ids]
```

**Ví dụ output:**

```python
# documents DataFrame (updated)
{
    "id": "doc_001",
    "title": "Tên tài liệu",
    "text": "Nội dung đầy đủ...",
    "text_unit_ids": ["tu_001", "tu_002", "tu_003"],
    "metadata": {...}
}
```

---

### 4️⃣ Extract Graph (Core LLM Operation)

**File:** `workflows/extract_graph.py`

**Mục đích:** Sử dụng LLM để trích xuất entities và relationships từ text.

```mermaid
flowchart TB
    subgraph "Entity & Relationship Extraction"
        TU[Text Unit] --> LLM[🤖 LLM<br/>GPT-4/Claude/etc]
        
        LLM --> ENT["Entities<br/>(Person, Organization, Location, Event)"]
        LLM --> REL["Relationships<br/>(WORKS_FOR, LOCATED_IN, etc)"]
    end
    
    subgraph "Summarization"
        ENT --> SUM1[Description Summarization]
        REL --> SUM2[Description Summarization]
        
        SUM1 --> FENT[Final Entities<br/>with merged descriptions]
        SUM2 --> FREL[Final Relationships<br/>with merged descriptions]
    end
    
    subgraph "Entity Merging"
        E1["Entity: 'Microsoft'<br/>from TU1"]
        E2["Entity: 'Microsoft'<br/>from TU2"]
        E3["Entity: 'Microsoft'<br/>from TU3"]
        
        E1 --> MERGE[Merge by Title]
        E2 --> MERGE
        E3 --> MERGE
        
        MERGE --> EM["Merged Entity<br/>descriptions: [d1, d2, d3]<br/>text_unit_ids: [tu1, tu2, tu3]<br/>frequency: 3"]
    end
```

**Entity Types mặc định:**
- `organization` - Tổ chức, công ty
- `person` - Người
- `geo` - Địa điểm địa lý
- `event` - Sự kiện

**Ví dụ extraction:**

```python
# Input Text Unit
text = """
Microsoft, công ty công nghệ có trụ sở tại Redmond, Washington, 
được thành lập bởi Bill Gates và Paul Allen vào năm 1975.
"""

# Extracted Entities
entities = [
    {"title": "MICROSOFT", "type": "organization", 
     "description": "Công ty công nghệ có trụ sở tại Redmond"},
    {"title": "BILL GATES", "type": "person", 
     "description": "Đồng sáng lập Microsoft"},
    {"title": "PAUL ALLEN", "type": "person", 
     "description": "Đồng sáng lập Microsoft"},
    {"title": "REDMOND", "type": "geo", 
     "description": "Thành phố tại Washington, trụ sở Microsoft"}
]

# Extracted Relationships
relationships = [
    {"source": "MICROSOFT", "target": "REDMOND", 
     "description": "có trụ sở tại", "weight": 1.0},
    {"source": "BILL GATES", "target": "MICROSOFT", 
     "description": "đồng sáng lập", "weight": 1.0},
    {"source": "PAUL ALLEN", "target": "MICROSOFT", 
     "description": "đồng sáng lập", "weight": 1.0}
]
```

---

### 5️⃣ Finalize Graph

**File:** `workflows/finalize_graph.py`

**Mục đích:** Hoàn thiện format của entities và relationships, tính toán các metrics.

```mermaid
flowchart TB
    subgraph "Finalization Steps"
        E[Entities] --> FE[finalize_entities]
        R[Relationships] --> FR[finalize_relationships]
        
        FE --> |Add metrics| EF[Final Entities]
        FR --> |Add metrics| RF[Final Relationships]
        
        subgraph "Added Fields"
            M1[human_readable_id]
            M2[node_degree]
            M3[edge_degree]
            M4[combined_degree]
        end
    end
    
    EF --> GRAPHML[Optional: GraphML Snapshot]
    RF --> GRAPHML
```

**Ví dụ output:**

```python
# entities DataFrame (finalized)
{
    "id": "ent_uuid_001",
    "title": "MICROSOFT",
    "type": "organization",
    "description": "Summarized description...",
    "human_readable_id": 0,
    "text_unit_ids": ["tu_001", "tu_002"],
    "frequency": 5,
    "degree": 12  # Số connections trong graph
}

# relationships DataFrame (finalized)
{
    "id": "rel_uuid_001",
    "source": "MICROSOFT",
    "target": "REDMOND",
    "description": "có trụ sở tại",
    "weight": 3.0,
    "human_readable_id": 0,
    "source_degree": 12,
    "target_degree": 5,
    "combined_degree": 17
}
```

---

### 6️⃣ Extract Covariates (Optional)

**File:** `workflows/extract_covariates.py`

**Mục đích:** Trích xuất claims/covariates từ text (nếu được enable).

```mermaid
flowchart LR
    TU[Text Units] --> LLM[🤖 LLM Claim Extraction]
    LLM --> COV["Covariates/Claims<br/>(subject, type, status, description)"]
    COV --> Storage[(covariates.parquet)]
```

**Ví dụ claim:**

```python
{
    "id": "claim_001",
    "subject": "MICROSOFT",
    "type": "FINANCIAL_CLAIM",
    "status": "TRUE",
    "description": "Microsoft có doanh thu hàng năm trên 100 tỷ USD",
    "text_unit_id": "tu_005"
}
```

---

### 7️⃣ Create Communities

**File:** `workflows/create_communities.py`

**Mục đích:** Phát hiện cộng đồng trong graph sử dụng thuật toán Hierarchical Leiden.

```mermaid
flowchart TB
    subgraph "Community Detection"
        G[Graph<br/>Entities + Relationships] --> LEIDEN[Hierarchical Leiden<br/>Algorithm]
        
        LEIDEN --> L0[Level 0 Communities<br/>Fine-grained]
        LEIDEN --> L1[Level 1 Communities<br/>Medium-grained]
        LEIDEN --> L2[Level 2 Communities<br/>Coarse-grained]
        
        subgraph "Hierarchy Example"
            C0A[Community 0-A<br/>5 entities]
            C0B[Community 0-B<br/>3 entities]
            C0C[Community 0-C<br/>4 entities]
            
            C1A[Community 1-A<br/>8 entities]
            C1B[Community 1-B<br/>4 entities]
            
            C2A[Community 2-A<br/>12 entities]
            
            C0A --> C1A
            C0B --> C1A
            C0C --> C1B
            C1A --> C2A
            C1B --> C2A
        end
    end
```

**Cấu hình:**

```yaml
cluster_graph:
  max_cluster_size: 10  # Kích thước cluster tối đa
  use_lcc: true         # Sử dụng Largest Connected Component
  seed: 0xDEADBEEF      # Random seed for reproducibility
```

**Ví dụ output:**

```python
# communities DataFrame
{
    "id": "comm_uuid_001",
    "community": 0,
    "level": 0,
    "title": "Community 0",
    "parent": -1,
    "children": [1, 2, 3],
    "entity_ids": ["ent_001", "ent_002", "ent_003"],
    "relationship_ids": ["rel_001", "rel_002"],
    "text_unit_ids": ["tu_001", "tu_002"],
    "size": 3
}
```

---

### 8️⃣ Create Final Text Units

**File:** `workflows/create_final_text_units.py`

**Mục đích:** Cập nhật text units với references đến entities, relationships, và covariates.

```mermaid
flowchart LR
    TU[Text Units] --> JOIN1[+ entity_ids]
    E[Entities] --> JOIN1
    
    JOIN1 --> JOIN2[+ relationship_ids]
    R[Relationships] --> JOIN2
    
    JOIN2 --> JOIN3[+ covariate_ids]
    COV[Covariates] --> JOIN3
    
    JOIN3 --> FTU[Final Text Units]
```

**Ví dụ output:**

```python
# text_units DataFrame (finalized)
{
    "id": "tu_001",
    "text": "Nội dung text unit...",
    "document_ids": ["doc_001"],
    "n_tokens": 1150,
    "entity_ids": ["ent_001", "ent_002"],
    "relationship_ids": ["rel_001"],
    "covariate_ids": ["claim_001"]
}
```

---

### 9️⃣ Create Community Reports

**File:** `workflows/create_community_reports.py`

**Mục đích:** Sử dụng LLM để tạo báo cáo tổng hợp cho mỗi community.

```mermaid
flowchart TB
    subgraph "Context Building"
        E[Entities in Community]
        R[Relationships in Community]
        C[Claims in Community]
        
        E --> CONTEXT[Build Local Context]
        R --> CONTEXT
        C --> CONTEXT
    end
    
    subgraph "Report Generation"
        CONTEXT --> LLM[🤖 LLM Summary]
        
        LLM --> REPORT["Community Report<br/>- Title<br/>- Summary<br/>- Full Content<br/>- Rating<br/>- Findings"]
    end
    
    subgraph "Hierarchical Processing"
        L0[Level 0 Communities] --> PROC[Process by Level]
        L1[Level 1 Communities] --> PROC
        L2[Level 2 Communities] --> PROC
        
        PROC --> |Sub-report inheritance| FINAL[Final Reports]
    end
```

**Ví dụ output:**

```python
# community_reports DataFrame
{
    "id": "report_001",
    "community": 0,
    "level": 0,
    "title": "Microsoft Technology Ecosystem",
    "summary": "Cộng đồng này tập trung vào Microsoft và các mối quan hệ...",
    "full_content": "# Microsoft Technology Ecosystem\n\n## Overview\n...",
    "rating": 8.5,
    "rating_explanation": "High impact due to major tech company involvement",
    "findings": [
        "Microsoft là công ty công nghệ lớn có trụ sở tại Redmond",
        "Bill Gates và Paul Allen là những người sáng lập",
        ...
    ]
}
```

---

### 🔟 Generate Text Embeddings

**File:** `workflows/generate_text_embeddings.py`

**Mục đích:** Tạo vector embeddings cho tìm kiếm semantic.

```mermaid
flowchart TB
    subgraph "Embedding Targets"
        D[📄 Documents - text]
        TU[📝 Text Units - text]
        E[👤 Entities - title, description]
        R[🔗 Relationships - description]
        CR[📊 Community Reports - title, summary, full_content]
    end
    
    subgraph "Embedding Process"
        D --> EMB[Embedding Model<br/>OpenAI/Local]
        TU --> EMB
        E --> EMB
        R --> EMB
        CR --> EMB
        
        EMB --> VEC[Vector Store<br/>LanceDB/FAISS/etc]
    end
    
    subgraph "Embedding Types"
        ET[entity_title_embedding]
        ED[entity_description_embedding]
        TUT[text_unit_text_embedding]
        CT[community_title_embedding]
        CS[community_summary_embedding]
        CF[community_full_content_embedding]
    end
```

**Cấu hình embeddings:**

```yaml
embed_text:
  enabled: true
  model_id: text-embedding-ada-002
  names:
    - entity_description_embedding
    - text_unit_text_embedding
    - community_summary_embedding
```

---

## ⚡ Fast Pipeline

Fast pipeline sử dụng NLP extraction thay vì LLM, nhanh hơn nhưng kém chính xác hơn.

```mermaid
graph LR
    subgraph "Fast Pipeline"
        F1[load_input_documents] --> F2[create_base_text_units]
        F2 --> F3[create_final_documents]
        F3 --> F4[extract_graph_nlp]
        F4 --> F5[prune_graph]
        F5 --> F6[finalize_graph]
        F6 --> F7[create_communities]
        F7 --> F8[create_final_text_units]
        F8 --> F9[create_community_reports_text]
        F9 --> F10[generate_text_embeddings]
    end
    
    style F4 fill:#c8e6c9
    style F5 fill:#c8e6c9
    style F9 fill:#c8e6c9
```

**Khác biệt với Standard:**
- `extract_graph_nlp` - Sử dụng NLP thay vì LLM
- `prune_graph` - Loại bỏ noise từ NLP extraction
- `create_community_reports_text` - Text-based reports thay vì LLM

---

## 🔄 Incremental Update Pipeline

Cho phép cập nhật index mà không cần rebuild từ đầu.

```mermaid
graph LR
    subgraph "Update Pipeline"
        U1[load_update_documents] --> U2[Standard Workflows...]
        U2 --> U3[update_final_documents]
        U3 --> U4[update_entities_relationships]
        U4 --> U5[update_text_units]
        U5 --> U6[update_covariates]
        U6 --> U7[update_communities]
        U7 --> U8[update_community_reports]
        U8 --> U9[update_text_embeddings]
        U9 --> U10[update_clean_state]
    end
    
    style U3 fill:#fff3e0
    style U4 fill:#fff3e0
    style U5 fill:#fff3e0
    style U6 fill:#fff3e0
    style U7 fill:#fff3e0
    style U8 fill:#fff3e0
    style U9 fill:#fff3e0
```

---

## 📤 Output Files

Sau khi hoàn thành indexing, các files sau được tạo:

```
output/
├── documents.parquet          # Tài liệu gốc với metadata
├── text_units.parquet         # Text chunks với references
├── entities.parquet           # Entities được trích xuất
├── relationships.parquet      # Relationships giữa entities
├── communities.parquet        # Community assignments
├── community_reports.parquet  # LLM-generated summaries
├── covariates.parquet         # Claims/Covariates (nếu enabled)
├── context.json               # Pipeline state
└── stats.json                 # Execution statistics
```

---

## 🛠️ Ví Dụ Chạy Pipeline

### Command Line

```bash
# Standard indexing
graphrag index --root ./my-project

# Fast indexing (NLP-based)
graphrag index --root ./my-project --method fast

# Incremental update
graphrag index --root ./my-project --update

# With verbose logging
graphrag index --root ./my-project --verbose
```

### Python API

```python
import asyncio
from graphrag.index import run_pipeline
from graphrag.config import GraphRagConfig
from graphrag.index.workflows import PipelineFactory

async def main():
    config = GraphRagConfig.from_file("settings.yaml")
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(
        config, 
        method="standard"  # or "fast", "update"
    )
    
    # Run pipeline
    async for result in run_pipeline(pipeline, config, callbacks):
        print(f"Completed: {result.workflow}")

asyncio.run(main())
```

---

## 📊 Data Flow Summary

```mermaid
flowchart TB
    subgraph "Data Transformation"
        RAW["📄 Raw Text<br/>Unstructured"] 
        --> CHUNKS["📝 Text Units<br/>Chunked"]
        --> GRAPH["🔷 Knowledge Graph<br/>Entities + Relationships"]
        --> COMMUNITIES["🔶 Communities<br/>Clustered"]
        --> REPORTS["📊 Reports<br/>Summarized"]
        --> VECTORS["🧮 Embeddings<br/>Searchable"]
    end
    
    style RAW fill:#ffcdd2
    style CHUNKS fill:#fff9c4
    style GRAPH fill:#c8e6c9
    style COMMUNITIES fill:#bbdefb
    style REPORTS fill:#e1bee7
    style VECTORS fill:#d1c4e9
```

---

## 🔗 Liên Kết Giữa Các Thành Phần

```mermaid
erDiagram
    DOCUMENT ||--o{ TEXT_UNIT : contains
    TEXT_UNIT ||--o{ ENTITY : mentions
    TEXT_UNIT ||--o{ RELATIONSHIP : mentions
    TEXT_UNIT ||--o{ COVARIATE : has
    
    ENTITY ||--o{ RELATIONSHIP : source_or_target
    ENTITY }o--o{ COMMUNITY : belongs_to
    
    COMMUNITY ||--o| COMMUNITY_REPORT : has
    COMMUNITY ||--o{ COMMUNITY : parent_child
    
    ENTITY ||--o| EMBEDDING : has
    TEXT_UNIT ||--o| EMBEDDING : has
    COMMUNITY_REPORT ||--o| EMBEDDING : has
```

---

## 📝 Kết Luận

GraphRAG Indexing Pipeline là một hệ thống phức tạp nhưng được thiết kế module hóa cao. Mỗi workflow đảm nhận một nhiệm vụ cụ thể và có thể được tùy chỉnh hoặc thay thế. Pipeline hỗ trợ cả:

1. **Standard Mode** - Sử dụng LLM cho extraction chính xác
2. **Fast Mode** - Sử dụng NLP cho tốc độ
3. **Update Mode** - Cập nhật incremental hiệu quả

Output cuối cùng là một Knowledge Graph hoàn chỉnh với embeddings, sẵn sàng cho các phương thức query như Local Search, Global Search, và ToG Search.
