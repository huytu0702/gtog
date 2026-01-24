# 3.2. Thiết kế kiến trúc tổng thể

Hệ thống được thiết kế theo kiến trúc 3 tầng (three-tier architecture), đảm bảo sự tách biệt và chuyên môn hóa vai trò của từng thành phần.

## Sơ đồ kiến trúc tổng thể

```mermaid
flowchart TB
    subgraph User["Người dùng"]
        Browser["Trình duyệt Web"]
    end

    subgraph Frontend["Frontend - Next.js 16 / React 19"]
        UI["Giao diện người dùng"]
        APIClient["API Client<br/>(Axios + TanStack Query)"]
    end

    subgraph Backend["Backend - FastAPI"]
        MainApp["FastAPI App"]
        subgraph Routers["API Routers"]
            R1["Collections"]
            R2["Documents"]
            R3["Indexing"]
            R4["Search"]
        end
        subgraph Services["Service Layer"]
            S1["Storage Service"]
            S2["Indexing Service"]
            S3["Query Service"]
        end
    end

    subgraph Core["Lõi xử lý - GraphRAG Library"]
        API["graphrag.api"]
        
        subgraph IndexPipeline["Indexing Pipeline"]
            P1["Phase 1: Compose TextUnits"]
            P2["Phase 2: Graph Extraction"]
            P3["Phase 3: Graph Augmentation"]
            P4["Phase 4: Community Summarization"]
            P5["Phase 5: Document Processing"]
            P6["Phase 6: Network Visualization"]
            P7["Phase 7: Text Embeddings"]
        end
        
        subgraph QueryEngine["Query Engine"]
            GlobalSearch["Global Search"]
            LocalSearch["Local Search"]
            DriftSearch["DRIFT Search"]
            ToGSearch["ToG Search"]
        end
    end

    subgraph Storage["Data Storage"]
        KnowledgeModel[("Knowledge Model<br/>(Parquet Files)")]
    end

    subgraph LLM["LLM Provider"]
        OpenAI["OpenAI API"]
    end

    Browser --> UI
    UI --> APIClient
    APIClient -->|"HTTP REST"| MainApp
    
    MainApp --> Routers
    R1 & R2 --> S1
    R3 --> S2
    R4 --> S3
    
    S2 --> API
    S3 --> API
    
    API --> IndexPipeline
    API --> QueryEngine
    
    IndexPipeline --> KnowledgeModel
    IndexPipeline --> OpenAI
    
    QueryEngine --> KnowledgeModel
    QueryEngine --> OpenAI
```

**Hình 3.1. Sơ đồ kiến trúc tổng thể của hệ thống GraphRAG với ToG Enhancement**

## Các thành phần chính

| Thành phần | Công nghệ | Mô tả |
|------------|-----------|-------|
| **Frontend** | Next.js 16, React 19, TanStack Query, Axios | Giao diện người dùng |
| **Backend** | FastAPI, Python | REST API wrapper |
| **Lõi GraphRAG** | graphrag library | Engine xây dựng đồ thị tri thức và truy vấn |
| **Storage** | Parquet Files | Lưu trữ Knowledge Model |
| **LLM Provider** | OpenAI API | GPT-4 và text-embedding-3-small |

---

## GraphRAG Knowledge Model

Knowledge Model là đặc tả dữ liệu đầu ra, bao gồm các loại thực thể sau:

| Thực thể | Mô tả |
|----------|-------|
| **Document** | Tài liệu đầu vào (file .txt hoặc row trong CSV) |
| **TextUnit** | Đoạn văn bản để phân tích (chunks) |
| **Entity** | Thực thể được trích xuất (người, địa điểm, sự kiện, tổ chức) |
| **Relationship** | Mối quan hệ giữa hai thực thể |
| **Covariate** | Thông tin claim với time-bound |
| **Community** | Cụm phân cấp từ hierarchical community detection |
| **Community Report** | Báo cáo tóm tắt nội dung mỗi community |

---

## Indexing Pipeline - 7 Phases

```mermaid
flowchart TB
    subgraph phase1["Phase 1: Compose TextUnits"]
        documents[Documents] --> chunk[Chunk]
        chunk --> textUnits[Text Units]
    end
    subgraph phase2["Phase 2: Graph Extraction"]
        textUnits --> graph_extract["Entity & Relationship Extraction"]
        graph_extract --> graph_summarize["Entity & Relationship Summarization"]
        graph_summarize --> claim_extraction["Claim Extraction"]
        claim_extraction --> graph_outputs["Graph Tables"]
    end
    subgraph phase3["Phase 3: Graph Augmentation"]
        graph_outputs --> community_detect["Community Detection<br/>(Leiden Algorithm)"]
        community_detect --> community_outputs["Communities Table"]
    end
    subgraph phase4["Phase 4: Community Summarization"]
        community_outputs --> summarized_communities["Community Summarization"]
        summarized_communities --> community_report_outputs["Community Reports Table"]
    end
    subgraph phase5["Phase 5: Document Processing"]
        documents --> link_to_text_units["Link to TextUnits"]
        textUnits --> link_to_text_units
        link_to_text_units --> document_outputs["Documents Table"]
    end
    subgraph phase6["Phase 6: Network Visualization"]
        graph_outputs --> graph_embed["Graph Embedding<br/>(Node2Vec)"]
        graph_embed --> umap_entities["UMAP Dimensionality Reduction"]
        umap_entities --> combine_nodes["Final Entities"]
    end
    subgraph phase7["Phase 7: Text Embeddings"]
        textUnits --> text_embed["Text Embedding"]
        graph_outputs --> description_embed["Description Embedding"]
        community_report_outputs --> content_embed["Content Embedding"]
    end
```

### Chi tiết từng Phase

#### Phase 1: Compose TextUnits
- **Input**: Documents (file .txt hoặc CSV)
- **Process**: Chia tài liệu thành các TextUnits (chunks)
- **Output**: TextUnits table
- **Config**: Chunk size mặc định 300 tokens, có thể cấu hình lên 1200 tokens

```mermaid
flowchart LR
    doc1["Document 1"] --> tu1["TextUnit 1"]
    doc1 --> tu2["TextUnit 2"]
    doc2["Document 2"] --> tu3["TextUnit 3"]
    doc2 --> tu4["TextUnit 4"]
```

#### Phase 2: Graph Extraction
- **Entity & Relationship Extraction**: Trích xuất thực thể với title, type, description và quan hệ với source, target, description
- **Summarization**: Tóm tắt các description trùng lặp thành một description duy nhất
- **Claim Extraction** (optional): Trích xuất claims với time-bounds

```mermaid
flowchart LR
    tu["TextUnit"] --> ge["Graph Extraction"] --> gs["Graph Summarization"]
    tu --> ce["Claim Extraction"]
```

#### Phase 3: Graph Augmentation
- **Community Detection**: Sử dụng thuật toán **Hierarchical Leiden** để phân cụm thực thể
- **Output**: Communities table với cấu trúc phân cấp

```mermaid
flowchart LR
    cd["Leiden Hierarchical Community Detection"] --> ag["Graph Tables"]
```

#### Phase 4: Community Summarization
- **Generate Community Reports**: Tạo báo cáo tóm tắt cho mỗi community bằng LLM
- **Summarize**: Tóm tắt ngắn gọn để sử dụng downstream

```mermaid
flowchart LR
    sc["Generate Community Reports"] --> ss["Summarize Community Reports"] --> co["Community Reports Table"]
```

#### Phase 5: Document Processing
- Liên kết Documents với TextUnits
- Export Documents table

```mermaid
flowchart LR
    aug["Augment"] --> dp["Link to TextUnits"] --> dg["Documents Table"]
```

#### Phase 6: Network Visualization (Optional)
- **Graph Embedding**: Tạo vector representation bằng **Node2Vec**
- **UMAP**: Giảm chiều xuống 2D để visualization

```mermaid
flowchart LR
    ag["Graph Table"] --> ge["Node2Vec Graph Embedding"] --> ne["UMAP Entities"] --> ng["Entities Table"]
```

#### Phase 7: Text Embeddings
- Tạo embeddings cho: TextUnits, Entity descriptions, Community reports
- Ghi vào vector store

```mermaid
flowchart LR
    textUnits["Text Units"] --> text_embed["Text Embedding"]
    graph_outputs["Graph Tables"] --> description_embed["Description Embedding"]
    community_report_outputs["Community Reports"] --> content_embed["Content Embedding"]
```

---

## Query Engine - 4 Phương thức Search

### So sánh các phương thức

| Phương thức | Approach | Depth | Transparency | Best For |
|-------------|----------|-------|--------------|----------|
| **Global Search** | Community-level aggregation (Map-Reduce) | Shallow | Low | Dataset overview, holistic questions |
| **Local Search** | Entity neighborhood | Shallow | Medium | Specific entity details |
| **DRIFT Search** | Adaptive local + global + community | Medium | Medium | Dynamic exploration |
| **ToG Search** | Iterative graph traversal (Beam Search) | Deep | High | Complex reasoning, multi-hop |

### 1. Global Search

Tìm kiếm toàn cục bằng cách tổng hợp thông tin từ tất cả community reports theo kiểu **map-reduce**.

**Use case**: Câu hỏi cần hiểu tổng quan về toàn bộ dataset
- "What are the most significant values of the herbs mentioned?"
- "Summarize the main themes in this document collection"

### 2. Local Search

Kết hợp dữ liệu từ knowledge graph với text chunks để trả lời câu hỏi về thực thể cụ thể.

**Use case**: Câu hỏi về thực thể cụ thể
- "What are the healing properties of chamomile?"
- "Who is John Smith and what is his role?"

### 3. DRIFT Search

Mở rộng Local Search bằng cách kết hợp thông tin community, tăng độ đa dạng của facts trong câu trả lời.

**Use case**: Exploration động, câu hỏi cần context rộng hơn
- Complex queries requiring both specific and contextual information

### 4. ToG (Think-on-Graph) Search

Thuật toán suy luận sâu thông qua **iterative graph traversal** với **beam search**.

```mermaid
flowchart TD
    A["User Query"] --> B["Find Starting Entities"]
    B --> C["Initialize Search State"]
    C --> D{"Current Depth < Max Depth?"}
    D -->|Yes| E["Get Current Frontier"]
    E --> F["For Each Node in Frontier"]
    F --> G["Get Relations for Entity"]
    G --> H["Score Relations using LLM"]
    H --> I["Keep Top-K Relations"]
    I --> J["Create New Exploration Nodes"]
    J --> K["Add to Next Level"]
    K --> L{"More Nodes in Frontier?"}
    L -->|Yes| F
    L -->|No| M["Prune to Beam Width"]
    M --> N["Check Early Termination"]
    N --> O{"Can Answer Confidently?"}
    O -->|Yes| P["Generate Early Answer"]
    O -->|No| Q["Increment Depth"]
    Q --> D
    D -->|No| R["Collect All Exploration Paths"]
    R --> S["Generate Final Answer"]
    S --> T["Return Answer with Reasoning"]
    P --> T
```

**Các phase của ToG:**

1. **Initialization**: Tìm entities liên quan đến query, tạo root nodes
2. **Exploration**: Duyệt relations, LLM scoring, giữ top-K
3. **Pruning**: Beam search giới hạn số paths song song
4. **Reasoning**: Tổng hợp paths, sinh câu trả lời có evidence

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 3 | Beam width - số paths song song |
| `depth` | 3 | Độ sâu exploration tối đa |
| `prune_strategy` | "llm" | Phương thức scoring (llm/semantic) |
| `num_retain_entity` | 5 | Số relations giữ lại mỗi entity |

**Ưu điểm ToG:**
- **Deep Reasoning**: Khám phá nhiều hops trong knowledge graph
- **Evidence-Based**: Câu trả lời có exploration paths minh bạch
- **Transparent**: User thấy được quá trình suy luận
- **Controlled**: Beam search ngăn exponential growth

---

## Luồng dữ liệu chi tiết

### Luồng 1: Indexing

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant FE as Frontend
    participant BE as Backend
    participant GR as GraphRAG
    participant LLM as OpenAI API
    participant DB as Storage

    User->>FE: 1. Upload tài liệu
    FE->>BE: 2. POST /api/collections/{id}/documents
    BE->>DB: 3. Lưu file vào collection
    BE-->>FE: 4. Xác nhận upload
    
    User->>FE: 5. Bấm "Start Indexing"
    FE->>BE: 6. POST /api/collections/{id}/index
    BE->>GR: 7. Gọi graphrag.api.build_index()
    
    rect rgb(40, 40, 40)
        Note over GR,LLM: Phase 1-7: Indexing Pipeline
        GR->>GR: Phase 1: Compose TextUnits
        GR->>LLM: Phase 2: Entity & Relationship Extraction
        LLM-->>GR: Entities, Relationships
        GR->>GR: Phase 3: Community Detection (Leiden)
        GR->>LLM: Phase 4: Community Summarization
        LLM-->>GR: Community Reports
        GR->>GR: Phase 5: Document Processing
        GR->>GR: Phase 6: Network Visualization
        GR->>LLM: Phase 7: Text Embeddings
        LLM-->>GR: Vector embeddings
    end
    
    GR->>DB: 8. Lưu Parquet files (Knowledge Model)
    GR-->>BE: 9. Indexing completed
    BE-->>FE: 10. Status: completed
    FE-->>User: 11. Hiển thị "Indexed"
```

### Luồng 2: Query

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant FE as Frontend
    participant BE as Backend
    participant QS as Query Service
    participant GR as graphrag.api
    participant LLM as OpenAI API
    participant DB as Storage

    User->>FE: 1. Nhập câu hỏi + chọn phương thức
    FE->>BE: 2. POST /api/collections/{id}/search/{method}
    BE->>QS: 3. Chuyển đến Query Service
    
    QS->>DB: 4. Load Parquet files
    DB-->>QS: 5. entities, relationships, communities, text_units

    QS->>GR: 6. Gọi api.{method}_search()
    
    alt Global Search
        GR->>LLM: Map-reduce trên community reports
    else Local Search
        GR->>LLM: Entity-centric retrieval + text chunks
    else DRIFT Search
        GR->>LLM: Local search + community information
    else ToG Search
        loop Beam Search Exploration
            GR->>LLM: Score relations
            GR->>GR: Prune to beam width
        end
        GR->>LLM: Generate answer with reasoning chains
    end
    
    LLM-->>GR: 7. Generated response
    GR-->>QS: 8. (response, context_data)
    QS-->>BE: 9. SearchResponse
    BE-->>FE: 10. JSON response
    FE-->>User: 11. Hiển thị kết quả
```

---

## API Endpoints

### Collections API
```
GET    /api/collections              # Liệt kê collections
POST   /api/collections              # Tạo collection mới
GET    /api/collections/{id}         # Lấy thông tin collection
DELETE /api/collections/{id}         # Xóa collection
```

### Documents API
```
GET    /api/collections/{id}/documents           # Liệt kê tài liệu
POST   /api/collections/{id}/documents           # Upload tài liệu
DELETE /api/collections/{id}/documents/{name}    # Xóa tài liệu
```

### Indexing API
```
POST   /api/collections/{id}/index    # Bắt đầu indexing
GET    /api/collections/{id}/index    # Lấy trạng thái indexing
```

### Search API
```
POST   /api/collections/{id}/search/global   # Global search
POST   /api/collections/{id}/search/local    # Local search
POST   /api/collections/{id}/search/tog      # ToG search
POST   /api/collections/{id}/search/drift    # DRIFT search
```

---

## Công nghệ sử dụng

### Frontend
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| Next.js | 16.0.7 | React framework |
| React | 19.2.0 | UI library |
| TanStack Query | 5.90.11 | Server state management |
| Axios | 1.13.2 | HTTP client |
| Tailwind CSS | 4 | Styling |

### Backend
| Thư viện | Mục đích |
|----------|----------|
| FastAPI | Web framework |
| Uvicorn | ASGI server |
| Pandas | Data manipulation |

### GraphRAG Core
| Thành phần | Mục đích |
|------------|----------|
| OpenAI GPT-4 | LLM cho extraction và reasoning |
| text-embedding-3-small | Vector embeddings |
| Leiden Algorithm | Hierarchical community detection |
| Node2Vec | Graph embedding |
| UMAP | Dimensionality reduction |
| Parquet | Data storage format |
