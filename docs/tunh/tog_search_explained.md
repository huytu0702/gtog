# Giải thích ToG Search — Cho trẻ 5 tuổi cũng hiểu được

> **Đây là tài liệu giải thích cách mình xây dựng ToG (Think-on-Graph) Search trong GraphRAG.**  
> Viết theo kiểu "bạn thân giải thích", kèm ví dụ thực tế và sơ đồ ASCII.

---

## 🧠 ToG Search là gì? (Giải thích siêu đơn giản)

Tưởng tượng bạn đang chơi **trò tìm kho báu** trong một thành phố lớn.

- **Thành phố** = Knowledge Graph (đồ thị tri thức)
- **Các tòa nhà** = Entities (thực thể: người, tổ chức, địa điểm, sự kiện)
- **Các con đường nối giữa tòa nhà** = Relationships (mối quan hệ)
- **Câu hỏi của bạn** = Manh mối để tìm kho báu
- **AI (LLM)** = Người hướng dẫn thông minh giúp bạn chọn đường đi

ToG Search sẽ **leo từng bước qua đồ thị**, mỗi bước hỏi AI "đường này có dẫn đến câu trả lời không?", và cuối cùng **tổng hợp câu trả lời** từ tất cả đường đi tìm được.

---

## 📁 Cấu trúc file — Mỗi file làm 1 việc

```
tog_search/
├── search.py       ← Trưởng nhóm, điều phối toàn bộ
├── state.py        ← Bộ nhớ, ghi nhớ mình đang ở đâu
├── exploration.py  ← Thám tử, đi khám phá bản đồ
├── pruning.py      ← Người phán xét, chọn đường nào đáng đi
└── reasoning.py    ← Nhà thông thái, đọc kết quả và trả lời
```

---

## 🗂️ Giải thích từng file

### 1. `state.py` — Bộ nhớ của cuộc tìm kiếm

> Như tờ giấy ghi chép khi chơi tìm kho báu: "Mình đang ở đây, đã qua những chỗ này, điểm số là bao nhiêu."

```python
@dataclass
class ExplorationNode:
    entity_id: str          # Tòa nhà đang đứng (vd: "MICROSOFT")
    entity_name: str        # Tên thật của nó
    entity_description: str # Mô tả ngắn
    depth: int              # Mình đã đi bao nhiêu bước từ đầu
    score: float            # Điểm "hữu ích" của node này (0.0 → 1.0)
    parent: ExplorationNode # Tòa nhà trước đó
    relation_from_parent: str # Con đường nào dẫn mình đến đây
```

```python
@dataclass
class ToGSearchState:
    query: str                          # Câu hỏi gốc
    current_depth: int                  # Đang ở bước thứ mấy
    nodes_by_depth: Dict[int, List[...]] # Bản đồ: bước X có những node nào
    beam_width: int                     # Chỉ giữ tối đa N đường tốt nhất
    max_depth: int                      # Tối đa đi bao nhiêu bước
```

**Hàm quan trọng:**
- `get_current_frontier()` → "Mình đang đứng ở tòa nhà nào?"
- `prune_current_frontier()` → "Quá nhiều đường, chỉ giữ lại TOP-K tốt nhất."

---

### 2. `exploration.py` — Thám tử khám phá đồ thị

> Như người cầm bản đồ, nhìn xem từ tòa nhà hiện tại có thể đi đến đâu.

**Khởi tạo:** Xây dựng 2 danh sách kề (adjacency lists):
```python
self.outgoing = {}   # A → [B, C, D]   (A kết nối đến ai)
self.incoming = {}   # B → [A, E]       (ai kết nối đến B)
```

**Hàm quan trọng:**

```python
async def find_starting_entities_semantic(query, top_k=3):
    # Chuyển câu hỏi thành vector số (embedding)
    # So sánh với vector của TẤT CẢ entity trong graph
    # Lấy top-K entity giống nhau nhất
    # → Đây là "cửa vào" của hành trình
```
*(Nếu không có embedding model → dùng keyword matching làm fallback)*

```python
def get_relations(entity_id, bidirectional=True):
    # Lấy tất cả các con đường từ tòa nhà này
    # Gồm cả đường đi ra (outgoing) và đường đi vào (incoming)
    # Returns: [(mô_tả_quan_hệ, entity_đích, hướng, trọng_số), ...]
```

```python
def get_text_units_for_nodes(nodes):
    # Thu thập các đoạn văn bản gốc liên quan đến các node đã đi qua
    # Dùng để "trích dẫn nguồn" trong câu trả lời cuối
```

---

### 3. `pruning.py` — Người phán xét (Chọn đường nào đáng đi)

> Từ tòa nhà hiện tại có 10 con đường, nhưng chỉ đi được 3. AI sẽ chấm điểm và chọn.

Có **3 chiến lược** chấm điểm:

#### 🧠 `LLMPruning` — Hỏi AI để chấm điểm

```python
async def score_relations(query, entity_name, relations):
    # Gửi prompt cho LLM: "Câu hỏi là X, entity là Y,
    # có các quan hệ sau: 1. ..., 2. ..., 3. ...
    # Hãy chấm điểm 1-10 cho mỗi quan hệ"
    # Parse kết quả: [8, 3, 7, ...]

async def score_entities(query, current_path, entities):
    # Tương tự nhưng chấm cho entity đích
    # "Đường này dẫn đến A, B, C — cái nào hữu ích nhất?"
```

Cách parse kết quả LLM (xử lý output không ổn định):
```python
def _parse_scores(response, expected_count):
    # Tìm pattern [8, 3, 7] trong text
    # Nếu không có → tìm số nào trong text
    # Clamp về range 1-10
    # Thiếu score → pad bằng 5.0 (trung bình)
```

#### 📐 `SemanticPruning` — Dùng toán học vector (không cần gọi LLM)

```python
async def score_relations(query, entity_name, relations):
    # Embed query thành vector
    # Embed mô tả của từng quan hệ thành vector
    # Tính cosine similarity
    # Score = (similarity + 1) * 5  → scale về 1-10
```

#### 📝 `BM25Pruning` — Khớp từ khóa kiểu cũ (nhanh, không cần AI)

```python
def _compute_bm25_scores(query, documents):
    # Thuật toán BM25 cổ điển (như search engine thập niên 90s)
    # Đếm từ khóa xuất hiện bao nhiêu lần
    # Tính IDF (từ hiếm thì giá trị hơn)
    # Không cần gọi LLM, không cần embedding → siêu nhanh
```

---

### 4. `reasoning.py` — Nhà thông thái tổng hợp câu trả lời

> Sau khi thám tử tìm được các đường đi, nhà thông thái đọc toàn bộ và viết câu trả lời.

**Hàm quan trọng:**

```python
async def check_early_termination(query, frontier_nodes, ...):
    # Hỏi LLM: "Với thông tin hiện có, mày đủ trả lời chưa?"
    # Nếu đủ → trả về (True, answer)
    # Nếu chưa → (False, None) → tiếp tục đào sâu hơn
```

```python
async def generate_answer(query, all_paths, ...):
    # Tổng hợp TẤT CẢ đường đi thành một đoạn context
    # Gửi cho LLM: "Dựa vào các path này, trả lời câu hỏi X"
    # Kết quả kèm theo các path đã dùng (để trace lại)
```

---

### 5. `search.py` — Trưởng nhóm điều phối

> Đây là file chính, nó gọi tất cả các module trên theo đúng thứ tự.

**Khởi tạo:**
```python
class ToGSearch:
    def __init__(self, ..., width=3, depth=3, num_retain_entity=5):
        self.explorer = GraphExplorer(entities, relationships, ...)
        self.pruning_strategy = pruning_strategy  # LLM / Semantic / BM25
        self.reasoning_module = reasoning_module
        self.width = width    # Giữ tối đa 3 đường tốt nhất
        self.depth = depth    # Đi tối đa 3 bước
        self.num_retain_entity = num_retain_entity  # Chấm điểm tối đa 5 entity
```

**Tham số quan trọng:**
| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `width` (beam_width) | Số đường đi giữ lại mỗi bước | 3 |
| `depth` (max_depth) | Số bước tối đa từ entity đầu | 3 |
| `num_retain_entity` | Số entity candidate đưa vào chấm điểm | 5 |

---

## 🚶 Luồng hoạt động — Từng bước một

```
Câu hỏi vào
    ↓
[1] Tìm Entity đầu vào (Entity Linking)
    ↓
[2] Kiểm tra: Entities đầu vào đã đủ trả lời chưa? (Depth-0 check)
    ↓ (Chưa đủ)
[3] VÒNG LẶP (tối đa depth lần):
    a. Lấy tất cả quan hệ từ các node hiện tại
    b. Chấm điểm quan hệ (Pruning Relations)
    c. Giữ top-W quan hệ tốt nhất
    d. Chấm điểm entity đích (Pruning Entities)
    e. Tính score tổng hợp → tạo node mới
    f. Giữ top-W node tốt nhất (Beam Search)
    g. Kiểm tra: Đủ trả lời chưa? (Early Termination)
    ↓ (Hết vòng lặp)
[4] Tổng hợp tất cả paths → Sinh câu trả lời cuối
```

---

## 🌟 Ví dụ thực tế: "Satya Nadella là CEO của công ty nào và công ty đó làm gì?"

### Giả sử Knowledge Graph có:

```
[SATYA NADELLA] --[CEO_OF]--> [MICROSOFT]
[MICROSOFT]     --[DEVELOPS]--> [AZURE]
[MICROSOFT]     --[ACQUIRES]--> [GITHUB]
[MICROSOFT]     --[PARTNERS_WITH]--> [OPENAI]
[AZURE]         --[IS_A]--> [CLOUD_PLATFORM]
```

---

### ASCII Flow — Chạy với width=2, depth=2

```
═══════════════════════════════════════════════════════════════════════
 QUERY: "Satya Nadella là CEO của công ty nào và công ty đó làm gì?"
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: Entity Linking (Tìm cửa vào)                               │
│                                                                     │
│  Query Embedding ──► So sánh với 500 entities trong graph          │
│                                                                     │
│  Score cao nhất:                                                    │
│  ✅ "SATYA NADELLA"   score=0.92  ← CHỌN (top-1)                  │
│  ✅ "MICROSOFT"       score=0.85  ← CHỌN (top-2, width=2)         │
│  ❌ "AZURE"           score=0.71  ← BỎ (vượt width)               │
│                                                                     │
│  Starting nodes: [SATYA NADELLA, MICROSOFT]                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2: Early Termination Check (Depth=0)                          │
│                                                                     │
│  LLM: "Chỉ với SATYA NADELLA + MICROSOFT, đủ trả lời chưa?"       │
│  → "Chưa đủ, chưa biết Microsoft làm gì"                          │
│  → Tiếp tục đào sâu...                                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3a: Expand Depth=1 — Lấy relations từ các starting nodes      │
│                                                                     │
│  Từ [SATYA NADELLA]:                                               │
│  ├── CEO_OF        → MICROSOFT        (outgoing)                   │
│  └── BORN_IN       → INDIA            (outgoing)                   │
│                                                                     │
│  Từ [MICROSOFT]:                                                   │
│  ├── DEVELOPS      → AZURE            (outgoing)                   │
│  ├── ACQUIRES      → GITHUB           (outgoing)                   │
│  └── PARTNERS_WITH → OPENAI           (outgoing)                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3b: Pruning Relations (LLM chấm điểm)                        │
│                                                                     │
│  Prompt → LLM: "Câu hỏi: Satya Nadella CEO của ai và làm gì?      │
│  Từ SATYA NADELLA có các quan hệ: 1.CEO_OF 2.BORN_IN              │
│  Chấm 1-10 cho từng quan hệ:"                                      │
│                                                                     │
│  LLM trả về: [9, 1]                                                │
│  ✅ CEO_OF   → score=9.0  ← GIỮ LẠI                               │
│  ❌ BORN_IN  → score=1.0  ← LOẠI (không liên quan)                │
│                                                                     │
│  Từ MICROSOFT: [8, 7, 6]                                           │
│  ✅ DEVELOPS → AZURE   score=8.0                                   │
│  ✅ ACQUIRES → GITHUB  score=7.0                                   │
│  ❌ PARTNERS → OPENAI  score=6.0  ← (vượt width=2 → loại)         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3c: Pruning Entities (LLM chấm điểm entity đích)              │
│                                                                     │
│  Candidates: [MICROSOFT(9), AZURE(8), GITHUB(7)]                   │
│  Nếu > num_retain_entity=5 → random sample, rồi chấm điểm         │
│                                                                     │
│  Prompt → LLM: "Path hiện tại: SATYA NADELLA --CEO_OF--> ...      │
│  Entity candidates: 1.MICROSOFT 2.AZURE 3.GITHUB                   │
│  Chấm 1-10:"                                                       │
│  LLM: [10, 8, 6]                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3d: Tính Combined Score & Beam Search                         │
│                                                                     │
│  combined_score = parent.score × rel_score × (entity_score / 10)   │
│                                                                     │
│  Node A: SATYA NADELLA→MICROSOFT                                   │
│    score = 1.0 × 9.0 × (10/10) = 9.0                              │
│                                                                     │
│  Node B: MICROSOFT→AZURE                                           │
│    score = 1.0 × 8.0 × (8/10)  = 6.4                              │
│                                                                     │
│  Node C: MICROSOFT→GITHUB                                          │
│    score = 1.0 × 7.0 × (6/10)  = 4.2                              │
│                                                                     │
│  Beam Search: Giữ top-2 (width=2):                                 │
│  ✅ Node A: SATYA NADELLA→MICROSOFT  score=9.0  ← GIỮ             │
│  ✅ Node B: MICROSOFT→AZURE          score=6.4  ← GIỮ             │
│  ❌ Node C: MICROSOFT→GITHUB         score=4.2  ← BỎ              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3e: Early Termination Check (Depth=1)                         │
│                                                                     │
│  Frontier: [MICROSOFT(từ Nadella), AZURE(từ Microsoft)]            │
│                                                                     │
│  LLM: "Với info này, đủ trả lời câu hỏi chưa?"                    │
│  → "Đủ rồi! SATYA NADELLA là CEO của MICROSOFT.                    │
│       MICROSOFT phát triển AZURE (cloud platform)."                │
│  → should_terminate = True ✅                                       │
│  → Kết thúc sớm! Không cần đào đến depth=2                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 4: Generate Final Answer                                      │
│                                                                     │
│  Reasoning paths:                                                   │
│  • SATYA NADELLA --[CEO_OF]--> MICROSOFT                           │
│  • MICROSOFT --[DEVELOPS]--> AZURE (cloud platform)               │
│                                                                     │
│  + Text Units: Đoạn văn gốc trong corpus về Microsoft & Nadella    │
│                                                                     │
│  LLM sinh câu trả lời:                                             │
│  "Satya Nadella là CEO của Microsoft. Microsoft là công ty         │
│   công nghệ phát triển các sản phẩm như Azure (nền tảng           │
│   điện toán đám mây), GitHub, và hợp tác với OpenAI..."           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🌳 Cây tìm kiếm đầy đủ (Beam Search Tree)

```
                    [QUERY]
                       │
         ┌─────────────┴──────────────┐
         │                            │
    [SATYA NADELLA]              [MICROSOFT]
      score=1.0                   score=1.0
         │                            │
    ┌────┴────┐                 ┌─────┼─────┐
    │         │                 │     │     │
[MICROSOFT] [INDIA]         [AZURE][GITHUB][OPENAI]
  9.0×1.0    1.0×1.0         8.0    7.0    6.0
  =9.0  ✅   =1.0  ❌        ✅     ✅     ❌

         BEAM WIDTH=2: Chỉ giữ top-2
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         ✅ SATYA→MICROSOFT   (9.0)
         ✅ MICROSOFT→AZURE   (6.4)
         ❌ MICROSOFT→GITHUB  (4.2)  ← bị loại
```

---

## 🔁 Scoring Formula — Công thức tính điểm

```
hop_score     = rel_score × (entity_score / 10.0)
combined_score = parent_node.score × hop_score

Ví dụ:
  rel_score    = 9.0   (LLM chấm quan hệ CEO_OF)
  entity_score = 10.0  (LLM chấm entity Microsoft)
  parent.score = 1.0   (node gốc)

  hop_score     = 9.0 × (10.0 / 10.0) = 9.0
  combined_score = 1.0 × 9.0 = 9.0

  → Giá trị này được dùng để xếp hạng, giữ top-W
```

---

## 📊 Metrics thu thập

Mỗi lần gọi, `ToGMetrics` đếm:

```
┌──────────────────────────────────────────────────────┐
│ Exploration (pruning)     │ Reasoning                │
├──────────────────────────────────────────────────────┤
│ llm_calls (LLM gọi)       │ llm_calls                │
│ prompt_tokens (token vào) │ prompt_tokens            │
│ output_tokens (token ra)  │ output_tokens            │
│ embedding_calls           │                          │
│ embedding_tokens          │                          │
└──────────────────────────────────────────────────────┘
```

---

## 🏗️ Sơ đồ kiến trúc tổng thể

```
                         ┌─────────────────┐
                         │   ToGSearch     │  ← Trưởng nhóm
                         │   (search.py)   │
                         └────────┬────────┘
                                  │ điều phối
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  GraphExplorer   │   │ PruningStrategy  │   │  ToGReasoning    │
│ (exploration.py) │   │  (pruning.py)    │   │ (reasoning.py)   │
│                  │   │                  │   │                  │
│ • Tìm entity đầu │   │ • LLMPruning     │   │ • Early stop?    │
│ • Lấy relations  │   │ • SemanticPruning│   │ • Gen answer     │
│ • Lấy text units │   │ • BM25Pruning    │   │ • Format paths   │
└──────────────────┘   └──────────────────┘   └──────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │ dùng chung
                                  ▼
                         ┌─────────────────┐
                         │  ToGSearchState │  ← Bộ nhớ
                         │   (state.py)    │
                         │                 │
                         │ ExplorationNode │
                         │ ToGSearchState  │
                         └─────────────────┘
```

---

## ⚡ Tại sao thiết kế như vậy?

| Quyết định | Lý do |
|---|---|
| **3 pruning strategy** (LLM, Semantic, BM25) | Linh hoạt: LLM chính xác nhất, Semantic nhanh hơn, BM25 dùng offline |
| **Beam Search** thay vì BFS/DFS | Kiểm soát được chi phí (token LLM) — không "đi lạc" ra vô hạn |
| **Early termination** sau mỗi depth | Không tốn LLM call thừa khi đã có đủ thông tin |
| **num_retain_entity + random sample** | Tránh bottleneck khi node có quá nhiều quan hệ |
| **Streaming generator** (AsyncGenerator) | UI nhận được response từng phần, không cần chờ toàn bộ xong |
| **Tách exploration / pruning / reasoning** | SRP: mỗi class 1 việc, dễ test, dễ thay thế từng phần |
| **Metric tracking tách biệt** | Phân tích rõ: tốn token ở bước nào (exploration vs reasoning) |

---

*Được viết bởi tác giả khóa luận. Codebase: `graphrag/query/structured_search/tog_search/`*
