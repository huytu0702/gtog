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
[SATYA NADELLA] ──[CEO_OF]──────► [MICROSOFT]
[SATYA NADELLA] ──[BORN_IN]─────► [INDIA]
[SATYA NADELLA] ──[LEADS]───────► [AI_STRATEGY]
[MICROSOFT]     ──[DEVELOPS]────► [AZURE]
[MICROSOFT]     ──[ACQUIRES]────► [GITHUB]
[MICROSOFT]     ──[INVESTS_IN]──► [OPENAI]
[MICROSOFT]     ──[FOUNDED_BY]──► [BILL_GATES]
[AZURE]         ──[POWERS]──────► [CHATGPT]
[AZURE]         ──[COMPETES]────► [AWS]
[OPENAI]        ──[BUILDS]──────► [GPT4]
[GITHUB]        ──[HOSTS]───────► [COPILOT]
```

> **Config:** `width=3`, `depth=3`, `num_entity_retain=5`

---

### ASCII Flow — Chi tiết với width=3, depth=3, num_entity_retain=5

```
╔═════════════════════════════════════════════════════════════════════════╗
║  QUERY: "Satya Nadella là CEO của công ty nào và công ty đó làm gì?"   ║
║  CONFIG: width=3 │ depth=3 │ num_entity_retain=5                       ║
╚═════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BƯỚC 1 │ ENTITY LINKING  — Tìm 3 cửa vào (width=3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Query ──embed──► vector [0.12, -0.34, ...]
                      │
                      ▼  cosine_similarity với TẤT CẢ entity
  ┌──────────────────────────────────────────────────────────────┐
  │  Entity          │ Sim Score │ Rank │ Kết quả               │
  ├──────────────────┼───────────┼──────┼───────────────────────┤
  │  SATYA NADELLA   │   0.92    │  #1  │ ✅ CHỌN (top-1)       │
  │  MICROSOFT       │   0.85    │  #2  │ ✅ CHỌN (top-2)       │
  │  AZURE           │   0.71    │  #3  │ ✅ CHỌN (top-3)       │
  │  OPENAI          │   0.64    │  #4  │ ❌ BỎ  (vượt width=3) │
  │  BILL GATES      │   0.51    │  #5  │ ❌ BỎ                 │
  │  ...             │   ...     │ ...  │ ❌ BỎ                 │
  └──────────────────────────────────────────────────────────────┘

  Starting frontier (depth=0):
  ┌──────────────────┬────────────┬────────┐
  │ Node             │ Score      │ Parent │
  ├──────────────────┼────────────┼────────┤
  │ SATYA NADELLA    │ 1.0        │ ROOT   │
  │ MICROSOFT        │ 1.0        │ ROOT   │
  │ AZURE            │ 1.0        │ ROOT   │
  └──────────────────┴────────────┴────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BƯỚC 2 │ EARLY TERMINATION CHECK — Depth=0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Frontier: [SATYA NADELLA, MICROSOFT, AZURE]

  LLM: "Chỉ với 3 entity trên, đủ trả lời câu hỏi chưa?"
  → "Chưa — biết tên nhưng chưa rõ Microsoft làm gì cụ thể"
  → should_terminate = False → Tiếp tục vòng lặp depth=1

┌─────────────────────────────────────────────────────────────────────────┐
│                    ╔══════════════════╗                                 │
│                    ║  DEPTH = 1 / 3   ║                                 │
│                    ╚══════════════════╝                                 │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ STEP A — Lấy TẤT CẢ quan hệ từ 3 frontier nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [SATYA NADELLA] → 3 relations:            [MICROSOFT] → 4 relations:
  ┌──────────────────────────────┐           ┌──────────────────────────────┐
  │ CEO_OF      → MICROSOFT      │           │ DEVELOPS    → AZURE          │
  │ BORN_IN     → INDIA          │           │ ACQUIRES    → GITHUB         │
  │ LEADS       → AI_STRATEGY    │           │ INVESTS_IN  → OPENAI         │
  └──────────────────────────────┘           │ FOUNDED_BY  → BILL_GATES     │
                                             └──────────────────────────────┘
  [AZURE] → 2 relations:
  ┌──────────────────────────────┐
  │ POWERS      → CHATGPT        │
  │ COMPETES    → AWS            │
  └──────────────────────────────┘

  Tổng: 9 quan hệ từ 3 nodes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ STEP B — Pruning Relations (LLM chấm, giữ top-3 per node)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ Từ SATYA NADELLA ─────────────────────────────────────────────────┐
  │  LLM prompt: "Query=..., Entity=SATYA NADELLA, score các quan hệ:" │
  │  Quan hệ        → Entity đích    │ LLM score │ Kết quả             │
  │  CEO_OF         → MICROSOFT      │     9     │ ✅ GIỮ (top-1)      │
  │  LEADS          → AI_STRATEGY    │     6     │ ✅ GIỮ (top-2)      │
  │  BORN_IN        → INDIA          │     1     │ ❌ LOẠI (score thấp)│
  └────────────────────────────────────────────────────────────────────┘

  ┌─ Từ MICROSOFT ─────────────────────────────────────────────────────┐
  │  LLM score:                                                        │
  │  DEVELOPS    → AZURE          │  8  │ ✅ GIỮ (top-1)              │
  │  ACQUIRES    → GITHUB         │  7  │ ✅ GIỮ (top-2)              │
  │  INVESTS_IN  → OPENAI         │  7  │ ✅ GIỮ (top-3)              │
  │  FOUNDED_BY  → BILL_GATES     │  2  │ ❌ LOẠI (vượt width=3)      │
  └────────────────────────────────────────────────────────────────────┘

  ┌─ Từ AZURE ─────────────────────────────────────────────────────────┐
  │  LLM score:                                                        │
  │  POWERS      → CHATGPT        │  7  │ ✅ GIỮ (top-1)              │
  │  COMPETES    → AWS            │  4  │ ✅ GIỮ (top-2, width cho phép)│
  └────────────────────────────────────────────────────────────────────┘

  → Còn lại 7 cặp (relation → entity đích) sau pruning relations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ STEP C — num_entity_retain=5: Sample trước khi chấm entity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Candidate entities (từ 7 cặp trên):
  [MICROSOFT, AI_STRATEGY, AZURE, GITHUB, OPENAI, CHATGPT, AWS]
                             │ = 7 entities
                             │ > num_entity_retain=5
                             ▼
               random.sample(candidates, k=5)
                             │
                             ▼
  ┌──────────────────────────────────────────────────┐
  │  Sampled 5 để đưa vào LLM:                      │
  │  1. MICROSOFT   ← được chọn ngẫu nhiên          │
  │  2. AZURE       ← được chọn                     │
  │  3. GITHUB      ← được chọn                     │
  │  4. OPENAI      ← được chọn                     │
  │  5. CHATGPT     ← được chọn                     │
  │  ✗ AI_STRATEGY  ← bị loại khỏi sample lần này  │
  │  ✗ AWS          ← bị loại khỏi sample lần này  │
  └──────────────────────────────────────────────────┘

  ⚠️  Mục đích: Tránh gửi quá nhiều token vào LLM khi node có hàng
      chục/trăm kết nối. Chỉ gửi tối đa 5 entity để chấm điểm.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ STEP D — Pruning Entities (LLM chấm entity đích)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LLM prompt:
  "Query: Satya Nadella CEO của ai, và công ty đó làm gì?
   Current paths:
     • ROOT → SATYA NADELLA → (CEO_OF) → ???
     • ROOT → MICROSOFT     → (DEVELOPS) → ???
     • ROOT → AZURE         → (POWERS) → ???
   Entity candidates: 1.MICROSOFT 2.AZURE 3.GITHUB 4.OPENAI 5.CHATGPT
   Chấm 1-10 xem entity nào giúp trả lời câu hỏi nhất:"

  LLM trả về: [10, 8, 6, 7, 5]
  ┌────────────┬────────────┬─────────┐
  │ Entity     │ Rel Score  │ Ent Score│
  ├────────────┼────────────┼─────────┤
  │ MICROSOFT  │ 9 (CEO_OF) │   10    │
  │ OPENAI     │ 7 (INVEST) │    7    │
  │ AZURE      │ 8 (DEVELO) │    8    │
  │ GITHUB     │ 7 (ACQUIR) │    6    │
  │ CHATGPT    │ 7 (POWERS) │    5    │
  └────────────┴────────────┴─────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ STEP E — Combined Score & Beam Search (giữ top-3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Công thức:
  combined_score = parent.score × rel_score × (entity_score / 10.0)

  ┌────────────────────────────────────────────────────────────────────┐
  │ Node mới              │ parent │ rel │ ent │ Combined              │
  ├───────────────────────┼────────┼─────┼─────┼───────────────────────┤
  │ NADELLA→MICROSOFT     │  1.0   │  9  │ 10  │ 1.0×9×(10/10) = 9.00 │
  │ MICROSOFT→AZURE       │  1.0   │  8  │  8  │ 1.0×8×(8/10)  = 6.40 │
  │ MICROSOFT→OPENAI      │  1.0   │  7  │  7  │ 1.0×7×(7/10)  = 4.90 │
  │ MICROSOFT→GITHUB      │  1.0   │  7  │  6  │ 1.0×7×(6/10)  = 4.20 │
  │ AZURE→CHATGPT         │  1.0   │  7  │  5  │ 1.0×7×(5/10)  = 3.50 │
  └───────────────────────┴────────┴─────┴─────┴───────────────────────┘

  Beam Search — Sắp xếp và giữ top-3 (width=3):
  ✅ #1  NADELLA→MICROSOFT   score=9.00  ← GIỮ
  ✅ #2  MICROSOFT→AZURE     score=6.40  ← GIỮ
  ✅ #3  MICROSOFT→OPENAI    score=4.90  ← GIỮ
  ❌ #4  MICROSOFT→GITHUB    score=4.20  ← BỎ (vượt width=3)
  ❌ #5  AZURE→CHATGPT       score=3.50  ← BỎ

  Frontier sau depth=1:
  ┌─────────────────────┬───────┬──────────────────────────────┐
  │ Node                │ Score │ Path                         │
  ├─────────────────────┼───────┼──────────────────────────────┤
  │ MICROSOFT           │  9.00 │ ROOT→NADELLA→MICROSOFT       │
  │ AZURE               │  6.40 │ ROOT→MICROSOFT→AZURE         │
  │ OPENAI              │  4.90 │ ROOT→MICROSOFT→OPENAI        │
  └─────────────────────┴───────┴──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=1 │ EARLY TERMINATION CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LLM: "Path hiện có: NADELLA→MICROSOFT, MICROSOFT→AZURE, MICROSOFT→OPENAI
        Đủ trả lời 'Nadella là CEO của ai và công ty đó làm gì?' chưa?"
  → "Biết Nadella là CEO Microsoft, Microsoft có Azure, nhưng
     chưa rõ Azure/OpenAI đang làm GÌ cụ thể"
  → should_terminate = False → Tiếp tục depth=2

┌─────────────────────────────────────────────────────────────────────────┐
│                    ╔══════════════════╗                                 │
│                    ║  DEPTH = 2 / 3   ║                                 │
│                    ╚══════════════════╝                                 │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=2 │ STEP A — Lấy relations từ 3 frontier nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [MICROSOFT]  → DEVELOPS→AZURE, ACQUIRES→GITHUB, INVESTS_IN→OPENAI,
                 FOUNDED_BY→BILL_GATES   (4 relations)

  [AZURE]      → POWERS→CHATGPT, COMPETES→AWS, IS_A→CLOUD_PLATFORM
                 (3 relations)

  [OPENAI]     → BUILDS→GPT4, PARTNERED_BY→MICROSOFT, LED_BY→SAM_ALTMAN
                 (3 relations)

  Tổng: 10 quan hệ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=2 │ STEP B — Pruning Relations (top-3 per node)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Từ MICROSOFT:    DEVELOPS(8)✅  ACQUIRES(7)✅  INVESTS_IN(7)✅
                   FOUNDED_BY(2)❌
  Từ AZURE:        IS_A(9)✅      POWERS(7)✅    COMPETES(4)✅
  Từ OPENAI:       BUILDS(9)✅    LED_BY(5)✅    PARTNERED_BY(3)✅

  → 9 cặp sau pruning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=2 │ STEP C — num_entity_retain=5: Không cần sample (≤5 đủ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Candidates từ AZURE: [CLOUD_PLATFORM, CHATGPT, AWS] = 3 entities
  3 ≤ 5 → KHÔNG cần sample, gửi thẳng cả 3 vào LLM ✅

  Candidates từ OPENAI: [GPT4, SAM_ALTMAN, MICROSOFT] = 3 entities
  3 ≤ 5 → KHÔNG cần sample ✅

  Candidates từ MICROSOFT: [AZURE, GITHUB, OPENAI] = 3 entities
  3 ≤ 5 → KHÔNG cần sample ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=2 │ STEP D+E — Score Entities & Combined Score & Beam Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌───────────────────────────────────────────────────────────────────────┐
  │ Node mới           │ parent_score│ rel│ ent │ Combined               │
  ├────────────────────┼─────────────┼────┼─────┼────────────────────────┤
  │ AZURE→CLOUD_PLATFORM│   6.40     │  9 │  9  │ 6.40×9×(9/10) = 51.84 │
  │ OPENAI→GPT4        │   4.90     │  9 │  9  │ 4.90×9×(9/10) = 39.69 │
  │ AZURE→CHATGPT      │   6.40     │  7 │  8  │ 6.40×7×(8/10) = 35.84 │
  │ MICROSOFT→AZURE    │   9.00     │  8 │  7  │ 9.00×8×(7/10) = 50.40 │
  │ OPENAI→SAM_ALTMAN  │   4.90     │  5 │  6  │ 4.90×5×(6/10) = 14.70 │
  │ AZURE→AWS          │   6.40     │  4 │  4  │ 6.40×4×(4/10) = 10.24 │
  └────────────────────┴─────────────┴────┴─────┴────────────────────────┘

  Beam Search — Giữ top-3 (width=3):
  ✅ #1  AZURE→CLOUD_PLATFORM   score=51.84  ← GIỮ
  ✅ #2  MICROSOFT→AZURE        score=50.40  ← GIỮ
  ✅ #3  OPENAI→GPT4            score=39.69  ← GIỮ
  ❌ #4  AZURE→CHATGPT          score=35.84  ← BỎ
  ❌ ...                                     ← BỎ

  Frontier sau depth=2:
  ┌──────────────────────┬────────┬────────────────────────────────────┐
  │ Node                 │ Score  │ Path                               │
  ├──────────────────────┼────────┼────────────────────────────────────┤
  │ CLOUD_PLATFORM       │ 51.84  │ NADELLA→MICROSOFT→AZURE→CLOUD_PLT  │
  │ AZURE (2nd path)     │ 50.40  │ MICROSOFT→AZURE (từ MICROSOFT)     │
  │ GPT4                 │ 39.69  │ MICROSOFT→OPENAI→GPT4              │
  └──────────────────────┴────────┴────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPTH=2 │ EARLY TERMINATION CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LLM nhận top paths:
  • NADELLA → MICROSOFT → AZURE → CLOUD_PLATFORM
  • NADELLA → MICROSOFT → OPENAI → GPT4

  LLM: "Giờ đã biết: Nadella là CEO Microsoft. Microsoft phát triển
        Azure (nền tảng CLOUD). Đầu tư OpenAI, xây GPT4.
        Câu hỏi hỏi 'công ty đó làm gì?' → ĐÃ ĐỦ!"
  → should_terminate = True ✅
  → Dừng sớm! Không cần chạy depth=3

┌─────────────────────────────────────────────────────────────────────────┐
│                    ╔══════════════════╗                                 │
│                    ║  DEPTH = 3 / 3   ║  ← BỎ QUA (đã terminate)      │
│                    ╚══════════════════╝                                 │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BƯỚC CUỐI │ GENERATE FINAL ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Reasoning paths thu thập được:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Path 1: SATYA NADELLA ──[CEO_OF]──► MICROSOFT                      │
  │                                         └──[DEVELOPS]──► AZURE     │
  │                                                   └──[IS_A]──► CLOUD│
  │                                                                     │
  │ Path 2: MICROSOFT ──[INVESTS_IN]──► OPENAI ──[BUILDS]──► GPT4      │
  └─────────────────────────────────────────────────────────────────────┘

  + Text Units: Đoạn văn gốc trong corpus về Microsoft, Azure, OpenAI

  LLM sinh câu trả lời cuối:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ "Satya Nadella là CEO của Microsoft. Microsoft là công ty công nghệ │
  │  lớn với các mảng kinh doanh chính: Azure (nền tảng điện toán đám  │
  │  mây), GitHub (nền tảng lưu trữ mã nguồn), và đầu tư vào OpenAI   │
  │  — tổ chức xây dựng GPT-4 và ChatGPT."                            │
  │                                                                     │
  │  [Nguồn: path NADELLA→MICROSOFT→AZURE→CLOUD_PLATFORM (score 51.8)] │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 🌳 Cây tìm kiếm đầy đủ (width=3, depth=3)

```
                              [QUERY]
                                 │
            ┌────────────────────┼────────────────────┐
            │                   │                    │
      [SATYA NADELLA]       [MICROSOFT]           [AZURE]
        score=1.0             score=1.0            score=1.0
   (Entity Linking #1)   (Entity Linking #2)  (Entity Linking #3)
            │                   │                    │
       ┌────┴─────┐         ┌───┼───┐            ┌───┴───┐
       │          │         │   │   │            │       │
  [MICROSOFT] [AI_STRAT] [AZURE][GITHUB][OPENAI][CHATGPT][AWS]
    rel=9       rel=6      rel=8  rel=7   rel=7   rel=7   rel=4
    ent=10      ent=7      ent=8  ent=6   ent=7   ent=5   ent=4
   =9.00✅    =4.20❌    =6.40✅ =4.20❌ =4.90✅ =3.50❌ =2.80❌

  ┌─────────────────────────────────────────────────────────────┐
  │ BEAM SEARCH depth=1: Giữ TOP-3 toàn cục                    │
  │   ✅ MICROSOFT  9.00  (path: NADELLA→MICROSOFT)             │
  │   ✅ AZURE      6.40  (path: MICROSOFT→AZURE)               │
  │   ✅ OPENAI     4.90  (path: MICROSOFT→OPENAI)              │
  └─────────────────────────────────────────────────────────────┘
            │                   │                    │
       ┌────┴──┐          ┌─────┼────┐          ┌───┴────┐
       │       │          │     │    │          │        │
  [AZURE] [GITHUB]  [CLOUD_PLT][CHATGPT][AWS] [GPT4] [SAM_ALTMAN]
    r=8    r=7       r=9   r=7   r=4    r=9     r=5
    e=7    e=6       e=9   e=8   e=4    e=9     e=6
  =50.4  =37.8    =51.8  =35.8 =10.2  =39.7  =14.7

  ┌─────────────────────────────────────────────────────────────┐
  │ BEAM SEARCH depth=2: Giữ TOP-3 toàn cục                    │
  │   ✅ CLOUD_PLT  51.84  (NADELLA→MSFT→AZURE→CLOUD_PLATFORM)  │
  │   ✅ AZURE      50.40  (NADELLA→MSFT→AZURE — 2nd path)      │
  │   ✅ GPT4       39.69  (MSFT→OPENAI→GPT4)                   │
  └─────────────────────────────────────────────────────────────┘
            │
    [EARLY TERMINATION ✅]
    Đủ thông tin → Dừng, không chạy depth=3

  ╔═════════════════════════════════════════════════════════════╗
  ║  FINAL ANSWER tổng hợp từ 3 winning paths                  ║
  ╚═════════════════════════════════════════════════════════════╝
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
