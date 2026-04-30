# ToG Search: Luồng Hoạt Động Chi Tiết

**Thông số ví dụ:**
```yaml
width: 3
depth: 3
prune_strategy: llm
num_retain_entity: 5
temperature_exploration: 0.4
temperature_reasoning: 0.0
max_context_tokens: 8000
```

**Query ví dụ:** `"Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?"`

---

## Tổng quan luồng

```
QUERY
  │
  ▼
[1] ENTITY LINKING ──► top_k=width=3 entities
  │
  ▼
[2] DEPTH-0 EARLY CHECK ──► LLM: "đủ thông tin chưa?"
  │ NO
  ▼
[3] EXPLORATION LOOP (depth 1..3)
  │   ├─ _process_node()
  │   │     ├─ get_relations()
  │   │     ├─ _filter_backtrack_relations() ──► loại cạnh ngược lại (anti-cycle)
  │   │     ├─ score_relations() ──► LLM/semantic
  │   │     └─ score_entities() ──► LLM/semantic
  │   ├─ prune_current_frontier() ──► giữ top beam_width=3
  │   └─ check_early_termination() ──► LLM kiểm tra sớm
  │
  ▼
[4] GENERATE FINAL ANSWER ──► reasoning.generate_answer()
  │
  ▼
SearchResult(response, context_data, llm_calls, ...)
```

---

## Bước 1: Entity Linking

**Hàm:** `ToGSearch._stream_search_with_metrics()` → `GraphExplorer.find_starting_entities_semantic()`

```
Query: "Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?"
         │
         ▼ embed query
query_embedding = [0.12, -0.34, 0.78, ...]   # vector 1536 chiều
         │
         ▼ dot product với tất cả entity embeddings
MIT_embedding      · query = 0.91  ← cao nhất
Google_embedding   · query = 0.87
OpenAI_embedding   · query = 0.83
Harvard_embedding  · query = 0.71
...
         │
         ▼ top_k = width = 3
starting_entities = ["MIT", "Google", "OpenAI"]
```

**Kết quả:** 3 ExplorationNode khởi đầu với score=1.0, depth=0, parent=None

---

## Bước 2: Kiểm tra Early Termination tại Depth-0

**Hàm:** `ToGReasoning.check_early_termination()` — temperature=0.0

```
LLM prompt:
  "Question: Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?
   Current exploration paths:
   [MIT] Description: Đại học MIT, chuyên về khoa học và công nghệ...
   [Google] Description: Tập đoàn Google, chuyên về...
   [OpenAI] Description: ...
   Can you answer with high confidence? YES/NO"

LLM response: "NO: cần thêm thông tin về mối quan hệ tài trợ"
```

→ Tiếp tục exploration loop.

---

## Bước 3: Exploration Loop (Depth 1 → 3)

### Cơ chế Backtrack Prevention (mới)

Trước khi score relations, `_process_node` lọc bỏ relation sẽ đi ngược lại cạnh vừa dùng để đến node hiện tại — tránh vòng lặp A→B→A.

```
Cơ chế: _filter_backtrack_relations(node, relations)

Nếu node B được tạo ra bằng cách đi qua:
  relation_from_parent     = "tài trợ bởi"
  relation_direction       = "outgoing"   (A → B)

Thì khi expand B, loại bỏ:
  ("tài trợ bởi", A, "incoming", ...)    ← đây là bước ngược B → A

Giữ nguyên:
  ("tài trợ bởi", C, "outgoing", ...)    ← cùng tên nhưng đi tiếp, không phải backtrack
  ("hợp tác với", A, "incoming", ...)    ← khác relation, không lọc

Tương đương pre_relations + pre_heads trong ToG gốc (GasolSun36/ToG).
Root nodes (relation_from_parent=None) → không lọc gì.
```

**Ví dụ trực quan:**
```
Before filter (tất cả relations của Google):
  ("tài trợ bởi",  "MIT",     "incoming", 0.9)   ← BACKTRACK → bị loại
  ("sở hữu",       "DeepMind","outgoing", 0.8)   → giữ lại
  ("đầu tư vào",   "Waymo",   "outgoing", 0.7)   → giữ lại
  ("hợp tác với",  "MIT",     "incoming", 0.6)   → giữ lại (khác relation)

After filter → [("sở hữu",...), ("đầu tư vào",...), ("hợp tác với",...)]
```

---

### Depth 1

```
Frontier = [MIT, Google, OpenAI]  (3 nodes, = width)
         │
         ▼ asyncio.gather — xử lý song song 3 nodes
         │
         ├─ _process_node(MIT)
         │     │
         │     ├─ GraphExplorer.get_relations("MIT")
         │     │     → [("tài trợ bởi", "Google", "incoming", 0.9),
         │     │          ("hợp tác với", "OpenAI", "outgoing", 0.8),
         │     │          ("nghiên cứu về", "Deep Learning", "outgoing", 0.7),
         │     │          ("đặt tại", "Cambridge", "outgoing", 0.5), ...]
         │     │
         │     ├─ _filter_backtrack_relations()   ← MIT là root node, không lọc gì
         │     │
         │     ├─ LLMPruning.score_relations()   ← temperature=0.4
         │     │     prompt: "Query: ... Entity: MIT
         │     │              Relations:
         │     │              1. [incoming] tài trợ bởi... (weight: 0.90)
         │     │              2. [outgoing] hợp tác với... (weight: 0.80)
         │     │              3. [outgoing] nghiên cứu về... (weight: 0.70)
         │     │              4. [outgoing] đặt tại... (weight: 0.50)
         │     │              Score each 1-10:"
         │     │     LLM: "[9, 7, 5, 2]"
         │     │
         │     ├─ sort by score → [("tài trợ bởi", score=9), ("hợp tác với", score=7), ...]
         │     ├─ chọn top width=3 relation groups
         │     │
         │     ├─ build entity candidates per group
         │     │     "tài trợ bởi": Google → 1 candidate
         │     │     "hợp tác với": OpenAI, Microsoft → 2 candidates
         │     │     "nghiên cứu về": Deep Learning, NLP → 2 candidates
         │     │     Total = 5 candidates (< ENTITY_CANDIDATE_SAMPLE_THRESHOLD=20, giữ hết)
         │     │
         │     └─ LLMPruning.score_entities()   ← temperature=0.4
         │           entities_text:
         │             "1. Google: công ty công nghệ..."
         │             "2. OpenAI: công ty AI..."
         │             "3. Microsoft: ..."
         │             "4. Deep Learning: ..."
         │             "5. NLP: ..."
         │           LLM: "[9, 7, 6, 4, 3]"
         │           → new nodes với scores:
         │             Google: score = MIT.score(1.0) × rel_score(9/10) × entity_score(9/10) = 0.81
         │             OpenAI: score = 1.0 × 0.7 × 0.7 = 0.49
         │             Microsoft: score = 1.0 × 0.7 × 0.6 = 0.42
         │             Deep Learning: score = 1.0 × 0.5 × 0.4 = 0.20
         │             NLP: score = 1.0 × 0.5 × 0.3 = 0.15
         │
         ├─ _process_node(Google) → tương tự...
         │     → new nodes: [MIT_research, DeepMind, Alphabet, ...]
         │
         └─ _process_node(OpenAI) → tương tự...
               → new nodes: [Microsoft, Sam_Altman, GPT4, ...]
```

**Sau gather:** next_level_nodes = tất cả nodes từ 3 process_node (có thể 10-15 nodes)

```
state.nodes_by_depth[1] = [Google(0.81), OpenAI(0.49), Microsoft(0.42),
                            DeepMind(0.38), MIT_research(0.35),
                            Alphabet(0.31), Sam_Altman(0.28), ...]
```

**Prune:**
```python
# state.prune_current_frontier() — state.py:81-85
frontier.sort(key=lambda n: n.score, reverse=True)
self.nodes_by_depth[1] = frontier[:beam_width]  # beam_width = width = 3
```

```
After prune depth-1: [Google(0.81), OpenAI(0.49), Microsoft(0.42)]
                                      ↑ giữ top-3
```

**Early termination check:**
```
LLM prompt: "Question: ... Current paths:
  MIT --[tài trợ bởi]--> Google
  MIT --[hợp tác với]--> OpenAI
  Google --[sở hữu]--> DeepMind
  Can you answer? YES/NO"

LLM: "YES: Google tài trợ cho dự án nghiên cứu AI của MIT [Data: Entities (Google, MIT)]"
```

→ Nếu not force_max_depth: **trả về ngay**, không cần depth 2-3.

---

### Depth 2 (nếu không early terminate)

```
Frontier depth-1 = [Google, OpenAI, Microsoft]
         │
         ▼ _process_node song song
         │
         ├─ _process_node(Google, parent=MIT)
         │     current_path = "MIT --[tài trợ bởi]--> Google"   ← từ _node_to_path_string()
         │     relation_history = "MIT --[tài trợ bởi]--> Google"
         │     │
         │     get_relations("Google")
         │       → [("tài trợ bởi", "MIT", "incoming", 0.9),   ← BACKTRACK
         │            ("sở hữu",     "DeepMind", "outgoing", 0.8),
         │            ("đầu tư vào", "Waymo",    "outgoing", 0.7), ...]
         │     │
         │     _filter_backtrack_relations(Google_node, relations)
         │       Google arrived via ("tài trợ bởi", "outgoing")
         │       → loại ("tài trợ bởi", "MIT", "incoming")
         │       → giữ [("sở hữu",...), ("đầu tư vào",...)]
         │     │
         │     score_relations(relation_history=..., current_path=...)
         │     │
         │     score_entities() → new nodes
         │
         └─ tương tự OpenAI, Microsoft
```

**Prune depth-2:** giữ top-3 nodes.

**Early termination check** → nếu có đủ thông tin thì terminate.

---

### Depth 3 (nếu vẫn chưa terminate)

Tương tự depth 2. Đây là `max_depth` — sau depth 3 loop kết thúc.

---

## Bước 4: Generate Final Answer

**Hàm:** `ToGReasoning.generate_answer()` — temperature=0.0

```python
# Collect tất cả nodes từ mọi depth
all_paths = []
for depth_nodes in state.nodes_by_depth.values():
    all_paths.extend(depth_nodes)
# all_paths = nodes từ depth 0, 1, 2, 3

# Lấy text units gắn với các entities
all_text_units = explorer.get_text_units_for_nodes(all_paths)

# Format context
context_text = reasoning.format_paths(all_paths, text_units=all_text_units)
```

**format_paths() output:**
```
=== CHUNKS ===
[Chunk 1] "Google đã ký thỏa thuận tài trợ 50 triệu USD cho MIT..."
[Chunk 2] "Dự án CSAIL của MIT nhận được đầu tư từ Google AI..."

=== ENTITIES ===
[MIT]
Description: Viện Công nghệ Massachusetts, đại học hàng đầu...
[Google]
Description: Tập đoàn Alphabet's Google, chuyên về AI và tìm kiếm...

=== RELATIONSHIPS ===
- MIT <--[tài trợ bởi]-- Google
  Description: Google tài trợ 50M USD cho các dự án AI tại MIT CSAIL
- MIT --[hợp tác với]--> OpenAI
```

**LLM reasoning prompt:**
```
Question: Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?

Exploration Paths:
=== CHUNKS ===
...
=== ENTITIES ===
...
=== RELATIONSHIPS ===
...
```

**LLM response** (temperature=0.0 → deterministic):
```
Google là công ty chính tài trợ cho các dự án nghiên cứu AI của MIT.
[Data: Entities (Google, MIT)]

Cụ thể, Google đã ký thỏa thuận tài trợ 50 triệu USD cho MIT CSAIL
(Computer Science and Artificial Intelligence Laboratory)...
```

---

## Luồng Score trong _process_node

```
hop_score = rel_score × (max(entity_score, 0.0) / 10.0)
combined_score = parent_node.score × hop_score

Ví dụ:
  parent (MIT).score = 1.0
  rel_score("tài trợ bởi") = 9  (từ LLM, scale 1-10)
  entity_score(Google) = 9      (từ LLM, scale 1-10)

  hop_score = 9 × (9/10) = 8.1
  combined_score = 1.0 × (8.1/10) = 0.81
                                    ↑ vì hop_score không chia 10
```

**Thực tế trong code** (`search.py:651`):
```python
hop_score = rel_score * (max(entity_score, 0.0) / 10.0)
combined_score = node.score * hop_score
```

Score tích lũy theo depth → nodes sâu hơn có score nhỏ hơn → ưu tiên paths gần query.

---

## Xử lý num_retain_entity=5

```python
# search.py:584-600
if (len(group_candidates) >= ENTITY_CANDIDATE_SAMPLE_THRESHOLD  # = 20
        and len(group_candidates) > self.num_retain_entity):     # = 5
    # Random sample nếu group quá lớn (>= 20 entities cho 1 relation)
    candidate_data.extend(random.sample(group_candidates, self.num_retain_entity))
else:
    # Giữ hết nếu nhỏ hơn threshold
    candidate_data.extend(group_candidates)
```

**Ví dụ:** relation "tài trợ bởi MIT" có 25 công ty → random sample 5.
**Ví dụ:** relation "hợp tác với MIT" có 8 công ty → giữ cả 8.

---

## ASCII Flow Đầy Đủ

```
Query: "Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?"
  │
  ▼ find_starting_entities_semantic(top_k=3)
  │
  ┌─────────────────────────────────┐
  │  Depth 0 (initial frontier)     │
  │  ┌──────┐ ┌────────┐ ┌───────┐ │
  │  │ MIT  │ │ Google │ │OpenAI │ │  ← score=1.0 mỗi node
  │  │ s=1.0│ │  s=1.0 │ │ s=1.0 │ │
  │  └──────┘ └────────┘ └───────┘ │
  └─────────────────────────────────┘
  │
  ▼ check_early_termination → NO
  │
  ▼ _process_node × 3 (parallel)
  │   ├─ get_relations()
  │   ├─ _filter_backtrack_relations()  ← root nodes: no-op
  │   ├─ score_relations()
  │   └─ score_entities()
  │
  ┌──────────────────────────────────────────────┐
  │  Depth 1 candidates (before prune)           │
  │  Google(0.81) OpenAI(0.49) Microsoft(0.42)   │
  │  DeepMind(0.38) MIT_research(0.35) ...       │
  └──────────────────────────────────────────────┘
  │
  ▼ prune_current_frontier() → keep top beam_width=3
  │
  ┌─────────────────────────────────────┐
  │  Depth 1 (pruned frontier)          │
  │  ┌────────┐ ┌────────┐ ┌─────────┐ │
  │  │ Google │ │ OpenAI │ │Microsoft│ │
  │  │ s=0.81 │ │ s=0.49 │ │  s=0.42 │ │
  │  └────────┘ └────────┘ └─────────┘ │
  └─────────────────────────────────────┘
  │
  ▼ check_early_termination
  │   "YES: Google tài trợ MIT..."
  │
  ▼ (if not force_max_depth) RETURN EARLY
  │
  OR continue to depth 2, 3...
  │
  ▼ generate_answer(all_paths, temperature=0.0)
  │
  ▼ SearchResult(response="Google là...", llm_calls=N, ...)
```

---

## Tổng LLM Calls cho ví dụ trên

| Bước | Hàm | Calls |
|------|-----|-------|
| Depth-0 early check | `check_early_termination` | 1 |
| Depth-1: score_relations × 3 nodes | `LLMPruning.score_relations` | 3 |
| Depth-1: score_entities × 3 nodes | `LLMPruning.score_entities` | 3 |
| Depth-1 early check | `check_early_termination` | 1 |
| Final answer (nếu không early term) | `generate_answer` | 1 |
| **Total (nếu term tại depth-1)** | | **8** |
| **Total (nếu chạy hết depth-3)** | | **~26** |

Với `prune_strategy: llm`, mỗi bước exploration đều tốn 2 LLM calls/node (score_relations + score_entities).
