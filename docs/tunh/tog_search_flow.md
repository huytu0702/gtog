# ToG Search: Luồng Hoạt Động Chi Tiết

**Thông số ví dụ:**

```yaml
width: 3
depth: 3
prune_strategy: llm
num_retain_entity: 5
temperature_exploration: 0.4
temperature_reasoning: 0.0
```

**Query ví dụ:** `"Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?"`

---

## Tổng quan luồng

```text
QUERY
  │
  ▼
[1] ENTITY LINKING ──► top_k=width=3 entities
  │
  ▼
[2] DEPTH-0 EARLY CHECK ──► LLM kiểm tra: "đủ thông tin chưa?"
  │ NO
  ▼
[3] EXPLORATION LOOP (depth 1..3)
  │   ├─ _process_node()
  │   │     ├─ get_relations()
  │   │     ├─ _filter_backtrack_relations() ──► chặn bước quay ngược ngay lập tức
  │   │     ├─ score_relations() ──► LLM chấm relation theo query + path + history
  │   │     ├─ group theo (relation_name, direction)
  │   │     └─ score_entities() ──► LLM chấm entity ứng viên
  │   ├─ prune_current_frontier() ──► giữ top beam_width=3
  │   └─ check_early_termination() ──► LLM kiểm tra sớm sau mỗi depth
  │
  ▼
[4] GENERATE FINAL ANSWER ──► reasoning.generate_answer()
  │
  ▼
SearchResult(response, context_data, llm_calls, ...)
```

---

## Điểm cốt lõi từ code hiện tại

- **Depth 0 không chỉ nhìn entity description**: prompt early termination lấy cả `CHUNKS`, `ENTITIES`, và `RELATIONSHIPS` từ các node khởi đầu.
- **Backtrack filter là exact match**: chỉ chặn relation cùng tên và ngược hướng với relation vừa dùng để tới node hiện tại.
- **Dedupe relation group là theo `(relation_name, direction)`**: không fuzzy, không semantic matching.
- **`score_relations()` dùng cùng một LLM prompt** để chấm các relation ứng viên trong một node.
- **`score_entities()` chấm entity theo cùng `current_path`**, sau khi relation đã được chọn nhóm.
- **Final answer dùng prompt reasoning riêng** với citation rules rõ ràng.

---

## Bước 1: Entity Linking

**Hàm:** `ToGSearch._stream_search_with_metrics()` → `GraphExplorer.find_starting_entities_semantic()`

```text
Query: "Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?"
         │
         ▼ embed query
query_embedding = [0.12, -0.34, 0.78, ...]
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

**Kết quả trong code:** mỗi entity khởi đầu trở thành một `ExplorationNode` với:

- `depth = 0`
- `score = 1.0`
- `parent = None`
- `relation_from_parent = None`
- `entity_description = full_description`
- `entity_full_description = full_description`

---

## Bước 2: Kiểm tra Early Termination tại Depth-0

**Hàm:** `ToGReasoning.check_early_termination()` — `temperature=0.0`

### Dữ liệu đầu vào thực tế

`search.py` truyền vào:

- `frontier_nodes = state.get_current_frontier()`
- `frontier_text_units = self.explorer.get_text_units_for_nodes(frontier_nodes)`
- `conversation_history_context = history_context`

Trong `check_early_termination()`, prompt được tạo bằng:

- `history_prefix` nếu có conversation history
- `Question: {query}`
- `Current exploration paths:` → lấy từ `_format_paths(current_nodes[:3], text_units=...)`

### Prompt đầu ra gồm gì

`_format_paths()` hiện tại build 3 section:

1. `=== CHUNKS ===`
   - sinh từ `build_text_unit_context(text_units=..., context_name="CHUNKS")`
   - nếu không có text unit thì hiện `No chunk context available.`

2. `=== ENTITIES ===`
   - mỗi entity là một block:
     ```text
     [MIT]
     Description: ...
     ```
   - ưu tiên `entity_full_description`, fallback `entity_description`

3. `=== RELATIONSHIPS ===`
   - mỗi relation là một dòng:
     ```text
     - MIT --[tài trợ bởi]--> Google
     ```
   - nếu relation có mô tả dài hơn label thì thêm:
     ```text
       Description: Google tài trợ 50M USD...
     ```

### Ví dụ prompt rút gọn

```text
Question: Công ty nào tài trợ cho dự án nghiên cứu AI của MIT?

Current exploration paths:
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

Can you answer the question with high confidence based on these paths?
If yes, provide a complete answer that cites entities using [Data: Entities (...)] format.
Only cite entity names that appear in the exploration paths above.

Respond with:
- "YES: [answer with citations]" if you can answer confidently
- "NO: [reason]" if more exploration is needed
```

### Ý nghĩa

Depth-0 early check là một **LLM yes/no gate** dựa trên:

- entity khởi đầu
- text units liên quan
- mô tả entity
- relation đã có sẵn ở frontier

Nếu LLM trả `YES: ...` và không bật `debug_force_max_depth`, search trả kết quả sớm luôn.

---

## Bước 3: Exploration Loop (Depth 1 → 3)

### 3.1. Backtrack prevention

Trước khi chấm điểm relation, `_process_node()` gọi:

```python
_filter_backtrack_relations(node, relations)
```

**Luật lọc:**

- nếu node hiện tại đi tới từ relation `R` theo hướng `outgoing`
- thì relation ứng viên `R` ở hướng `incoming` sẽ bị loại
- ngược lại cũng tương tự
- root node thì không lọc gì

### Ví dụ

Nếu node `B` được tạo ra bởi:

```text
A --[tài trợ bởi]--> B
```

thì khi expand `B`, candidate này sẽ bị loại:

```python
("tài trợ bởi", "A", "incoming", 0.9)
```

nhưng các candidate sau vẫn giữ:

```python
("sở hữu", "DeepMind", "outgoing", 0.8)
("đầu tư vào", "Waymo", "outgoing", 0.7)
("hợp tác với", "MIT", "incoming", 0.6)
```

### 3.2. Prompt đưa vào `score_relations()`

**Hàm:** `LLMPruning.score_relations()`

Code build prompt từ template `TOG_RELATION_SCORING_PROMPT` với các biến:

- `query`
- `entity_name`
- `current_path`
- `relation_history`
- `relations`

#### `current_path`

Được tạo từ `_node_to_path_string(node)`.

Ví dụ:

```text
MIT --[tài trợ bởi]--> Google | Google --[sở hữu]--> DeepMind
```

#### `relation_history`

Được tạo từ `node.get_relation_history_text()`.

Ví dụ:

```text
MIT --[tài trợ bởi]--> Google
Google --[sở hữu]--> DeepMind
```

#### `relations`

Danh sách relation ứng viên sau backtrack filter, format như:

```text
1. [incoming] tài trợ bởi... (weight: 0.90)
2. [outgoing] hợp tác với... (weight: 0.80)
3. [outgoing] nghiên cứu về... (weight: 0.70)
4. [outgoing] đặt tại... (weight: 0.50)
```

#### Template prompt gốc

```text
Question: "{query}"
Entity: {entity_name}

Current reasoning path:
{current_path}

Previous relations followed:
{relation_history}

Available relations:
{relations}

Score each relation (1-10) as the next-hop option given the current path.
Consider:
- Direct relevance to the question's intent
- Whether the relation leads toward the answer
- The semantic connection between the relation and question
- Avoiding redundant backtracking unless it is useful for answering the question

Output ONLY a list of numbers in brackets, e.g., [8, 3, 6, 4]
```

### 3.3. Cách chọn group relation

Sau khi có `scored_relations`, code:

1. sort giảm dần theo score của relation
2. group theo key `(rel_desc, direction)`
3. giữ tối đa `width` group relation đầu tiên xuất hiện

Điều quan trọng: **trùng relation name chỉ được check bằng so sánh bằng nhau tuyệt đối**, không fuzzy, không semantic matching.

Ví dụ:

```python
("tài trợ bởi", "incoming")
("tài trợ bởi", "outgoing")
```

là **2 group khác nhau**.

### 3.4. Cách build entity candidates

Với mỗi relation group đã chọn, code gom toàn bộ target entity của group đó thành candidate list.

Nếu group có quá nhiều candidate:

```python
len(group_candidates) >= ENTITY_CANDIDATE_SAMPLE_THRESHOLD  # = 20
and len(group_candidates) > self.num_retain_entity           # = 5
```

thì random sample 5 entity, còn nhỏ hơn ngưỡng thì giữ hết.

### 3.5. Prompt đưa vào `score_entities()`

**Hàm:** `LLMPruning.score_entities()`

`entities_text` có dạng:

```text
1. Google: công ty công nghệ...
2. OpenAI: công ty AI...
3. Microsoft: ...
```

Prompt template nhận:

- `query`
- `current_path`
- `candidate_entities`

### 3.6. Công thức điểm

Code hiện tại:

```python
hop_score = rel_score * (max(entity_score, 0.0) / 10.0)
combined_score = node.score * hop_score
```

Ví dụ:

- `parent.score = 1.0`
- `rel_score = 9`
- `entity_score = 9`

thì:

```text
hop_score = 9 × (9/10) = 8.1
combined_score = 1.0 × 8.1 = 8.1
```

Sau đó node mới được lưu với score theo công thức này, và về sau frontier sẽ prune theo score giảm dần.

### 3.7. Prune frontier

Sau khi `_process_node()` song song xong cho toàn frontier hiện tại:

```python
frontier.sort(key=lambda n: n.score, reverse=True)
self.nodes_by_depth[self.current_depth] = frontier[:beam_width]
```

Tức là chỉ giữ top `width` node ở mỗi depth.

### 3.8. Early termination sau mỗi depth

Sau khi prune, code lại gọi `check_early_termination()`.

Nếu LLM trả `YES:` và không bật debug force-max-depth thì search dừng ngay, không đi tiếp depth sau.

---

## Bước 4: Generate Final Answer

**Hàm:** `ToGReasoning.generate_answer()` — `temperature=0.0`

### Dữ liệu đầu vào

- `query`
- `exploration_paths = all nodes đã duyệt`
- `conversation_history_context` nếu có
- `text_units = explorer.get_text_units_for_nodes(all_paths)`

### Prompt cuối

`generate_answer()` dùng `TOG_REASONING_PROMPT` với format:

```text
Given a question and the associated retrieved graph context, you are asked to answer the question using only the evidence provided below.

The retrieved context contains three evidence sections:
- CHUNKS
- ENTITIES
- RELATIONSHIPS

IMPORTANT CITATION RULES:
1. Every factual claim in the answer must include inline data citations.
2. Prefer citing Sources whenever a claim is grounded in CHUNKS.
3. Also cite Entities and Relationships when they directly support the claim.
4. Use the exact entity names and relationship identifiers as they appear below.
...

Question: {query}

Retrieved context:
{exploration_paths}
```

`exploration_paths` ở đây chính là output của `_format_paths()`:

```text
=== CHUNKS ===
...

=== ENTITIES ===
...

=== RELATIONSHIPS ===
...
```

### Ý nghĩa

- Final answer được sinh từ **toàn bộ path đã khám phá**, không chỉ frontier cuối.
- Prompt yêu cầu **citation nội dòng**.
- Nếu prompt template khác format placeholder, code có fallback generic prompt.

---

## Mẫu luồng dữ liệu thực tế trong một vòng depth

```text
1. current frontier nodes
   └─ get_text_units_for_nodes(frontier_nodes)
   └─ check_early_termination(frontier_nodes, text_units)

2. expand từng node
   └─ get_relations(entity_id)
   └─ _filter_backtrack_relations()
   └─ score_relations(query, entity_name, relations, relation_history, current_path)
   └─ group theo (relation_name, direction)
   └─ score_entities(query, current_path, entities)
   └─ tạo node mới + combined_score

3. prune
   └─ giữ top beam_width node ở depth hiện tại

4. check early termination lần nữa
   └─ nếu YES → return sớm
   └─ nếu NO → đi depth tiếp theo
```

---

## Tổng LLM Calls cho ví dụ trên

| Bước                                | Hàm                         | Calls |
| ----------------------------------- | --------------------------- | ----- |
| Depth-0 early check                 | `check_early_termination`   | 1     |
| Depth-1: score_relations × 3 nodes  | `LLMPruning.score_relations`| 3     |
| Depth-1: score_entities × 3 nodes   | `LLMPruning.score_entities` | 3     |
| Depth-1 early check                 | `check_early_termination`   | 1     |
| Final answer (nếu không early term) | `generate_answer`           | 1     |
| **Total (nếu term tại depth-1)**    |                             | **8** |
| **Total (nếu chạy hết depth-3)**    |                             | **~26** |

---

## ASCII Flow Đầy Đủ

```text
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
  │   ├─ _filter_backtrack_relations()
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

## Ghi chú cập nhật so với mô tả cũ

- `check_early_termination()` **có `text_units`** và **có `conversation_history_context`**.
- `generate_answer()` cũng nhận `text_units` và history context.
- `score_relations()` không chỉ chấm relation đơn lẻ; nó dùng `current_path` và `relation_history` để ưu tiên đường đi hợp lý.
- `_filter_backtrack_relations()` chỉ chặn **reverse edge ngay lập tức**, không phải toàn bộ cycle dài.
- `relations` và `entities` đều được build theo hướng **dedupe exact-match**, không fuzzy matching.
