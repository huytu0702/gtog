# ToG (Think-on-Graph) Search - Chi Tiết Từng Bước Hoạt Động

## Tổng Quan

ToG (Think-on-Graph) là thuật toán tìm kiếm sâu trên đồ thị tri thức (Knowledge Graph), kết hợp khả năng khám phá đồ thị với lập luận của LLM. ToG mở rộng từ các phương pháp RAG truyền thống bằng cách **iteratively** (lặp đi lặp lại) khám phá đồ thị theo nhiều bước (depth) và nhiều nhánh (width) song song.

### Các Thành Phần Chính

```
ToG Search
├── GraphExplorer       → Khám phá đồ thị, tìm entities ban đầu
├── PruningStrategy    → Đánh giá relations/entities bằng LLM
├── ToGReasoning       → Tổng hợp câu trả lời từ các đường dẫn đã khám phá
└── ToGSearchState     → Quản lý trạng thái exploration
```

---

## Luồng Hoạt Động Chi Tiết

### Bước 1: Khởi Tạo & Chuẩn Bị Query

```python
# search.py:171-189
effective_query = query
if conversation_history:
    past_questions = "\n".join(conversation_history.get_user_turns(max_user_turns=5))
    if past_questions:
        effective_query = f{past_questions"{query}\n}"

history_context, _ = conversation_history.build_context(...)
```

**Mục đích:** 
- Enrich query với lịch sử hội thoại để context đầy đủ hơn
- Tạo history context string cho reasoning phase

---

### Bước 2: Entity Linking - Tìm Entities Khởi Đầu

ToG sử dụng **embedding similarity** (như trong paper gốc) để tìm các entities liên quan nhất đến câu hỏi.

```python
# search.py:190-208
if self.embedding_model:
    starting_entities = await self.explorer.find_starting_entities_semantic(
        effective_query, top_k=self.width
    )
else:
    starting_entities = self.explorer.find_starting_entities(
        effective_query, top_k=self.width
    )
```

#### Cách hoạt động của Semantic Entity Linking:

```python
# exploration.py:211-255
async def find_starting_entities_semantic(self, query: str, top_k: int = 3):
    # 1. Embed query
    query_embedding = await self.embedding_model.aembed(text=query)
    
    # 2. Embed tất cả entities trong knowledge graph
    await self._compute_entity_embeddings()
    
    # 3. Tính dot product scores (similarity)
    scores = np.dot(self._entity_embeddings, query_emb)
    
    # 4. Lấy top-k entities
 np.argsort(scores)[::-1    top_indices =][:top_k]
    result = [self.entity_list[i].title for i in top_indices]
```

**Ví dụ thực tế:**

```
Query: "Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?"

Knowledge Graph có các entities:
- "Pulitzer Prize" (score: 0.95)
- "2020" (score: 0.82)
- "Tác giả" (score: 0.78)
- "Hội sách" (score: 0.65)
- ...

→ Starting entities: ["Pulitzer Prize"] (top-3 = width = 3)
```

---

### Bước 3: Khởi Tạo Search State

```python
# search.py:210-236
state = ToGSearchState(
    query=query,
    current_depth=0,
    nodes_by_depth={0: []},
    finished_paths=[],
    max_depth=self.depth,
    beam_width=self.width,
)

# Tạo initial nodes từ starting entities
for entity_id in starting_entities:
    entity_info = self.explorer.get_full_entity_info(entity_id)
    initial_node = ExplorationNode(
        entity_id=entity_id,
        entity_name=name,
        entity_description=full_description,
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )
    state.add_node(initial_node)
```

**Cấu trúc ExplorationNode:**
```python
@dataclass
class ExplorationNode:
    entity_id: str
    entity_name: str
    entity_description: str
    depth: int              # Độ sâu hiện tại (0, 1, 2, ...)
    score: float            # Điểm kết hợp (relation_score × entity_score)
    parent: ExplorationNode # Node cha (None cho root)
    relation_from_parent: str  # Relation từ cha đến node này
```

---

### Bước 4: Exploration Loop - Vòng Lặp Khám Phá

ToG thực hiện khám phá theo từng **depth** (độ sâu). Tại mỗi depth, ToG:
1. Lấy các nodes hiện tại (frontier)
2. Với mỗi node, tìm tất cả relations
3. **Score relations** bằng LLM
4. **Score entities** (target entities từ relations) bằng LLM  
5. Tạo new nodes với combined score
6. **Prune** (cắt bớt) chỉ giữ lại top-k (beam_width) nodes

```python
# search.py:238-353
while state.current_depth < state.max_depth:
    current_nodes = state.get_current_frontier()
    
    if not current_nodes:
        break
    
    next_depth = state.current_depth + 1
    next_level_nodes = []
    
    # Explore each node in current frontier
    for node in current_nodes:
        # 1. Get relations từ entity
        relations = self.explorer.get_relations(node.entity_id)
        
        if not relations:
            continue
        
        # 2. Score relations using LLM
        scored_relations, pruning_metrics = await self.pruning_strategy.score_relations(
            query, node.entity_name, relations
        )
        
        # 3. Keep top entities based on scores
        scored_relations.sort(key=lambda x: x[4], reverse=True)
        top_relations = scored_relations[: self.num_retain_entity]
        
        # 4. Build entity candidates for second-stage pruning
        candidate_data = []
        for rel_desc, target_id, direction, weight, rel_score in top_relations:
            target_info = self.explorer.get_full_entity_info(target_id)
            if target_info:
                candidate_data.append(...)
        
        # 5. Score entities using LLM
        entity_candidates = [(target_id, target_name, target_full_desc) for ...]
        current_path = self._node_to_path_string(node)
        
        entity_scores, entity_metrics = await self.pruning_strategy.score_entities(
            query=query,
            current_path=current_path,
            entities=entity_candidates,
        )
        
        # 6. Create new nodes với combined score
        for idx, ... in enumerate(candidate_data):
            entity_score = entity_scores[idx] if idx < len(entity_scores) else 5.0
            combined_score = rel_score * (max(entity_score, 0.0) / 10.0)
            new_node = ExplorationNode(...)
            next_level_nodes.append(new_node)
    
    # 7. Add next level nodes to state
    state.nodes_by_depth[next_depth] = next_level_nodes
    
    # 8. Prune to beam width (giữ lại chỉ width nodes tốt nhất)
    state.current_depth = next_depth
    state.prune_current_frontier()
```

#### Chi Tiết Từng Bước Trong Exploration:

##### 4a. Score Relations (LLM Pruning)

```python
# pruning.py:112-172
async def score_relations(self, query, entity_name, relations):
    # Giới hạn số relations để tránh token quá lớn
    if len(relations) > self.max_relations_for_llm:
        scored_subset_relations = sorted_relations[:self.max_relations_for_llm]
    
    # Tạo prompt cho LLM
    prompt = TOG_RELATION_SCORING_PROMPT.format(
        query=query,
        entity_name=entity_name,
        relations=relations_text
    )
    
    # Gọi LLM
    response = await self.model.achat_stream(prompt=prompt, ...)
    
    # Parse scores từ response
    scores = self._parse_scores(response, len(relations))
    
    # Return: [(rel_desc, target_id, direction, weight, score), ...]
    scored_relations = [...]
```

**Prompt mẫu:**
```
Question: "Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?"
Entity: Pulitzer Prize

Available relations:
1. [outgoing] award.award_winning_work
2. [outgoing] award.award_category
3. [incoming] person.awards_received

Score each relation (1-10) based on how likely it leads to answering the question.
Output: [9, 3, 5]
```

**Ví dụ kết quả:**
```
Relation 1 (award_winning_work): 9/10 → trực tiếp dẫn đến tác phẩm đoạt giải
Relation 2 (award_category): 3/10 → không liên quan đến người thắng cuộc
Relation 3 (awards_received): 5/10 → có thể liên quan đến người nhận giải
```

##### 4b. Score Entities (LLM Pruning)

```python
# pruning.py:174-209
async def score_entities(self, query, current_path, entities):
    # Tạo prompt với current path và candidate entities
    prompt = TOG_ENTITY_SCORING_PROMPT.format(
        query=query,
        current_path=current_path,
        candidate_entities=entities_text
    )
    
    response = await self.model.achat_stream(prompt=prompt, ...)
    scores = self._parse_scores(response, len(entities))
```

**Prompt mẫu:**
```
Question: "Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?"
Current exploration path: Pulitzer Prize -> award.award_winning_work

Candidate entities to explore:
1. The Nickel Boys
2. Pulitzer Prize Ceremony
3. 2020 Awards

Score each entity (1-10) based on relevance to the question.
Output: [10, 4, 7]
```

##### 4c. Combine Scores

```python
# search.py:322-346
# Kết hợp relation score và entity score
combined_score = rel_score * (max(entity_score, 0.0) / 10.0)

# Ví dụ:
# rel_score = 9, entity_score = 10
# combined_score = 9 * (10/10) = 9.0 (rất cao - đường dẫn tốt)
```

##### 4d. Prune (Beam Search)

```python
# state.py:52-56
def prune_current_frontier(self):
    frontier = self.get_current_frontier()
    frontier.sort(key=lambda n: n.score, reverse=True)
    self.nodes_by_depth[self.current_depth] = frontier[: self.beam_width]
```

**Ví dụ với width=3:**
```
Depth 1 candidates (sau khi score):
- Pulitzer Prize --[award_winning_work]--> The Nickel Boys (score: 9.0)
- Pulitzer Prize --[award_winning_work]--> The Night Watchman (score: 8.5)
- Pulitzer Prize --[award_winning_work]--> The Plot (score: 7.0)
- Pulitzer Prize --[award_category]--> Fiction (score: 3.0)
- ...

→ Sau prune (width=3): Giữ lại 3 candidates đầu tiên
```

---

### Bước 5: Early Termination Check

Tại mỗi depth, ToG kiểm tra xem đã có đủ thông tin để trả lời câi chưau hỏ. Nếu đủ → dừng sớm, không cần explore sâu hơn.

```python
# search.py:368-398
should_terminate, answer, early_term_metrics = await self.reasoning_module.check_early_termination(
    query, state.get_current_frontier(), conversation_history_context=history_context
)

if should_terminate and answer:
    # Return answer sớm
    yield (answer, reasoning_paths, early_term_metrics, early_context_text)
```

**Cách hoạt động của Early Termination:**

```python
# reasoning.py:233-279
async def check_early_termination(self, query, current_nodes, ...):
    paths_text = self._format_paths(current_nodes[:3])
    
    prompt = f"""Question: {query}
Current exploration paths:
{paths_text}

Can you answer the question with high confidence based on these paths?
Respond with:
- "YES: [answer]" if you can answer confidently
- "NO: [reason]" if more exploration is needed
"""
    
    response = await self.model.achat_stream(prompt=prompt, ...)
    
    if response.strip().upper().startswith("YES:"):
        answer = response[4:].strip()
        return True, answer, metrics
    
    return False, None, metrics
```

**Ví dụ:**
```
Query: "Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?"

Exploration paths:
- Pulitzer Prize --[award_warding_work]--> The Nickel Boys
- The Nickel Boys --[written_by]--> Colson Whitehead

LLM check: "YES: Colson Whitehead là tác giả của The Nickel Boys, 
cuốn sách đoạt giải Pulitzer Fiction năm 2020."

→ Early termination! Không cần explore thêm depth
```

---

### Bước 6: Final Reasoning - Tổng Hợp Câu Trả Lời

Sau khi hoàn thành exploration (đạt max_depth hoặc early termination), ToG tổng hợp tất cả các đường dẫn đã khám phá để tạo câu trả lời cuối cùng.

```python
# search.py:400-477
all_paths = []
for depth_nodes in state.nodes_by_depth.values():
    all_paths.extend(depth_nodes)

# Format paths với rich context
context_text = self.reasoning_module._format_paths(all_paths)

# Generate final answer
answer, reasoning_paths, answer_metrics = await self.reasoning_module.generate_answer(
    query, all_paths, conversation_history_context=history_context
)
```

#### Format Exploration Paths:

```python
# reasoning.py:114-193
def _format_paths(self, nodes):
    # Collect unique entities và relationships
    seen_entities = {}
    relationships = []
    
    for node in nodes:
        # Lấy thông tin entity
        entity_desc = node.entity_full_description or node.entity_description
        seen_entities[node.entity_name] = entity_desc
        
        # Lấy relationship info
        if node.parent:
            relationships.append((
                node.parent.entity_name,
                node.relation_from_parent,
                node.entity_name,
                node.parent.entity_full_description,
                node.entity_full_description,
            ))
    
    # Format output
    output = "=== ENTITIES ===\n"
    for entity_name, description in seen_entities.items():
        output += f"\n[{entity_name}]\nDescription: {description}"
    
    output += "\n\n=== RELATIONSHIPS ===\n"
    for source, relation, target, ...:
        output += f"\n- {source} --[{relation}]--> {target}"
```

**Ví dụ context_text đầu ra:**

```
=== ENTITIES ===

[Pulitzer Prize]
Description: An annual award given to outstanding achievements in journalism, literature, and musical composition.

[The Nickel Boys]
Description: A 2019 novel by Colson Whitehead about a young Black man attending a brutal Florida reform school.

[Colson Whitehead]
Description: An American novelist known for works exploring African American history and experiences.

[2020]
Description: The year 2020, during which the Pulitzer Prizes were awarded.

=== RELATIONSHIPS ===

- Pulitzer Prize --[award_winning_work]--> The Nickel Boys
  Description: The Nickel Boys won the Pulitzer Prize

- The Nickel Boys --[written_by]--> Colson Whitehead
  Description: Colson Whitehead is the author
```

#### Generate Final Answer:

```python
# reasoning.py:30-112
async def generate_answer(self, query, exploration_paths, ...):
    paths_text = self._format_paths(exploration_paths)
    
    prompt = TOG_REASONING_PROMPT.format(
        query=query,
        exploration_paths=paths_text
    )
    
    answer = ""
    async for chunk in self.model.achat_stream(prompt=prompt, ...):
        answer += chunk
    
    return answer, reasoning_paths, metrics
```

**Prompt reasoning cuối cùng:**

```
Given a question and the associated retrieved knowledge graph triplets, 
answer the question with these triplets and your knowledge.

IMPORTANT: When citing entities, use exact names from Exploration Paths.

Question: Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?

Exploration Paths:
=== ENTITIES ===
[Pulitzer Prize]
[The Nickel Boys]
[Colson Whitehead]

=== RELATIONSHIPS ===
- Pulitzer Prize --[award_winning_work]--> The Nickel Boys
- The Nickel Boys --[written_by]--> Colson Whitehead
```

---

## Ví Dụ Toàn Bộ Quy Trình

### Query:
```
"Ai là tác giả của cuốn sách đoạt giải Pulitzer năm 2020?"
```

### Cấu hình:
- width = 3 (số nhánh tối đa mỗi depth)
- depth = 3 (độ sâu tối đa)
- num_retain_entity = 5 (số entities giữ lại sau relation scoring)

### Thực Hiện:

#### Depth 0: Entity Linking
```
Query embedding → Compare với tất cả entities trong KG
→ Top-3 starting entities: ["Pulitzer Prize", "2020", "Pulitzer Prize Ceremony"]
→ Initial nodes: Pulitzer Prize (depth=0, score=1.0)
```

#### Depth 1: Relation + Entity Scoring
```
From: Pulitzer Prize
Relations found:
1. award_winning_work → The Nickel Boys (weight: 0.9)
2. award_winning_work → The Night Watchman (weight: 0.85)
3. award_category → Fiction (weight: 0.7)
4. award_year → 2020 (weight: 0.8)
5. ...

LLM Score Relations:
- award_winning_work: 9/10 (directly leads to winning works)
- award_category: 3/10
- award_year: 8/10 (relevant to year)

Top-5 relations kept → Get target entities
→ Candidates: [The Nickel Boys, The Night Watchman, 2020, ...]

LLM Score Entities:
- The Nickel Boys: 10/10 (matches year 2020 + Pulitzer)
- The Night Watchman: 7/10 (not 2020)
- 2020: 8/10 (matches year)

Combined Scores:
- The Nickel Boys: 9 × (10/10) = 9.0
- 2020: 8 × (8/10) = 6.4
- The Night Watchman: 9 × (7/10) = 6.3

Prune (width=3): Keep top 3
→ Nodes at depth-1: [The Nickel Boys, 2020, The Night Watchman]
```

#### Early Termination Check at Depth 1
```
Paths:
- Pulitzer Prize --[award_winning_work]--> The Nickel Boys

LLM: "YES: Có thể trả lời. The Nickel Boys đoạt giải Pulitzer năm 2020. 
Cần thêm thông tin về tác giả."

→ Continue to depth 2
```

#### Depth 2: Continue Exploration
```
From: The Nickel Boys
Relations:
1. written_by → Colson Whitehead
2. genre → Fiction
3. published_by → Doubleday
...

LLM Score Relations:
- written_by: 10/10 (directly gives author!)
- genre: 2/10
- published_by: 4/10

Target entities: [Colson Whitehead, Fiction, Doubleday]

LLM Score Entities:
- Colson Whitehead: 10/10 (the author!)
- Fiction: 1/10
- Doubleday: 5/10

Combined Scores:
- Colson Whitehead: 10 × (10/10) = 10.0 (MAX!)

Prune → Keep [Colson Whitehead]
```

#### Early Termination at Depth 2
```
Paths:
- Pulitzer Prize --[award_winning_work]--> The Nickel Boys --[written_by]--> Colson Whitehead

LLM: "YES: Colson Whitehead là tác giả của The Nickel Boys, 
cuốn sách đoạt giải Pulitzer Fiction năm 2020."

→ EARLY TERMINATION! Return answer
```

### Final Answer:
```
Dựa trên exploration paths:
- Pulitzer Prize --[award_winning_work]--> The Nickel Boys
- The Nickel Boys --[written_by]--> Colson Whitehead

Câu trả lời: Colson Whitehead là tác giả của cuốn sách "The Nickel Boys", 
cuốn sách đoạt giải Pulitzer Fiction năm 2020.

[Data: Entities (Colson Whitehead, The Nickel Boys, Pulitzer Prize)]
```

---

## So Sánh Các Thành Phần

### Pruning Strategies

| Strategy | Scoring Method | Pros | Cons |
|----------|---------------|------|------|
| **LLMPruning** | Dùng LLM đánh giá | Chính xác cao, hiểu semantic | Tốn LLM calls |
| **SemanticPruning** | Embedding similarity | Nhanh, không tốn LLM | Ít chính xác hơn |
| **BM25Pruning** | BM25 lexical matching | Rất nhanh | Chỉ matching từ vựng |

### Parameters

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `width` | 3 | Số nhánh tối đa mỗi depth (beam width) |
| `depth` | 3 | Độ sâu tối đa của exploration |
| `num_retain_entity` | 5 | Số entities giữ lại sau relation scoring |
| `max_relations_for_llm` | 10 | Giới hạn relations cho LLM scoring |

---

## Metrics Thu Thập

```python
@dataclass
class ToGMetrics:
    llm_calls: int                    # Tổng số LLM calls
    prompt_tokens: int                # Tổng prompt tokens
    output_tokens: int                # Tổng output tokens
    exploration_llm_calls: int        # LLM calls trong exploration
    reasoning_llm_calls: int         # LLM calls trong reasoning
    embedding_calls: int              # Số lần embedding
    embedding_tokens: int             # Tokens cho embedding
```

---

## Tóm Tắt Luồng Hoạt Động

```
┌─────────────────────────────────────────────────────────────────┐
│                    ToG Search Pipeline                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. ENTITY LINKING                                               │
│    Query → Embedding → Top-K Entities (width)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. EXPLORATION LOOP (while depth < max_depth)                  │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ For each node in frontier:                              │ │
│    │   a. Get all relations                                  │ │
│    │   b. LLM Score Relations (1-10)                         │ │
│    │   c. LLM Score Target Entities (1-10)                   │ │
│    │   d. Combined Score = rel_score × entity_score          │ │
│    │   e. Create new nodes with combined score               │ │
│    └─────────────────────────────────────────────────────────┘ │
│                              │                                  │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ Prune: Keep top-K (beam_width) nodes                    │ │
│    └─────────────────────────────────────────────────────────┘ │
│                              │                                  │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ Early Termination Check:                                │ │
│    │   LLM có thể trả lời với confidence cao?                │ │
│    │   YES → Return answer early                             │ │
│    │   NO  → Continue to next depth                         │ │
│    └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. FINAL REASONING                                             │
│    Format all paths → LLM Generate Final Answer                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Final Answer + Reasoning Paths + Metrics                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ưu Điểm của ToG So Với RAG Truyền Thống

1. **Iterative Exploration**: Không chỉ retrieve một lần mà explore nhiều bước
2. **Beam Search**: Khám phá nhiều đường dẫn song song, không bỏ lỡ context quan trọng
3. **LLM-Guided Pruning**: Dùng LLM để đánh giá relevance, không chỉ vector similarity
4. **Transparent Reasoning**: Có thể truy vết được đường dẫn nào dẫn đến câu trả lời
5. **Early Termination**: Tiết kiệm cost khi đã tìm được câu trả lời sớm
