# Plan: Fix ToG Search High Latency (>200s)

## Context

ToG search có latency >200s cho query đầu. Điều tra phát hiện nhiều bottleneck liên quan đến embedding và LLM calls.

---

## Root Causes

### Bug 1: Sequential node processing (search.py lines 277-377) 🔴 CRITICAL

Mỗi node trong frontier được xử lý tuần tự. Với width=3, depth=3: ~18 sequential LLM/embedding calls.

```python
for node in current_nodes:
    scored_relations = await self.pruning_strategy.score_relations(...)  # BLOCKS
    entity_scores = await self.pruning_strategy.score_entities(...)      # BLOCKS
```

### Bug 2: Query embedding lại 19 lần — không cache (pruning.py) 🔴 CRITICAL

Cùng query string bị embed lại tại mỗi node per depth, không có cross-call cache:

| Nơi embed query | File | Số lần gọi (width=3, depth=3) |
|---|---|---|
| `find_starting_entities_semantic` | exploration.py:251 | 1 |
| `SemanticPruning.score_relations` | pruning.py:263 | 3 nodes × 3 depths = **9** |
| `SemanticPruning.score_entities` | pruning.py:310 | 3 nodes × 3 depths = **9** |
| **TOTAL** | | **19 lần cùng query** |

### Bug 3: Relation embeddings tính lại từ đầu dù đã pre-compute khi index 🔴 CRITICAL

`embeddings.relationship.description` **đã được tính và lưu khi indexing** (hằng số `relationship_description_embedding` đã có trong `graphrag/config/embeddings.py`). Nhưng factory `get_tog_search_engine` **không load** chúng và **không truyền** vào `SemanticPruning`:

```python
# factory.py: line 342-347 — chỉ load entity embeddings
entity_text_embeddings = get_embedding_store(
    config_args=vector_store_args,
    embedding_name=entity_description_embedding,  # ← chỉ entity!
)

# SemanticPruning.score_relations — luôn tính lại relation embeddings
relation_embeddings = await self.embedding_model.aembed_batch(text_list=relation_texts)  # WASTEFUL
```

Tương tự với **query embedding**: cũng không được cache — tính lại 19 lần/search.

### Bug 4: `achat_stream` bypass cache trong LLMPruning (with_cache.py lines 81-83) 🟡 HIGH

`LLMPruning` dùng `achat_stream` → cache bị skip → với LLM pruning, **0 cache hits** cho bất kỳ query nào.

---

## Fix Plan

### Fix 1: Load relationship embeddings store và truyền vào SemanticPruning

**File**: `graphrag/query/factory.py` — function `get_tog_search_engine`

```python
from graphrag.config.embeddings import entity_description_embedding, relationship_description_embedding

def get_tog_search_engine(..., relationship_text_embeddings: BaseVectorStore | None = None):
    # Existing entity store load...
    
    # NEW: Load relationship embeddings store
    if relationship_text_embeddings is None:
        vector_store_args = config.vector_store.model_dump()
        relationship_text_embeddings = get_embedding_store(
            config_args=vector_store_args,
            embedding_name=relationship_description_embedding,
        )
    
    # Pass to SemanticPruning:
    elif config.tog_search.prune_strategy == "semantic":
        pruning_strategy = SemanticPruning(
            embedding_model=embedding_model,
            entity_embedding_store=entity_text_embeddings,
            relationship_embedding_store=relationship_text_embeddings,  # NEW
        )
```

**File**: `graphrag/query/structured_search/tog_search/pruning.py` — `SemanticPruning`

```python
class SemanticPruning(PruningStrategy):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        entity_embedding_store: BaseVectorStore | None = None,
        relationship_embedding_store: BaseVectorStore | None = None,  # NEW
    ):
        self.relationship_embedding_store = relationship_embedding_store
    
    async def score_relations(self, query, entity_name, relations, ...):
        # NEW: try pre-computed embeddings first
        if self.relationship_embedding_store:
            relation_embeddings = self._load_relation_embeddings(relations)
            # fallback aembed_batch for missing
        else:
            relation_embeddings = await self.embedding_model.aembed_batch(...)
```

`_load_relation_embeddings` pattern giống `_load_entity_embeddings` (pruning.py:327-373): lookup by relation id từ vector store, fallback aembed_batch cho missing.

### Fix 2: Cache query embedding trong search session

**File**: `graphrag/query/structured_search/tog_search/search.py`

Trước exploration loop, tính query embedding 1 lần và truyền xuống:
```python
# Before exploration loop
query_embedding: np.ndarray | None = None
if isinstance(self.pruning_strategy, SemanticPruning) and self.embedding_model:
    query_embedding = np.array(await self.embedding_model.aembed(text=effective_query))
```

Update signature của `score_relations` và `score_entities` trong `PruningStrategy` base class và `SemanticPruning`:
```python
async def score_relations(self, query, entity_name, relations, query_embedding=None)
async def score_entities(self, query, current_path, entities, query_embedding=None)
```

Nếu `query_embedding` được truyền vào thì dùng trực tiếp, không gọi `aembed(query)`.

### Fix 3: Parallelize frontier node processing

**File**: `graphrag/query/structured_search/tog_search/search.py`

Tách logic per-node thành method, dùng `asyncio.gather`:
```python
import asyncio

async def _process_node(self, query, node, query_embedding=None):
    relations = self.explorer.get_relations(node.entity_id)
    if not relations:
        return [], []
    scored_relations, p_metrics = await self.pruning_strategy.score_relations(
        query, node.entity_name, relations, query_embedding=query_embedding
    )
    # ... build candidate_data, sample ...
    entity_scores, e_metrics = await self.pruning_strategy.score_entities(
        query=query, current_path=..., entities=..., query_embedding=query_embedding
    )
    # ... build ExplorationNode list ...
    return new_nodes, [p_metrics, e_metrics]

# In exploration loop, replace for loop:
tasks = [self._process_node(query, node, query_embedding) for node in current_nodes]
results = await asyncio.gather(*tasks)
for new_nodes, metrics_list in results:
    next_level_nodes.extend(new_nodes)
    for m in metrics_list:
        yield ("", [], m, "")
```

**Note**: `yield` inside `asyncio.gather` task không hoạt động — metrics cần collect từ return value của task, không yield trong task. Cần tách generator yields ra ngoài gather.

### Fix 4: Switch `achat_stream` → `achat` trong LLMPruning

**File**: `graphrag/query/structured_search/tog_search/pruning.py`

```python
# Trước:
response = ""
async for chunk in self.model.achat_stream(prompt=prompt, history=[], ...):
    response += chunk

# Sau:
response_obj = await self.model.achat(prompt=prompt, history=[], ...)
response = response_obj.output.content
```

Áp dụng cho cả `score_relations` và `score_entities` trong `LLMPruning`.

---

## Files to Modify

| File | Change |
|------|--------|
| `graphrag/query/factory.py` | Load relationship embeddings store; truyền vào SemanticPruning |
| `graphrag/query/structured_search/tog_search/pruning.py` | Add `relationship_embedding_store`; cache query_embedding param; switch achat_stream → achat |
| `graphrag/query/structured_search/tog_search/search.py` | Pre-compute query_embedding; parallelize node loop với asyncio.gather |

## Existing Utilities to Reuse

- `get_embedding_store()` — `graphrag/utils/api.py` — đã dùng cho entity, dùng lại cho relationship
- `relationship_description_embedding` — `graphrag/config/embeddings.py:8` — constant đã có
- `_load_entity_embeddings()` pattern — `pruning.py:327-373` — copy pattern cho `_load_relation_embeddings()`
- `BaseVectorStore.search_by_id()` — đã hoạt động cho entities, dùng lại cho relations

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Query embeddings per search | 19 | 1 |
| Relation embeddings per search | 27+ | 0 (pre-computed) |
| Node processing | Sequential | Parallel per depth |
| LLM pruning cache | 0% | ~80%+ for repeated queries |

## Verification

```bash
# Benchmark trước fix
time graphrag query --root ./my-graphrag-project --method tog "test query"

# Benchmark sau fix
time graphrag query --root ./my-graphrag-project --method tog "test query"
# Lần 2 (test LLM cache cho LLMPruning):
time graphrag query --root ./my-graphrag-project --method tog "test query"

# Unit tests
pytest ./tests/unit -k tog -v
```

Verify answer quality không đổi (parallelization thay đổi execution order không thay đổi logic).
