# ToG Conversation History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `ConversationHistory` support to `ToGSearch` following the same pattern as `LocalSearch`.

**Architecture:** History is used in two places: (1) enriching the entity-linking query so follow-up questions resolve correctly, and (2) prepending a formatted history block to the final reasoning prompt. Pruning prompts stay unchanged (original query only). Three files touched, no new classes, no config changes.

**Tech Stack:** Python, existing `ConversationHistory` from `graphrag/query/context_builder/conversation_history.py`

---

### Task 1: Update `ToGReasoning` to accept history context

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/reasoning.py`

The reasoning module's `generate_answer()` and `check_early_termination()` build prompts from scratch. We add a `conversation_history_context: str = ""` parameter to each — when non-empty, it is prepended before the exploration paths so the LLM knows what was asked before.

**Step 1: Write the failing test**

Create `tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from graphrag.query.structured_search.tog_search.reasoning import ToGReasoning
from graphrag.query.structured_search.tog_search.state import ExplorationNode


def _make_node(name: str) -> ExplorationNode:
    return ExplorationNode(
        entity_id="e1",
        entity_name=name,
        entity_description="desc",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )


@pytest.mark.asyncio
async def test_generate_answer_includes_history_context():
    """History context appears in the prompt sent to the LLM."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "answer"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]
    history_ctx = "-----Conversation History-----\nuser|Who is Entity A?"

    await reasoning.generate_answer("Tell me more", nodes, conversation_history_context=history_ctx)

    assert len(captured_prompt) == 1
    assert history_ctx in captured_prompt[0]


@pytest.mark.asyncio
async def test_generate_answer_no_history_unchanged():
    """When no history, prompt is unaffected."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "answer"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]

    await reasoning.generate_answer("Tell me more", nodes)

    assert len(captured_prompt) == 1
    assert "Conversation History" not in captured_prompt[0]


@pytest.mark.asyncio
async def test_check_early_termination_includes_history():
    """History context appears in early termination prompt."""
    captured_prompt = []

    async def fake_stream(prompt, history, model_parameters):
        captured_prompt.append(prompt)
        yield "NO: need more"

    mock_model = MagicMock()
    mock_model.achat_stream = fake_stream

    reasoning = ToGReasoning(model=mock_model)
    nodes = [_make_node("Entity A")]
    history_ctx = "-----Conversation History-----\nuser|Previous question"

    should_terminate, answer, _ = await reasoning.check_early_termination(
        "Follow-up?", nodes, conversation_history_context=history_ctx
    )

    assert history_ctx in captured_prompt[0]
    assert should_terminate is False
```

**Step 2: Run test to verify it fails**

```bash
cd F:/KL/gtog
pytest tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py -v
```

Expected: `FAILED` — `generate_answer()` and `check_early_termination()` don't accept `conversation_history_context`.

**Step 3: Implement — update `reasoning.py`**

In `graphrag/query/structured_search/tog_search/reasoning.py`:

Change `generate_answer` signature (line 30):
```python
async def generate_answer(
    self,
    query: str,
    exploration_paths: List[ExplorationNode],
    conversation_history_context: str = "",
) -> Tuple[str, List[str], ReasoningMetrics]:
```

Inside `generate_answer`, before building `prompt`, prepend history. Replace the line:
```python
prompt = prompt_template.format(query=query, exploration_paths=paths_text)
```
with:
```python
history_prefix = f"{conversation_history_context}\n\n" if conversation_history_context.strip() else ""
prompt = history_prefix + prompt_template.format(query=query, exploration_paths=paths_text)
```

Do the same for the fallback `except KeyError` block — prepend `history_prefix` to the manually-built prompt string.

Change `check_early_termination` signature (line 231):
```python
async def check_early_termination(
    self,
    query: str,
    current_nodes: List[ExplorationNode],
    conversation_history_context: str = "",
) -> Tuple[bool, str | None, ReasoningMetrics]:
```

Inside `check_early_termination`, prepend history to the prompt. Replace:
```python
prompt = f"""Question: {query}
```
with:
```python
history_prefix = f"{conversation_history_context}\n\n" if conversation_history_context.strip() else ""
prompt = f"""{history_prefix}Question: {query}
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py -v
```

Expected: 3 tests PASS.

**Step 5: Commit**

```bash
cd F:/KL/gtog
git add graphrag/query/structured_search/tog_search/reasoning.py tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py
git commit -m "feat: add conversation_history_context param to ToGReasoning"
```

---

### Task 2: Update `ToGSearch` to accept and use `conversation_history`

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/search.py`

Three methods need updating: `search()`, `stream_search()`, `_stream_search_with_metrics()`.

Inside `_stream_search_with_metrics`:
- **Entity linking**: build an enriched query string from history before calling `find_starting_entities_semantic`
- **Reasoning**: format history using `conversation_history.build_context()` and pass result to `generate_answer()` and `check_early_termination()`

**Step 1: Write the failing test**

Create `tests/unit/query/structured_search/tog_search/test_tog_search_history.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
    ConversationRole,
)


def _make_engine(captured_queries: list) -> ToGSearch:
    """Build a minimal ToGSearch that records what query is used for entity linking."""

    async def fake_find_semantic(query, top_k):
        captured_queries.append(query)
        return []  # No entities — triggers early exit

    mock_explorer = MagicMock()
    mock_explorer.find_starting_entities_semantic = fake_find_semantic

    mock_pruning = MagicMock()
    mock_reasoning = MagicMock()

    engine = ToGSearch.__new__(ToGSearch)
    engine.explorer = mock_explorer
    engine.pruning_strategy = mock_pruning
    engine.reasoning_module = mock_reasoning
    engine.embedding_model = MagicMock()  # triggers semantic path
    engine.width = 2
    engine.depth = 2
    engine.num_retain_entity = 3
    engine.callbacks = []
    engine._debug = False
    engine.model = MagicMock()
    engine.tokenizer = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_search_enriches_entity_query_with_history():
    """History user turns are appended to entity-linking query."""
    captured = []
    engine = _make_engine(captured)

    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "Tell me about Inception")
    history.add_turn(ConversationRole.ASSISTANT, "Inception is a film...")

    result = await engine.search("Who directed it?", conversation_history=history)

    assert len(captured) == 1
    assert "Who directed it?" in captured[0]
    assert "Tell me about Inception" in captured[0]


@pytest.mark.asyncio
async def test_search_no_history_uses_original_query():
    """Without history the query is used as-is."""
    captured = []
    engine = _make_engine(captured)

    result = await engine.search("Who directed it?", conversation_history=None)

    assert captured[0] == "Who directed it?"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/query/structured_search/tog_search/test_tog_search_history.py -v
```

Expected: `FAILED` — `search()` doesn't accept `conversation_history`.

**Step 3: Implement — update `search.py`**

At the top of `graphrag/query/structured_search/tog_search/search.py`, add the import:
```python
from graphrag.query.context_builder.conversation_history import ConversationHistory
```

Update `search()` signature (line 89):
```python
async def search(
    self,
    query: str,
    conversation_history: ConversationHistory | None = None,
) -> SearchResult:
```

Update the internal call to `_stream_search_with_metrics` inside `search()` (line 98):
```python
async for (
    chunk,
    paths,
    chunk_metrics,
    ctx_text,
) in self._stream_search_with_metrics(query, conversation_history):
```

Update `stream_search()` signature (line 144):
```python
async def stream_search(
    self,
    query: str,
    conversation_history: ConversationHistory | None = None,
) -> AsyncGenerator[str, None]:
```

Update the call to `_stream_search_with_metrics` inside `stream_search()` (line 146):
```python
async for chunk, _, _, _ in self._stream_search_with_metrics(query, conversation_history):
```

Update `_stream_search_with_metrics()` signature (line 150):
```python
async def _stream_search_with_metrics(
    self,
    query: str,
    conversation_history: ConversationHistory | None = None,
) -> AsyncGenerator[...]:
```

Inside `_stream_search_with_metrics`, before entity linking (before the `if self.embedding_model:` block), add:

```python
# Enrich query for entity linking with previous user questions
effective_query = query
if conversation_history:
    past_questions = "\n".join(
        conversation_history.get_user_turns(max_user_turns=5)
    )
    if past_questions:
        effective_query = f"{query}\n{past_questions}"
```

Then change the entity-linking calls to use `effective_query`:
```python
# was: starting_entities = await self.explorer.find_starting_entities_semantic(query, ...)
starting_entities = await self.explorer.find_starting_entities_semantic(
    effective_query, top_k=self.width
)
# (non-embedding fallback)
starting_entities = self.explorer.find_starting_entities(
    effective_query, top_k=self.width
)
```

Build history context string once, before the exploration loop:
```python
history_context = ""
if conversation_history:
    history_context, _ = conversation_history.build_context(
        include_user_turns_only=False,
        max_qa_turns=5,
        recency_bias=False,
    )
```

Pass `history_context` to both reasoning calls:

`check_early_termination` call (line ~286):
```python
(
    should_terminate,
    answer,
    early_term_metrics,
) = await self.reasoning_module.check_early_termination(
    query, state.get_current_frontier(), conversation_history_context=history_context
)
```

`generate_answer` call (line ~336):
```python
(
    answer,
    reasoning_paths,
    answer_metrics,
) = await self.reasoning_module.generate_answer(
    query, all_paths, conversation_history_context=history_context
)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/query/structured_search/tog_search/test_tog_search_history.py -v
pytest tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
cd F:/KL/gtog
git add graphrag/query/structured_search/tog_search/search.py tests/unit/query/structured_search/tog_search/test_tog_search_history.py
git commit -m "feat: add conversation_history support to ToGSearch"
```

---

### Task 3: Thread `conversation_history` through the API layer

**Files:**
- Modify: `graphrag/api/query.py`

Both `tog_search()` and `tog_search_streaming()` call `search_engine.search(query=query)` / `stream_search(query=query)`. We add the parameter and pass it through.

**Step 1: Write the failing test**

Create `tests/unit/api/test_tog_api_history.py`:

```python
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
    ConversationRole,
)


@pytest.mark.asyncio
async def test_tog_search_api_passes_history_to_engine():
    """tog_search() passes conversation_history to search_engine.search()."""
    captured_calls = {}

    async def fake_search(query, conversation_history=None):
        captured_calls["query"] = query
        captured_calls["history"] = conversation_history
        mock_result = MagicMock()
        mock_result.response = "answer"
        mock_result.context_data = {}
        return mock_result

    mock_engine = MagicMock()
    mock_engine.search = fake_search

    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "Previous question")

    with patch("graphrag.api.query.get_tog_search_engine", return_value=mock_engine), \
         patch("graphrag.api.query.read_indexer_entities", return_value=[]), \
         patch("graphrag.api.query.read_indexer_relationships", return_value=[]), \
         patch("graphrag.api.query.get_embedding_store", return_value=MagicMock()), \
         patch("graphrag.api.query.init_loggers"):

        from graphrag.api.query import tog_search

        mock_config = MagicMock()
        mock_config.vector_store = {}

        await tog_search(
            config=mock_config,
            entities=pd.DataFrame(),
            relationships=pd.DataFrame(),
            query="Current question",
            conversation_history=history,
        )

    assert captured_calls["history"] is history
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/api/test_tog_api_history.py -v
```

Expected: `FAILED` — `tog_search()` doesn't accept `conversation_history`.

**Step 3: Implement — update `api/query.py`**

Add import near top of `graphrag/api/query.py` (alongside existing query imports):
```python
from graphrag.query.context_builder.conversation_history import ConversationHistory
```

Update `tog_search()` signature (around line 1232):
```python
async def tog_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    conversation_history: ConversationHistory | None = None,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame],
]:
```

Update the `search_engine.search()` call inside `tog_search()` (line ~1281):
```python
result = await search_engine.search(query=query, conversation_history=conversation_history)
```

Update `tog_search_streaming()` signature (around line 1287):
```python
def tog_search_streaming(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    conversation_history: ConversationHistory | None = None,
    callbacks: list[QueryCallbacks] | None = None,
    entity_text_embeddings: Optional[BaseVectorStore] = None,
    verbose: bool = False,
) -> AsyncGenerator:
```

Update the `stream_search()` call inside `tog_search_streaming()` (line ~1332):
```python
return search_engine.stream_search(query=query, conversation_history=conversation_history)
```

**Step 4: Run all tests**

```bash
pytest tests/unit/api/test_tog_api_history.py tests/unit/query/structured_search/tog_search/ -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd F:/KL/gtog
git add graphrag/api/query.py tests/unit/api/test_tog_api_history.py
git commit -m "feat: thread conversation_history through ToG API layer"
```

---

### Task 4: Smoke-test end-to-end wiring

No new files — just verify everything connects.

**Step 1: Run the full unit test suite**

```bash
cd F:/KL/gtog
pytest tests/unit/ -v --tb=short
```

Expected: All existing tests still PASS, new tests PASS.

**Step 2: Manual smoke check (no LLM needed)**

Run this quick Python snippet to confirm the signature chain is correct end-to-end:

```python
# Run from F:/KL/gtog with: python -c "..."
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory, ConversationRole
)
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.api.query import tog_search
import inspect

# Verify search() accepts conversation_history
sig = inspect.signature(ToGSearch.search)
assert "conversation_history" in sig.parameters, "Missing in ToGSearch.search"

# Verify stream_search() accepts conversation_history
sig2 = inspect.signature(ToGSearch.stream_search)
assert "conversation_history" in sig2.parameters, "Missing in ToGSearch.stream_search"

# Verify API accepts conversation_history
sig3 = inspect.signature(tog_search)
assert "conversation_history" in sig3.parameters, "Missing in api.tog_search"

print("All signature checks passed!")
```

Expected output: `All signature checks passed!`

**Step 3: Final commit if any fixups needed**

```bash
cd F:/KL/gtog
git add -p
git commit -m "fix: conversation_history wiring corrections"
```
