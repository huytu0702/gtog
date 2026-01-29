# Phase 1: ToG SearchResult Modification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modify ToGSearch to return `SearchResult` dataclass with LLM call and token metrics, matching other search methods.

**Architecture:** Add metric tracking to LLMPruning and ToGReasoning, then aggregate in ToGSearch.search() to return SearchResult.

**Tech Stack:** Python dataclasses, async/await

---

## Task 1: Add Metric Tracking to LLMPruning

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/pruning.py:43-126`
- Test: Manual verification with debug output

**Step 1: Write the failing test**

Create a test that verifies `score_relations` returns metrics:

```python
# tests/unit/tog/test_pruning_metrics.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from graphrag.query.structured_search.tog_search.pruning import LLMPruning

@pytest.mark.asyncio
async def test_llm_pruning_returns_metrics():
    """LLMPruning.score_relations should return metrics dict."""
    mock_model = MagicMock()
    mock_model.achat_stream = AsyncMock(return_value=AsyncIteratorMock(["[8, 7, 5]"]))

    pruning = LLMPruning(model=mock_model)
    relations = [
        ("rel1 description", "target1", "OUTGOING", 1.0),
        ("rel2 description", "target2", "INCOMING", 0.5),
        ("rel3 description", "target3", "OUTGOING", 0.8),
    ]

    scored, metrics = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

    assert "llm_calls" in metrics
    assert "prompt_tokens" in metrics
    assert "output_tokens" in metrics
    assert metrics["llm_calls"] == 1

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tog/test_pruning_metrics.py -v`
Expected: FAIL with "cannot unpack non-iterable list object" or similar (current returns tuple without metrics)

**Step 3: Modify LLMPruning.score_relations to return metrics**

Edit `graphrag/query/structured_search/tog_search/pruning.py`:

```python
# Add import at top
from dataclasses import dataclass

@dataclass
class PruningMetrics:
    """Metrics from a pruning operation."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

# Modify LLMPruning class
class LLMPruning(PruningStrategy):
    """Uses LLM to score relations and entities."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.4,
        relation_scoring_prompt: str | None = None,
        entity_scoring_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature

        # Load prompts - if file path is given, read the file content
        self.relation_scoring_prompt = self._load_prompt(
            relation_scoring_prompt, TOG_RELATION_SCORING_PROMPT
        )
        self.entity_scoring_prompt = self._load_prompt(
            entity_scoring_prompt, TOG_ENTITY_SCORING_PROMPT
        )

    def _load_prompt(self, prompt_or_path: str | None, default_prompt: str) -> str:
        """Load prompt from file if path is given, otherwise use directly."""
        import os

        if prompt_or_path is None:
            return default_prompt

        # Check if it's a file path (ends with .txt or .md)
        if prompt_or_path.endswith(('.txt', '.md')):
            if os.path.exists(prompt_or_path):
                try:
                    with open(prompt_or_path, 'r', encoding='utf-8') as f:
                        logger.debug(f"Loaded prompt from file: {prompt_or_path}")
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read prompt file {prompt_or_path}: {e}")
                    return default_prompt
            else:
                logger.warning(f"Prompt file not found: {prompt_or_path}, using default")
                return default_prompt

        # Otherwise, use the string directly as prompt
        return prompt_or_path

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using LLM. Returns (scored_relations, metrics)."""
        metrics = PruningMetrics()

        if not relations:
            return [], metrics

        # Build relations text
        relations_text = "\n".join([
            f"{i + 1}. [{direction}] {rel_desc[:100]}... (weight: {weight:.2f})"
            for i, (rel_desc, _, direction, weight) in enumerate(relations)
        ])

        prompt = self.relation_scoring_prompt.format(
            query=query, entity_name=entity_name, relations=relations_text
        )

        # Count prompt tokens (approximate)
        metrics.prompt_tokens = len(prompt.split()) * 4 // 3  # Rough token estimate

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        metrics.llm_calls = 1
        metrics.output_tokens = len(response.split()) * 4 // 3  # Rough token estimate

        # Parse scores
        scores = self._parse_scores(response, len(relations))

        # Combine with relation data
        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]

        return scored_relations, metrics

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """Score entities using LLM. Returns (scores, metrics)."""
        metrics = PruningMetrics()

        if not entities:
            return [], metrics

        entities_text = "\n".join([
            f"{i + 1}. {name}: {desc[:100]}..."
            for i, (_, name, desc) in enumerate(entities)
        ])

        prompt = self.entity_scoring_prompt.format(
            query=query, current_path=current_path, candidate_entities=entities_text
        )

        metrics.prompt_tokens = len(prompt.split()) * 4 // 3

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        metrics.llm_calls = 1
        metrics.output_tokens = len(response.split()) * 4 // 3

        scores = self._parse_scores(response, len(entities))
        return scores, metrics

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse score list from LLM response."""
        import re

        # Clean response and try to extract list pattern first
        response = response.strip()

        # Try to match list pattern: [1, 2, 3] or 1, 2, 3
        list_match = re.search(r"\[([\d\s,\.]+)\]", response)
        if list_match:
            numbers_str = list_match.group(1)
        else:
            # Look for comma-separated numbers
            numbers_str = response

        # Extract numbers
        numbers = re.findall(r"\d+\.?\d*", numbers_str)
        scores = []

        for num_str in numbers[:expected_count]:
            try:
                score = float(num_str)
                # Clamp to 1-10 range
                score = max(1.0, min(10.0, score))
                scores.append(score)
            except ValueError:
                scores.append(5.0)  # Default score

        # If not enough scores, pad with uniform distribution
        while len(scores) < expected_count:
            scores.append(5.0)

        return scores[:expected_count]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tog/test_pruning_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/query/structured_search/tog_search/pruning.py tests/unit/tog/test_pruning_metrics.py
git commit -m "$(cat <<'EOF'
feat(tog): add metric tracking to LLMPruning

LLMPruning.score_relations and score_entities now return a PruningMetrics
dataclass containing llm_calls, prompt_tokens, and output_tokens counts.

)"
```

---

## Task 2: Add Metric Tracking to SemanticPruning

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/pruning.py:193-313`

**Step 1: Write the failing test**

```python
# tests/unit/tog/test_semantic_pruning_metrics.py
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from graphrag.query.structured_search.tog_search.pruning import SemanticPruning, PruningMetrics

@pytest.mark.asyncio
async def test_semantic_pruning_returns_metrics():
    """SemanticPruning.score_relations should return metrics dict."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.aembed = AsyncMock(return_value=np.array([1.0, 0.0, 0.0]))
    mock_embedding_model.aembed_batch = AsyncMock(return_value=[
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ])

    pruning = SemanticPruning(embedding_model=mock_embedding_model)
    relations = [
        ("rel1 description", "target1", "OUTGOING", 1.0),
        ("rel2 description", "target2", "INCOMING", 0.5),
    ]

    scored, metrics = await pruning.score_relations(
        query="test query",
        entity_name="TestEntity",
        relations=relations,
    )

    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 0  # SemanticPruning uses embeddings, not LLM
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tog/test_semantic_pruning_metrics.py -v`
Expected: FAIL

**Step 3: Modify SemanticPruning to return metrics**

```python
# In pruning.py, update SemanticPruning class

class SemanticPruning(PruningStrategy):
    """Uses embedding similarity for pruning."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        entity_embedding_store: Optional[BaseVectorStore] = None,
    ):
        self.embedding_model = embedding_model
        self.entity_embedding_store = entity_embedding_store
        self._entity_embeddings: Optional[np.ndarray] = None
        self._entity_texts: Optional[List[str]] = None

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using embedding similarity. Returns (scored_relations, metrics)."""
        metrics = PruningMetrics()  # No LLM calls for semantic pruning

        if not relations:
            return [], metrics

        # Embed query
        query_emb = np.array(await self.embedding_model.aembed(text=query))

        # Embed relation descriptions
        relation_texts = [rel_desc for rel_desc, _, _, _ in relations]
        relation_embeddings = await self.embedding_model.aembed_batch(
            text_list=relation_texts
        )

        # Compute cosine similarities
        scores = []
        for rel_emb in relation_embeddings:
            rel_emb = np.array(rel_emb)
            similarity = np.dot(query_emb, rel_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(rel_emb)
            )
            # Scale to 1-10 range
            score = (similarity + 1) * 5  # cosine range [-1, 1] -> [0, 10]
            scores.append(score)

        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]

        return scored_relations, metrics

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """Score entities using embedding similarity. Returns (scores, metrics)."""
        metrics = PruningMetrics()  # No LLM calls for semantic pruning

        if not entities:
            return [], metrics

        # Load entity embeddings (pre-computed or computed)
        await self._load_entity_embeddings(entities)

        # Embed query
        query_emb = np.array(await self.embedding_model.aembed(text=query))

        # Compute similarities using pre-computed embeddings
        scores = []
        for i, ent_emb in enumerate(self._entity_embeddings):
            similarity = np.dot(query_emb, ent_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(ent_emb)
            )
            score = (similarity + 1) * 5  # Scale to 1-10 range
            scores.append(score)

        return scores, metrics

    # ... rest of class unchanged
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tog/test_semantic_pruning_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/query/structured_search/tog_search/pruning.py tests/unit/tog/test_semantic_pruning_metrics.py
git commit -m "$(cat <<'EOF'
feat(tog): add metric tracking to SemanticPruning

SemanticPruning now returns PruningMetrics (with zeros since it uses
embeddings, not LLM calls) for API consistency with LLMPruning.

)"
```

---

## Task 3: Add Metric Tracking to BM25Pruning

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/pruning.py:316-418`

**Step 1: Write the failing test**

```python
# tests/unit/tog/test_bm25_pruning_metrics.py
import pytest
from graphrag.query.structured_search.tog_search.pruning import BM25Pruning, PruningMetrics

@pytest.mark.asyncio
async def test_bm25_pruning_returns_metrics():
    """BM25Pruning.score_relations should return metrics dict."""
    pruning = BM25Pruning()
    relations = [
        ("rel1 description with query words", "target1", "OUTGOING", 1.0),
        ("rel2 different content", "target2", "INCOMING", 0.5),
    ]

    scored, metrics = await pruning.score_relations(
        query="query words test",
        entity_name="TestEntity",
        relations=relations,
    )

    assert isinstance(metrics, PruningMetrics)
    assert metrics.llm_calls == 0  # BM25 uses lexical matching, not LLM
    assert len(scored) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tog/test_bm25_pruning_metrics.py -v`
Expected: FAIL

**Step 3: Modify BM25Pruning to return metrics**

```python
# In pruning.py, update BM25Pruning class

class BM25Pruning(PruningStrategy):
    """Uses BM25 algorithm for lexical matching (from ToG paper)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    # ... _tokenize and _compute_bm25_scores unchanged ...

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using BM25. Returns (scored_relations, metrics)."""
        metrics = PruningMetrics()  # No LLM calls for BM25

        if not relations:
            return [], metrics

        # Create searchable text for each relation
        relation_texts = [
            f"{entity_name} {direction} {rel_desc}"
            for rel_desc, _, direction, _ in relations
        ]

        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query, relation_texts)

        # Normalize scores to 1-10 range
        max_score = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        normalized_scores = [
            max(1.0, min(10.0, (s / max_score) * 9 + 1)) for s in bm25_scores
        ]

        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, normalized_scores
            )
        ]

        return scored_relations, metrics
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tog/test_bm25_pruning_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/query/structured_search/tog_search/pruning.py tests/unit/tog/test_bm25_pruning_metrics.py
git commit -m "$(cat <<'EOF'
feat(tog): add metric tracking to BM25Pruning

BM25Pruning now returns PruningMetrics for API consistency.

)"
```

---

## Task 4: Add Metric Tracking to ToGReasoning

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/reasoning.py`

**Step 1: Write the failing test**

```python
# tests/unit/tog/test_reasoning_metrics.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from graphrag.query.structured_search.tog_search.reasoning import ToGReasoning, ReasoningMetrics
from graphrag.query.structured_search.tog_search.state import ExplorationNode

@pytest.mark.asyncio
async def test_reasoning_returns_metrics():
    """ToGReasoning.generate_answer should return metrics."""
    mock_model = MagicMock()
    mock_model.achat_stream = AsyncMock(return_value=AsyncIteratorMock(["The answer is 42."]))

    reasoning = ToGReasoning(model=mock_model)

    node = ExplorationNode(
        entity_id="e1",
        entity_name="Entity1",
        entity_description="Description",
        depth=0,
        score=1.0,
        parent=None,
        relation_from_parent=None,
    )

    answer, paths, metrics = await reasoning.generate_answer(
        query="What is the answer?",
        exploration_paths=[node],
    )

    assert isinstance(metrics, ReasoningMetrics)
    assert metrics.llm_calls == 1
    assert answer == "The answer is 42."

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tog/test_reasoning_metrics.py -v`
Expected: FAIL with "cannot unpack" or "ReasoningMetrics not defined"

**Step 3: Modify ToGReasoning to return metrics**

Edit `graphrag/query/structured_search/tog_search/reasoning.py`:

```python
from typing import List, Tuple
from dataclasses import dataclass
from graphrag.language_model.protocol.base import ChatModel
from .state import ExplorationNode
from graphrag.prompts.query.tog_reasoning_prompt import TOG_REASONING_PROMPT


@dataclass
class ReasoningMetrics:
    """Metrics from reasoning operations."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


class ToGReasoning:
    """Handles final reasoning and answer generation."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.0,
        reasoning_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_prompt = reasoning_prompt or TOG_REASONING_PROMPT

    async def generate_answer(
        self,
        query: str,
        exploration_paths: List[ExplorationNode],
    ) -> Tuple[str, List[str], ReasoningMetrics]:
        """
        Generate final answer from exploration paths.
        Returns: (answer, reasoning_paths, metrics)
        """
        metrics = ReasoningMetrics()

        # Format exploration paths
        paths_text = self._format_paths(exploration_paths)

        # Replace placeholders in the prompt
        try:
            # If reasoning_prompt is a file path, read it
            if hasattr(
                self.reasoning_prompt, "endswith"
            ) and self.reasoning_prompt.endswith(".txt"):
                import os

                if os.path.exists(self.reasoning_prompt):
                    with open(self.reasoning_prompt, "r", encoding="utf-8") as f:
                        prompt_template = f.read()
                else:
                    prompt_template = TOG_REASONING_PROMPT
            else:
                prompt_template = self.reasoning_prompt

            prompt = prompt_template.format(query=query, exploration_paths=paths_text)
        except KeyError:
            # Fallback if prompt has different placeholders
            prompt = f"""
You are an expert at synthesizing information from knowledge graph exploration to answer questions.

Question: {query}

Exploration Paths:
{paths_text}

Your task:
1. Analyze all the exploration paths provided
2. Identify the most relevant information for answering the question
3. Synthesize this information into a comprehensive answer
4. Explain your reasoning, citing specific entities and relationships

Requirements:
- Base your answer ONLY on the provided graph exploration results
- Cite specific entities and relationships in your answer
- If the exploration paths don't contain sufficient information, acknowledge this
- Provide a clear, well-structured response

Structure your response as:
1. Direct answer to the question
2. Supporting evidence from the graph exploration
3. Key relationships that support your answer
"""

        metrics.prompt_tokens = len(prompt.split()) * 4 // 3

        answer = ""
        try:
            async for chunk in self.model.achat_stream(
                prompt=prompt,
                history=[],
                model_parameters={"temperature": self.temperature},
            ):
                answer += chunk

            metrics.llm_calls = 1
            metrics.output_tokens = len(answer.split()) * 4 // 3

        except Exception as e:
            # Fallback response if LLM call fails
            answer = f"Error generating answer: {str(e)}\n\nBased on the exploration paths, I found {len(exploration_paths)} potential paths to explore."
            metrics.llm_calls = 1  # Still count the failed attempt

        # Extract reasoning paths for transparency
        reasoning_paths = [self._path_to_string(node) for node in exploration_paths]

        return answer, reasoning_paths, metrics

    async def check_early_termination(
        self,
        query: str,
        current_nodes: List[ExplorationNode],
    ) -> Tuple[bool, str | None, ReasoningMetrics]:
        """
        Check if exploration can terminate early with an answer.
        Returns: (should_terminate, answer_or_none, metrics)
        """
        metrics = ReasoningMetrics()

        paths_text = self._format_paths(current_nodes[:3])  # Check top 3 paths

        prompt = f"""Question: {query}

Current exploration paths:
{paths_text}

Can you answer the question with high confidence based on these paths?
Respond with:
- "YES: [answer]" if you can answer confidently
- "NO: [reason]" if more exploration is needed

Response:"""

        metrics.prompt_tokens = len(prompt.split()) * 4 // 3

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": 0.0},
        ):
            response += chunk

        metrics.llm_calls = 1
        metrics.output_tokens = len(response.split()) * 4 // 3

        if response.strip().upper().startswith("YES:"):
            answer = response[4:].strip()
            return True, answer, metrics

        return False, None, metrics

    # ... _format_paths, _extract_triplets, _path_to_string unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tog/test_reasoning_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/query/structured_search/tog_search/reasoning.py tests/unit/tog/test_reasoning_metrics.py
git commit -m "$(cat <<'EOF'
feat(tog): add metric tracking to ToGReasoning

ToGReasoning.generate_answer and check_early_termination now return
ReasoningMetrics with llm_calls, prompt_tokens, and output_tokens.

)"
```

---

## Task 5: Update ToGSearch to Return SearchResult

**Files:**
- Modify: `graphrag/query/structured_search/tog_search/search.py`
- Test: `tests/unit/tog/test_search_result.py`

**Step 1: Write the failing test**

```python
# tests/unit/tog/test_search_result.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.query.structured_search.tog_search.search import ToGSearch
from graphrag.query.structured_search.base import SearchResult

@pytest.mark.asyncio
async def test_tog_search_returns_search_result():
    """ToGSearch.search should return SearchResult."""
    # This test will fail until we modify ToGSearch
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pruning = MagicMock()
    mock_reasoning = MagicMock()

    search = ToGSearch(
        model=mock_model,
        entities=[],
        relationships=[],
        tokenizer=mock_tokenizer,
        pruning_strategy=mock_pruning,
        reasoning_module=mock_reasoning,
    )

    result = await search.search(query="test query")

    assert isinstance(result, SearchResult)
    assert hasattr(result, 'llm_calls')
    assert hasattr(result, 'prompt_tokens')
    assert hasattr(result, 'output_tokens')
    assert hasattr(result, 'completion_time')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tog/test_search_result.py -v`
Expected: FAIL with assertion error (returns str, not SearchResult)

**Step 3: Modify ToGSearch to return SearchResult**

Edit `graphrag/query/structured_search/tog_search/search.py`:

```python
import time
from typing import AsyncGenerator, List, Optional, Tuple
from dataclasses import dataclass
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.tokenizer import Tokenizer
from graphrag.vector_stores.base import BaseVectorStore
from graphrag.query.structured_search.base import SearchResult
from .state import ToGSearchState, ExplorationNode
from .exploration import GraphExplorer
from .pruning import PruningStrategy, LLMPruning, SemanticPruning, PruningMetrics
from .reasoning import ToGReasoning, ReasoningMetrics


@dataclass
class ToGMetrics:
    """Aggregated metrics for ToG search."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    exploration_llm_calls: int = 0
    reasoning_llm_calls: int = 0

    def add_pruning(self, m: PruningMetrics) -> None:
        """Add pruning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.exploration_llm_calls += m.llm_calls

    def add_reasoning(self, m: ReasoningMetrics) -> None:
        """Add reasoning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.reasoning_llm_calls += m.llm_calls


class ToGSearch:
    """
    ToG (Think-on-Graph) Search Engine for GraphRAG.
    """

    def __init__(
        self,
        model: ChatModel,
        entities: List[Entity],
        relationships: List[Relationship],
        tokenizer: Tokenizer,
        pruning_strategy: PruningStrategy,
        reasoning_module: ToGReasoning,
        embedding_model: Optional[EmbeddingModel] = None,
        entity_text_embeddings: Optional[BaseVectorStore] = None,
        width: int = 3,
        depth: int = 3,
        num_retain_entity: int = 5,
        callbacks: List[QueryCallbacks] | None = None,
        debug: bool = False,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.explorer = GraphExplorer(
            entities,
            relationships,
            embedding_model=embedding_model,
            entity_embedding_store=entity_text_embeddings,
        )
        self.tokenizer = tokenizer
        self.pruning_strategy = pruning_strategy
        self.reasoning_module = reasoning_module
        self.width = width
        self.depth = depth
        self.num_retain_entity = num_retain_entity
        self.callbacks = callbacks or []
        self._debug = debug

    async def search(self, query: str) -> SearchResult:
        """Perform ToG search and return SearchResult with metrics."""
        start_time = time.time()
        metrics = ToGMetrics()

        response_chunks = []
        context_paths = []

        async for chunk, paths, chunk_metrics in self._stream_search_with_metrics(query):
            response_chunks.append(chunk)
            if paths:
                context_paths = paths
            if chunk_metrics:
                if isinstance(chunk_metrics, PruningMetrics):
                    metrics.add_pruning(chunk_metrics)
                elif isinstance(chunk_metrics, ReasoningMetrics):
                    metrics.add_reasoning(chunk_metrics)

        response = "".join(response_chunks)
        completion_time = time.time() - start_time

        # Format context data
        context_text = "\n".join(context_paths) if context_paths else ""

        return SearchResult(
            response=response,
            context_data={"exploration_paths": context_paths},
            context_text=context_text,
            completion_time=completion_time,
            llm_calls=metrics.llm_calls,
            prompt_tokens=metrics.prompt_tokens,
            output_tokens=metrics.output_tokens,
            llm_calls_categories={
                "exploration": metrics.exploration_llm_calls,
                "reasoning": metrics.reasoning_llm_calls,
            },
        )

    async def stream_search(self, query: str) -> AsyncGenerator[str, None]:
        """Perform ToG search with streaming output (backward compatible)."""
        async for chunk, _, _ in self._stream_search_with_metrics(query):
            yield chunk

    async def _stream_search_with_metrics(
        self, query: str
    ) -> AsyncGenerator[Tuple[str, List[str], PruningMetrics | ReasoningMetrics | None], None]:
        """Internal streaming search that yields (chunk, paths, metrics)."""
        # Find initial entities using semantic similarity (like ToG paper)
        if self.embedding_model:
            starting_entities = await self.explorer.find_starting_entities_semantic(
                query, top_k=self.width
            )
        else:
            starting_entities = self.explorer.find_starting_entities(
                query, top_k=self.width
            )

        if not starting_entities:
            available_entities = list(self.explorer.entities.keys())[:10]
            yield f"No relevant entities found for query '{query}'. Available entities: {available_entities}", [], None
            return

        # Initialize search state
        state = ToGSearchState(
            query=query,
            current_depth=0,
            nodes_by_depth={0: []},
            finished_paths=[],
            max_depth=self.depth,
            beam_width=self.width,
        )

        # Create initial nodes from starting entities
        for entity_id in starting_entities:
            name, description = self.explorer.get_entity_info(entity_id)
            initial_node = ExplorationNode(
                entity_id=entity_id,
                entity_name=name,
                entity_description=description,
                depth=0,
                score=1.0,
                parent=None,
                relation_from_parent=None,
            )
            state.add_node(initial_node)

        # Exploration loop
        while state.current_depth < state.max_depth:
            current_nodes = state.get_current_frontier()

            if not current_nodes:
                break

            next_depth = state.current_depth + 1
            next_level_nodes = []

            for node in current_nodes:
                relations = self.explorer.get_relations(node.entity_id)

                if not relations:
                    continue

                # Score relations and get metrics
                scored_relations, pruning_metrics = await self.pruning_strategy.score_relations(
                    query, node.entity_name, relations
                )
                yield "", [], pruning_metrics

                scored_relations.sort(key=lambda x: x[4], reverse=True)
                top_relations = scored_relations[: self.num_retain_entity]

                for rel_desc, target_id, direction, weight, score in top_relations:
                    target_name, target_desc = self.explorer.get_entity_info(target_id)
                    if target_name:
                        new_node = ExplorationNode(
                            entity_id=target_id,
                            entity_name=target_name,
                            entity_description=target_desc,
                            depth=next_depth,
                            score=score,
                            parent=node,
                            relation_from_parent=rel_desc,
                        )
                        next_level_nodes.append(new_node)

            state.nodes_by_depth[next_depth] = next_level_nodes
            state.current_depth = next_depth
            state.prune_current_frontier()

            # Check for early termination
            should_terminate, answer, term_metrics = await self.reasoning_module.check_early_termination(
                query, state.get_current_frontier()
            )
            yield "", [], term_metrics

            if should_terminate and answer:
                yield answer, [], None
                return

        # Generate final answer from explored paths
        all_paths = []
        for depth_nodes in state.nodes_by_depth.values():
            all_paths.extend(depth_nodes)

        if not all_paths:
            yield "No exploration paths were generated. The knowledge graph may not contain relevant information for this query.", [], None
            return

        try:
            answer, reasoning_paths, reason_metrics = await self.reasoning_module.generate_answer(
                query, all_paths
            )
            yield answer, reasoning_paths, reason_metrics
        except Exception as e:
            paths_summary = "\n".join([
                f"- {node.entity_name}: {node.entity_description[:100]}..."
                for node in all_paths[:5]
            ])
            yield f"""Error during reasoning: {str(e)}

However, I found these relevant entities during exploration:
{paths_summary}

Based on the exploration, I found {len(all_paths)} potential paths.""", [], None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tog/test_search_result.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/query/structured_search/tog_search/search.py tests/unit/tog/test_search_result.py
git commit -m "$(cat <<'EOF'
feat(tog): return SearchResult from ToGSearch.search

ToGSearch.search now returns SearchResult dataclass with:
- response: the answer text
- context_data: exploration paths as dict
- context_text: formatted paths as string
- completion_time: total search time in seconds
- llm_calls: total LLM API calls
- prompt_tokens: approximate input tokens
- output_tokens: approximate output tokens
- llm_calls_categories: breakdown by exploration vs reasoning

Maintains backward compatibility with stream_search returning strings.

)"
```

---

## Task 6: Update API Layer for New Return Type

**Files:**
- Modify: `graphrag/api/query.py:1232-1291`

**Step 1: Write the failing test**

```python
# tests/unit/api/test_tog_api.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

@pytest.mark.asyncio
async def test_tog_search_api_returns_search_result():
    """api.tog_search should handle SearchResult from ToGSearch."""
    # Mock the search engine to return SearchResult
    from graphrag.query.structured_search.base import SearchResult

    mock_result = SearchResult(
        response="Test answer",
        context_data={"paths": []},
        context_text="",
        completion_time=1.5,
        llm_calls=5,
        prompt_tokens=1000,
        output_tokens=200,
    )

    with patch('graphrag.api.query.get_tog_search_engine') as mock_engine:
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=mock_result)
        mock_engine.return_value = mock_search

        # ... test implementation
```

**Step 2: Modify API to handle SearchResult**

Edit `graphrag/api/query.py`, update `tog_search` function:

```python
@validate_call(config={"arbitrary_types_allowed": True})
async def tog_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame],
]:
    """Perform a ToG search and return the context data and response."""
    init_loggers(config=config, verbose=verbose, filename="query.log")

    callbacks = callbacks or []

    def on_context(context: Any) -> None:
        pass  # Context is now included in SearchResult

    local_callbacks = NoopQueryCallbacks()
    local_callbacks.on_context = on_context
    callbacks.append(local_callbacks)

    # Load entity description embedding store
    vector_store_args = {}
    for index, store in config.vector_store.items():
        vector_store_args[index] = store.model_dump()
    entity_text_embeddings = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,
    )

    entities_ = read_indexer_entities(entities, communities=None, community_level=None)
    relationships_ = read_indexer_relationships(relationships)

    logger.debug("Executing ToG search query: %s", query)
    search_engine = get_tog_search_engine(
        config=config,
        entities=entities_,
        relationships=relationships_,
        response_type="detailed",
        callbacks=callbacks,
        entity_text_embeddings=entity_text_embeddings,
    )

    # Call search which now returns SearchResult
    result = await search_engine.search(query=query)

    logger.debug("Query response: %s", truncate(result.response, 400))
    return result.response, result.context_data
```

**Step 3: Run tests**

Run: `pytest tests/unit/api/test_tog_api.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add graphrag/api/query.py tests/unit/api/test_tog_api.py
git commit -m "$(cat <<'EOF'
feat(api): update tog_search to use SearchResult

tog_search now calls search_engine.search() which returns SearchResult,
extracting response and context_data for backward compatibility.

)"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all ToG-related tests**

Run: `pytest tests/unit/tog/ -v`
Expected: All tests PASS

**Step 2: Run existing graphrag tests to verify no regressions**

Run: `pytest tests/unit/ -v --ignore=tests/unit/tog/`
Expected: All tests PASS

**Step 3: Final commit for Phase 1**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(tog): complete Phase 1 - SearchResult with metrics

ToGSearch now returns SearchResult dataclass matching other search methods:
- Tracks LLM calls and token usage during exploration and reasoning
- Provides completion_time for performance measurement
- Includes context_data with exploration paths
- Maintains backward compatibility with stream_search

Changes:
- pruning.py: PruningMetrics dataclass, updated all pruning strategies
- reasoning.py: ReasoningMetrics dataclass, updated generate_answer/check_early_termination
- search.py: ToGMetrics aggregator, SearchResult return type
- api/query.py: Updated tog_search to handle SearchResult

)"
```

---

## Phase 1 Checklist

- [ ] Task 1: LLMPruning returns PruningMetrics
- [ ] Task 2: SemanticPruning returns PruningMetrics
- [ ] Task 3: BM25Pruning returns PruningMetrics
- [ ] Task 4: ToGReasoning returns ReasoningMetrics
- [ ] Task 5: ToGSearch returns SearchResult
- [ ] Task 6: API layer updated
- [ ] Task 7: All tests pass
