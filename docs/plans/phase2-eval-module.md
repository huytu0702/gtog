# Phase 2: Evaluation Core Module

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the `graphrag/eval/` package with LLM-as-Judge metrics, evaluation runner, and result aggregation.

**Architecture:** Modular design with separate concerns - metrics.py for LLM judging, runner.py for orchestration, results.py for aggregation and output.

**Tech Stack:** Python, asyncio, pydantic, pandas

---

## Prerequisites

- Phase 1 must be complete (ToGSearch returns SearchResult with metrics)

---

## Task 1: Create Package Structure

**Files:**
- Create: `graphrag/eval/__init__.py`

**Step 1: Create the directory and __init__.py**

```python
# graphrag/eval/__init__.py
"""
GraphRAG Evaluation Framework.

Provides tools for evaluating and comparing search methods:
- LLM-as-Judge metrics (correctness, faithfulness, relevance, completeness)
- Evaluation runner for batch processing
- Result aggregation and reporting
"""

from graphrag.eval.metrics import (
    LLMJudge,
    JudgeResult,
    MetricScores,
)
from graphrag.eval.runner import (
    EvaluationRunner,
    QueryResult,
)
from graphrag.eval.results import (
    EvaluationResults,
    aggregate_results,
)

__all__ = [
    "LLMJudge",
    "JudgeResult",
    "MetricScores",
    "EvaluationRunner",
    "QueryResult",
    "EvaluationResults",
    "aggregate_results",
]
```

**Step 2: Run import test**

```python
# Test in Python REPL
python -c "from graphrag.eval import LLMJudge"
```
Expected: ImportError (modules don't exist yet)

**Step 3: Commit placeholder**

```bash
git add graphrag/eval/__init__.py
git commit -m "$(cat <<'EOF'
feat(eval): create evaluation package structure

Placeholder __init__.py for graphrag.eval module.

EOF
)"
```

---

## Task 2: Implement JudgeResult and MetricScores Dataclasses

**Files:**
- Create: `graphrag/eval/metrics.py` (partial)

**Step 1: Write the failing test**

```python
# tests/unit/eval/test_metrics_dataclasses.py
import pytest
from graphrag.eval.metrics import JudgeResult, MetricScores

def test_judge_result_creation():
    """JudgeResult should store score and reason."""
    result = JudgeResult(score=1, reason="Answer is correct")
    assert result.score == 1
    assert result.reason == "Answer is correct"

def test_metric_scores_creation():
    """MetricScores should store all four metrics."""
    scores = MetricScores(
        correctness=JudgeResult(score=1, reason="Correct"),
        faithfulness=JudgeResult(score=1, reason="Faithful"),
        relevance=JudgeResult(score=1, reason="Relevant"),
        completeness=JudgeResult(score=0, reason="Incomplete"),
    )
    assert scores.correctness.score == 1
    assert scores.completeness.score == 0

def test_metric_scores_to_dict():
    """MetricScores should convert to dict for JSON serialization."""
    scores = MetricScores(
        correctness=JudgeResult(score=1, reason="Correct"),
        faithfulness=JudgeResult(score=1, reason="Faithful"),
        relevance=JudgeResult(score=1, reason="Relevant"),
        completeness=JudgeResult(score=1, reason="Complete"),
    )
    d = scores.to_dict()
    assert d["correctness"] == 1
    assert d["faithfulness"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_metrics_dataclasses.py -v`
Expected: FAIL (module not found)

**Step 3: Create metrics.py with dataclasses**

```python
# graphrag/eval/metrics.py
"""LLM-as-Judge metrics for evaluation."""

from dataclasses import dataclass, field
from typing import Any
import json
import logging

from graphrag.language_model.protocol.base import ChatModel

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from a single LLM judge evaluation."""
    score: int  # 0 or 1
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reason": self.reason}


@dataclass
class MetricScores:
    """All metric scores for a single query-response pair."""
    correctness: JudgeResult
    faithfulness: JudgeResult
    relevance: JudgeResult
    completeness: JudgeResult

    def to_dict(self) -> dict[str, int]:
        """Return just scores for summary aggregation."""
        return {
            "correctness": self.correctness.score,
            "faithfulness": self.faithfulness.score,
            "relevance": self.relevance.score,
            "completeness": self.completeness.score,
        }

    def to_detailed_dict(self) -> dict[str, dict[str, Any]]:
        """Return full scores with reasons."""
        return {
            "correctness": self.correctness.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "relevance": self.relevance.to_dict(),
            "completeness": self.completeness.to_dict(),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_metrics_dataclasses.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/eval/metrics.py tests/unit/eval/test_metrics_dataclasses.py
git commit -m "$(cat <<'EOF'
feat(eval): add JudgeResult and MetricScores dataclasses

Core data structures for LLM-as-Judge evaluation results.

EOF
)"
```

---

## Task 3: Implement LLMJudge Class

**Files:**
- Modify: `graphrag/eval/metrics.py`

**Step 1: Write the failing test**

```python
# tests/unit/eval/test_llm_judge.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from graphrag.eval.metrics import LLMJudge, JudgeResult

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

@pytest.mark.asyncio
async def test_judge_correctness():
    """LLMJudge should evaluate correctness."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "Matches ground truth"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_correctness(
        question="What color is the sky?",
        ground_truth="Blue",
        predicted="The sky is blue.",
    )

    assert isinstance(result, JudgeResult)
    assert result.score == 1

@pytest.mark.asyncio
async def test_judge_faithfulness():
    """LLMJudge should evaluate faithfulness."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 0, "reason": "Hallucinated content"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_faithfulness(
        context="The document discusses weather patterns.",
        answer="The stock market rose 5% today.",
    )

    assert result.score == 0
    assert "Hallucinated" in result.reason

@pytest.mark.asyncio
async def test_evaluate_all_returns_metric_scores():
    """LLMJudge.evaluate_all should return MetricScores."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "Good"}'])
    )

    judge = LLMJudge(model=mock_model)
    scores = await judge.evaluate_all(
        question="Test question",
        ground_truth="Expected answer",
        predicted="Predicted answer",
        context="Retrieved context",
    )

    from graphrag.eval.metrics import MetricScores
    assert isinstance(scores, MetricScores)
    assert scores.correctness.score == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_llm_judge.py -v`
Expected: FAIL (LLMJudge not implemented)

**Step 3: Implement LLMJudge class**

Add to `graphrag/eval/metrics.py`:

```python
# Prompts as module constants
CORRECTNESS_PROMPT = """Given the question, ground truth answer, and predicted answer,
judge if the prediction is correct.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Score 1 if the predicted answer conveys the same meaning as ground truth
(exact wording not required). Score 0 if incorrect or missing key information.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

FAITHFULNESS_PROMPT = """Given the context retrieved and the answer generated,
judge if the answer is supported by the context.

Context: {context}
Answer: {answer}

Score 1 if all claims in the answer can be traced to the context.
Score 0 if the answer contains unsupported claims (hallucination).

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

RELEVANCE_PROMPT = """Given the question and the context retrieved,
judge if the context is relevant to answering the question.

Question: {question}
Context: {context}

Score 1 if the context contains information useful for answering the question.
Score 0 if the context is unrelated or unhelpful.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

COMPLETENESS_PROMPT = """Given the question and the answer generated,
judge if the answer fully addresses the question.

Question: {question}
Answer: {answer}

Score 1 if the answer addresses all aspects of the question.
Score 0 if the answer is partial or misses key aspects.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""


class LLMJudge:
    """LLM-as-Judge for evaluating search responses."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature

    async def _call_judge(self, prompt: str) -> JudgeResult:
        """Call LLM and parse JSON response."""
        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        try:
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Extract content between code blocks
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    if line.startswith("```") and in_block:
                        break
                    if in_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            result = json.loads(response)
            return JudgeResult(
                score=int(result.get("score", 0)),
                reason=result.get("reason", "No reason provided"),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse judge response: {response[:100]}... Error: {e}")
            # Try to extract score from text
            if "1" in response and "0" not in response:
                return JudgeResult(score=1, reason=f"Inferred from: {response[:100]}")
            return JudgeResult(score=0, reason=f"Parse error: {response[:100]}")

    async def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
    ) -> JudgeResult:
        """Judge if predicted answer matches ground truth."""
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
        )
        return await self._call_judge(prompt)

    async def judge_faithfulness(
        self,
        context: str,
        answer: str,
    ) -> JudgeResult:
        """Judge if answer is supported by context."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            answer=answer,
        )
        return await self._call_judge(prompt)

    async def judge_relevance(
        self,
        question: str,
        context: str,
    ) -> JudgeResult:
        """Judge if context is relevant to question."""
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            context=context,
        )
        return await self._call_judge(prompt)

    async def judge_completeness(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Judge if answer fully addresses question."""
        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            answer=answer,
        )
        return await self._call_judge(prompt)

    async def evaluate_all(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
        context: str,
    ) -> MetricScores:
        """Evaluate all metrics for a query-response pair."""
        correctness = await self.judge_correctness(question, ground_truth, predicted)
        faithfulness = await self.judge_faithfulness(context, predicted)
        relevance = await self.judge_relevance(question, context)
        completeness = await self.judge_completeness(question, predicted)

        return MetricScores(
            correctness=correctness,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_llm_judge.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/eval/metrics.py tests/unit/eval/test_llm_judge.py
git commit -m "$(cat <<'EOF'
feat(eval): implement LLMJudge for metrics evaluation

LLMJudge evaluates four metrics using LLM-as-Judge:
- correctness: does answer match ground truth?
- faithfulness: is answer supported by context?
- relevance: is context relevant to question?
- completeness: does answer fully address question?

Uses temperature=0.0 for consistent scoring.

EOF
)"
```

---

## Task 4: Implement QueryResult Dataclass

**Files:**
- Create: `graphrag/eval/runner.py` (partial)

**Step 1: Write the failing test**

```python
# tests/unit/eval/test_query_result.py
import pytest
from graphrag.eval.runner import QueryResult, EfficiencyMetrics
from graphrag.eval.metrics import MetricScores, JudgeResult

def test_query_result_creation():
    """QueryResult should store all evaluation data."""
    scores = MetricScores(
        correctness=JudgeResult(score=1, reason="Correct"),
        faithfulness=JudgeResult(score=1, reason="Faithful"),
        relevance=JudgeResult(score=1, reason="Relevant"),
        completeness=JudgeResult(score=1, reason="Complete"),
    )
    efficiency = EfficiencyMetrics(
        latency=2.5,
        llm_calls=5,
        prompt_tokens=1000,
        output_tokens=200,
    )

    result = QueryResult(
        imdb_key="tt0097576",
        question="Who is the main character?",
        method="tog",
        response="Indiana Jones is the main character.",
        scores=scores,
        efficiency=efficiency,
    )

    assert result.imdb_key == "tt0097576"
    assert result.method == "tog"
    assert result.scores.correctness.score == 1
    assert result.efficiency.latency == 2.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_query_result.py -v`
Expected: FAIL (module not found)

**Step 3: Create runner.py with dataclasses**

```python
# graphrag/eval/runner.py
"""Evaluation runner for batch processing queries."""

from dataclasses import dataclass, field
from typing import Any
import json
import logging
import asyncio
from pathlib import Path

from graphrag.eval.metrics import MetricScores, LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics from a search operation."""
    latency: float  # seconds
    llm_calls: int
    prompt_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": self.latency,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class QueryResult:
    """Result from evaluating a single query with a single method."""
    imdb_key: str
    question: str
    method: str
    response: str
    scores: MetricScores
    efficiency: EfficiencyMetrics
    ground_truth: str = ""
    context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "imdb_key": self.imdb_key,
            "question": self.question,
            "method": self.method,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "scores": self.scores.to_dict(),
            "efficiency": self.efficiency.to_dict(),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_query_result.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/eval/runner.py tests/unit/eval/test_query_result.py
git commit -m "$(cat <<'EOF'
feat(eval): add QueryResult and EfficiencyMetrics dataclasses

Data structures for storing per-query evaluation results.

EOF
)"
```

---

## Task 5: Implement EvaluationRunner

**Files:**
- Modify: `graphrag/eval/runner.py`

**Step 1: Write the failing test**

```python
# tests/unit/eval/test_evaluation_runner.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from graphrag.eval.runner import EvaluationRunner

@pytest.mark.asyncio
async def test_runner_processes_single_qa():
    """EvaluationRunner should process a single QA pair."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    # Mock judge to return scores
    from graphrag.eval.metrics import MetricScores, JudgeResult
    mock_scores = MetricScores(
        correctness=JudgeResult(score=1, reason="OK"),
        faithfulness=JudgeResult(score=1, reason="OK"),
        relevance=JudgeResult(score=1, reason="OK"),
        completeness=JudgeResult(score=1, reason="OK"),
    )
    mock_judge.evaluate_all = AsyncMock(return_value=mock_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
    )

    # Mock the search functions
    from graphrag.query.structured_search.base import SearchResult
    mock_result = SearchResult(
        response="Test answer",
        context_data={},
        context_text="Test context",
        completion_time=1.5,
        llm_calls=5,
        prompt_tokens=1000,
        output_tokens=200,
    )

    with patch.object(runner, '_run_search', return_value=mock_result):
        result = await runner.evaluate_single(
            imdb_key="tt0097576",
            question="Who is Indiana Jones?",
            ground_truth="An archaeologist",
            method="tog",
        )

    assert result.imdb_key == "tt0097576"
    assert result.method == "tog"
    assert result.scores.correctness.score == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_evaluation_runner.py -v`
Expected: FAIL

**Step 3: Implement EvaluationRunner**

Add to `graphrag/eval/runner.py`:

```python
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.structured_search.base import SearchResult
import graphrag.api as api


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    dataset_path: str
    index_roots: dict[str, str]  # imdb_key -> root path
    methods: list[str] = field(default_factory=lambda: ["tog", "local", "basic"])
    output_dir: str = "eval/results"
    save_intermediate: bool = True


class EvaluationRunner:
    """Runs evaluation across multiple QA pairs and methods."""

    def __init__(
        self,
        config: GraphRagConfig,
        judge: LLMJudge,
        index_roots: dict[str, str] | None = None,
    ):
        self.config = config
        self.judge = judge
        self.index_roots = index_roots or {}
        self._loaded_indexes: dict[str, dict] = {}

    async def _load_index(self, imdb_key: str) -> dict:
        """Load index data for a movie."""
        if imdb_key in self._loaded_indexes:
            return self._loaded_indexes[imdb_key]

        root = self.index_roots.get(imdb_key, imdb_key)
        from graphrag.config.load_config import load_config
        from graphrag.utils.api import create_storage_from_config
        from graphrag.utils.storage import load_table_from_storage

        config = load_config(Path(root))
        storage = create_storage_from_config(config.output)

        # Load required tables
        entities = await load_table_from_storage("entities", storage)
        relationships = await load_table_from_storage("relationships", storage)

        # Load optional tables for local search
        try:
            communities = await load_table_from_storage("communities", storage)
            community_reports = await load_table_from_storage("community_reports", storage)
            text_units = await load_table_from_storage("text_units", storage)
        except Exception:
            communities = None
            community_reports = None
            text_units = None

        self._loaded_indexes[imdb_key] = {
            "config": config,
            "entities": entities,
            "relationships": relationships,
            "communities": communities,
            "community_reports": community_reports,
            "text_units": text_units,
        }

        return self._loaded_indexes[imdb_key]

    async def _run_search(
        self,
        method: str,
        query: str,
        index_data: dict,
    ) -> SearchResult:
        """Run a search method and return SearchResult."""
        config = index_data["config"]
        entities = index_data["entities"]
        relationships = index_data["relationships"]

        if method == "tog":
            response, context = await api.tog_search(
                config=config,
                entities=entities,
                relationships=relationships,
                query=query,
            )
            # Wrap in SearchResult if not already
            if isinstance(response, str):
                return SearchResult(
                    response=response,
                    context_data=context,
                    context_text=str(context),
                    completion_time=0,  # Not tracked in current API
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                )
            return response

        elif method == "local":
            communities = index_data["communities"]
            community_reports = index_data["community_reports"]
            text_units = index_data["text_units"]

            if communities is None:
                raise ValueError(f"Local search requires communities table")

            response, context = await api.local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=None,
                community_level=2,
                response_type="Multiple Paragraphs",
                query=query,
            )
            return SearchResult(
                response=response,
                context_data=context,
                context_text=str(context),
                completion_time=0,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

        elif method == "basic":
            text_units = index_data["text_units"]
            if text_units is None:
                raise ValueError(f"Basic search requires text_units table")

            response, context = await api.basic_search(
                config=config,
                text_units=text_units,
                query=query,
            )
            return SearchResult(
                response=response,
                context_data=context,
                context_text=str(context),
                completion_time=0,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    async def evaluate_single(
        self,
        imdb_key: str,
        question: str,
        ground_truth: str,
        method: str,
    ) -> QueryResult:
        """Evaluate a single query with a single method."""
        import time

        # Load index
        index_data = await self._load_index(imdb_key)

        # Run search
        start_time = time.time()
        try:
            result = await self._run_search(method, question, index_data)
            latency = time.time() - start_time
        except Exception as e:
            logger.error(f"Search failed for {imdb_key}/{method}: {e}")
            # Return failure result
            from graphrag.eval.metrics import JudgeResult
            return QueryResult(
                imdb_key=imdb_key,
                question=question,
                method=method,
                response=f"ERROR: {str(e)}",
                ground_truth=ground_truth,
                context_text="",
                scores=MetricScores(
                    correctness=JudgeResult(0, "Search failed"),
                    faithfulness=JudgeResult(0, "Search failed"),
                    relevance=JudgeResult(0, "Search failed"),
                    completeness=JudgeResult(0, "Search failed"),
                ),
                efficiency=EfficiencyMetrics(
                    latency=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                ),
            )

        # Evaluate with judge
        scores = await self.judge.evaluate_all(
            question=question,
            ground_truth=ground_truth,
            predicted=result.response if isinstance(result.response, str) else str(result.response),
            context=result.context_text if isinstance(result.context_text, str) else str(result.context_text),
        )

        return QueryResult(
            imdb_key=imdb_key,
            question=question,
            method=method,
            response=result.response if isinstance(result.response, str) else str(result.response),
            ground_truth=ground_truth,
            context_text=result.context_text if isinstance(result.context_text, str) else str(result.context_text),
            scores=scores,
            efficiency=EfficiencyMetrics(
                latency=latency,
                llm_calls=result.llm_calls,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
            ),
        )

    async def run_evaluation(
        self,
        dataset: list[dict],
        methods: list[str],
        progress_callback: callable = None,
    ) -> list[QueryResult]:
        """Run evaluation on full dataset."""
        results = []
        total = len(dataset) * len(methods)
        current = 0

        for qa in dataset:
            imdb_key = qa["imdb_key"]
            question = qa["question"]
            ground_truth = qa["answer"]

            for method in methods:
                current += 1
                if progress_callback:
                    progress_callback(current, total, imdb_key, method)

                try:
                    result = await self.evaluate_single(
                        imdb_key=imdb_key,
                        question=question,
                        ground_truth=ground_truth,
                        method=method,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")

        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_evaluation_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/eval/runner.py tests/unit/eval/test_evaluation_runner.py
git commit -m "$(cat <<'EOF'
feat(eval): implement EvaluationRunner

EvaluationRunner orchestrates evaluation:
- Loads index data per movie
- Runs search methods (tog, local, basic)
- Evaluates with LLMJudge
- Returns QueryResult with scores and efficiency metrics

EOF
)"
```

---

## Task 6: Implement Results Aggregation

**Files:**
- Create: `graphrag/eval/results.py`

**Step 1: Write the failing test**

```python
# tests/unit/eval/test_results.py
import pytest
from graphrag.eval.results import EvaluationResults, aggregate_results
from graphrag.eval.runner import QueryResult, EfficiencyMetrics
from graphrag.eval.metrics import MetricScores, JudgeResult

def create_mock_result(imdb_key: str, method: str, correct: int) -> QueryResult:
    """Helper to create mock QueryResult."""
    return QueryResult(
        imdb_key=imdb_key,
        question="Test question",
        method=method,
        response="Test response",
        ground_truth="Test ground truth",
        context_text="Test context",
        scores=MetricScores(
            correctness=JudgeResult(correct, "reason"),
            faithfulness=JudgeResult(1, "reason"),
            relevance=JudgeResult(1, "reason"),
            completeness=JudgeResult(1, "reason"),
        ),
        efficiency=EfficiencyMetrics(
            latency=1.5,
            llm_calls=5,
            prompt_tokens=1000,
            output_tokens=200,
        ),
    )

def test_aggregate_by_method():
    """aggregate_results should compute averages by method."""
    results = [
        create_mock_result("tt1", "tog", 1),
        create_mock_result("tt1", "tog", 0),
        create_mock_result("tt1", "local", 1),
        create_mock_result("tt1", "local", 1),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.by_method["tog"]["correctness"] == 0.5
    assert aggregated.by_method["local"]["correctness"] == 1.0

def test_aggregate_by_movie():
    """aggregate_results should compute averages by movie."""
    results = [
        create_mock_result("tt1", "tog", 1),
        create_mock_result("tt2", "tog", 0),
    ]

    aggregated = aggregate_results(results)

    assert "tt1" in aggregated.by_movie
    assert "tt2" in aggregated.by_movie

def test_to_json():
    """EvaluationResults should serialize to JSON."""
    results = [create_mock_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    json_str = aggregated.to_json()
    import json
    parsed = json.loads(json_str)

    assert "by_method" in parsed
    assert "by_movie" in parsed
    assert "efficiency" in parsed
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_results.py -v`
Expected: FAIL

**Step 3: Implement results.py**

```python
# graphrag/eval/results.py
"""Result aggregation and output formatting."""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import json

from graphrag.eval.runner import QueryResult


@dataclass
class MethodSummary:
    """Summary statistics for a single method."""
    correctness: float
    faithfulness: float
    relevance: float
    completeness: float
    avg_latency: float
    avg_llm_calls: float
    avg_prompt_tokens: float
    avg_output_tokens: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": round(self.correctness, 3),
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "completeness": round(self.completeness, 3),
            "avg_latency": round(self.avg_latency, 3),
            "avg_llm_calls": round(self.avg_llm_calls, 1),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 0),
            "avg_output_tokens": round(self.avg_output_tokens, 0),
            "count": self.count,
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    metadata: dict[str, Any]
    results: list[QueryResult]
    by_method: dict[str, dict[str, float]]
    by_movie: dict[str, dict[str, dict[str, float]]]
    efficiency: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "by_method": self.by_method,
            "by_movie": self.by_movie,
            "efficiency": self.efficiency,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_detailed_dict(self) -> dict[str, Any]:
        """Full results including per-query details."""
        return {
            "metadata": self.metadata,
            "results": [
                {
                    "imdb_key": r.imdb_key,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "methods": {
                        r.method: {
                            "response": r.response,
                            "scores": r.scores.to_dict(),
                            "efficiency": r.efficiency.to_dict(),
                        }
                    },
                }
                for r in self.results
            ],
        }

    def to_detailed_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_detailed_dict(), indent=indent)

    def save(self, output_dir: str) -> None:
        """Save results to files."""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path / "eval_results_summary.json"
        with open(summary_path, "w") as f:
            f.write(self.to_json())

        # Save detailed results
        detailed_path = output_path / "eval_results_detailed.json"
        with open(detailed_path, "w") as f:
            f.write(self.to_detailed_json())


def aggregate_results(results: list[QueryResult]) -> EvaluationResults:
    """Aggregate QueryResults into summary statistics."""
    if not results:
        return EvaluationResults(
            metadata={"timestamp": datetime.now().isoformat(), "total_questions": 0},
            results=[],
            by_method={},
            by_movie={},
            efficiency={},
        )

    # Collect unique methods and movies
    methods = set(r.method for r in results)
    movies = set(r.imdb_key for r in results)

    # Aggregate by method
    by_method = {}
    for method in methods:
        method_results = [r for r in results if r.method == method]
        if method_results:
            by_method[method] = {
                "correctness": sum(r.scores.correctness.score for r in method_results) / len(method_results),
                "faithfulness": sum(r.scores.faithfulness.score for r in method_results) / len(method_results),
                "relevance": sum(r.scores.relevance.score for r in method_results) / len(method_results),
                "completeness": sum(r.scores.completeness.score for r in method_results) / len(method_results),
            }

    # Aggregate by movie
    by_movie = {}
    for movie in movies:
        by_movie[movie] = {}
        for method in methods:
            movie_method_results = [r for r in results if r.imdb_key == movie and r.method == method]
            if movie_method_results:
                by_movie[movie][method] = {
                    "correctness": sum(r.scores.correctness.score for r in movie_method_results) / len(movie_method_results),
                    "faithfulness": sum(r.scores.faithfulness.score for r in movie_method_results) / len(movie_method_results),
                    "relevance": sum(r.scores.relevance.score for r in movie_method_results) / len(movie_method_results),
                    "completeness": sum(r.scores.completeness.score for r in movie_method_results) / len(movie_method_results),
                }

    # Aggregate efficiency by method
    efficiency = {}
    for method in methods:
        method_results = [r for r in results if r.method == method]
        if method_results:
            efficiency[method] = {
                "avg_latency": sum(r.efficiency.latency for r in method_results) / len(method_results),
                "avg_llm_calls": sum(r.efficiency.llm_calls for r in method_results) / len(method_results),
                "avg_prompt_tokens": sum(r.efficiency.prompt_tokens for r in method_results) / len(method_results),
                "avg_output_tokens": sum(r.efficiency.output_tokens for r in method_results) / len(method_results),
            }

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(set((r.imdb_key, r.question) for r in results)),
        "methods": list(methods),
        "movies": list(movies),
    }

    return EvaluationResults(
        metadata=metadata,
        results=results,
        by_method=by_method,
        by_movie=by_movie,
        efficiency=efficiency,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_results.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add graphrag/eval/results.py tests/unit/eval/test_results.py
git commit -m "$(cat <<'EOF'
feat(eval): implement result aggregation

EvaluationResults aggregates QueryResults into:
- by_method: average scores per search method
- by_movie: average scores per movie per method
- efficiency: average latency/tokens per method

Provides JSON serialization and file saving.

EOF
)"
```

---

## Task 7: Update Package Exports

**Files:**
- Modify: `graphrag/eval/__init__.py`

**Step 1: Update __init__.py with all exports**

```python
# graphrag/eval/__init__.py
"""
GraphRAG Evaluation Framework.

Provides tools for evaluating and comparing search methods:
- LLM-as-Judge metrics (correctness, faithfulness, relevance, completeness)
- Evaluation runner for batch processing
- Result aggregation and reporting
"""

from graphrag.eval.metrics import (
    LLMJudge,
    JudgeResult,
    MetricScores,
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    COMPLETENESS_PROMPT,
)
from graphrag.eval.runner import (
    EvaluationRunner,
    EvaluationConfig,
    QueryResult,
    EfficiencyMetrics,
)
from graphrag.eval.results import (
    EvaluationResults,
    MethodSummary,
    aggregate_results,
)

__all__ = [
    # Metrics
    "LLMJudge",
    "JudgeResult",
    "MetricScores",
    "CORRECTNESS_PROMPT",
    "FAITHFULNESS_PROMPT",
    "RELEVANCE_PROMPT",
    "COMPLETENESS_PROMPT",
    # Runner
    "EvaluationRunner",
    "EvaluationConfig",
    "QueryResult",
    "EfficiencyMetrics",
    # Results
    "EvaluationResults",
    "MethodSummary",
    "aggregate_results",
]
```

**Step 2: Test imports**

```bash
python -c "from graphrag.eval import LLMJudge, EvaluationRunner, aggregate_results; print('OK')"
```
Expected: "OK"

**Step 3: Commit**

```bash
git add graphrag/eval/__init__.py
git commit -m "$(cat <<'EOF'
feat(eval): export all evaluation components

Updated __init__.py to export all public APIs.

EOF
)"
```

---

## Task 8: Run All Phase 2 Tests

**Step 1: Run all eval tests**

Run: `pytest tests/unit/eval/ -v`
Expected: All tests PASS

**Step 2: Final commit for Phase 2**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(eval): complete Phase 2 - evaluation core module

Complete graphrag/eval package:
- metrics.py: LLMJudge with four evaluation metrics
- runner.py: EvaluationRunner for batch processing
- results.py: Result aggregation and JSON output

EOF
)"
```

---

## Phase 2 Checklist

- [ ] Task 1: Package structure created
- [ ] Task 2: JudgeResult and MetricScores dataclasses
- [ ] Task 3: LLMJudge implementation
- [ ] Task 4: QueryResult and EfficiencyMetrics dataclasses
- [ ] Task 5: EvaluationRunner implementation
- [ ] Task 6: Results aggregation
- [ ] Task 7: Package exports updated
- [ ] Task 8: All tests pass
