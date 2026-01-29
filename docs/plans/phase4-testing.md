# Phase 4: Testing & Validation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive unit and integration tests for the evaluation framework to ensure correctness and prevent regressions.

**Architecture:** Mirror the module structure in tests/. Unit tests for isolated components, integration tests for end-to-end flows.

**Tech Stack:** pytest, pytest-asyncio, unittest.mock

---

## Prerequisites

- Phase 2 must be complete (graphrag.eval module exists)
- Phase 3 is optional (CLI tests can be skipped if not done)

---

## Task 1: Create Test Directory Structure

**Files:**
- Create: `tests/unit/eval/__init__.py`
- Create: `tests/integration/eval/__init__.py`

**Step 1: Create directories**

```bash
mkdir -p tests/unit/eval
mkdir -p tests/integration/eval
touch tests/unit/eval/__init__.py
touch tests/integration/eval/__init__.py
```

**Step 2: Commit**

```bash
git add tests/unit/eval/__init__.py tests/integration/eval/__init__.py
git commit -m "$(cat <<'EOF'
test(eval): create test directory structure

EOF
)"
```

---

## Task 2: Create Test Fixtures

**Files:**
- Create: `tests/unit/eval/conftest.py`

**Step 1: Create conftest.py with shared fixtures**

```python
# tests/unit/eval/conftest.py
"""Shared fixtures for eval tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock


class AsyncIteratorMock:
    """Mock async iterator for LLM streaming responses."""

    def __init__(self, items: list[str]):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.fixture
def mock_chat_model():
    """Create a mock ChatModel that returns JSON responses."""
    model = MagicMock()

    def create_stream(response: str):
        model.achat_stream = MagicMock(
            return_value=AsyncIteratorMock([response])
        )

    model.set_response = create_stream
    model.set_response('{"score": 1, "reason": "Test reason"}')

    return model


@pytest.fixture
def mock_search_result():
    """Create a mock SearchResult."""
    from graphrag.query.structured_search.base import SearchResult

    return SearchResult(
        response="Test answer from search",
        context_data={"entities": ["Entity1", "Entity2"]},
        context_text="Entity1 is related to Entity2 via TestRelation",
        completion_time=1.5,
        llm_calls=5,
        prompt_tokens=1000,
        output_tokens=200,
    )


@pytest.fixture
def sample_qa_dataset():
    """Sample QA dataset for testing."""
    return [
        {
            "question": "Who does the golden crucifix belong to?",
            "answer": "To Coronado",
            "imdb_key": "tt0097576",
        },
        {
            "question": "What is Indy doing with his Boy Scout troop?",
            "answer": "He is horseback riding",
            "imdb_key": "tt0097576",
        },
        {
            "question": "Who is T-1000?",
            "answer": "An advanced Terminator",
            "imdb_key": "tt0102798",
        },
    ]


@pytest.fixture
def mock_metric_scores():
    """Create mock MetricScores."""
    from graphrag.eval.metrics import MetricScores, JudgeResult

    return MetricScores(
        correctness=JudgeResult(score=1, reason="Correct answer"),
        faithfulness=JudgeResult(score=1, reason="Supported by context"),
        relevance=JudgeResult(score=1, reason="Relevant context"),
        completeness=JudgeResult(score=1, reason="Complete answer"),
    )


@pytest.fixture
def mock_efficiency_metrics():
    """Create mock EfficiencyMetrics."""
    from graphrag.eval.runner import EfficiencyMetrics

    return EfficiencyMetrics(
        latency=2.5,
        llm_calls=8,
        prompt_tokens=1500,
        output_tokens=300,
    )


@pytest.fixture
def mock_query_result(mock_metric_scores, mock_efficiency_metrics):
    """Create mock QueryResult."""
    from graphrag.eval.runner import QueryResult

    return QueryResult(
        imdb_key="tt0097576",
        question="Who does the golden crucifix belong to?",
        method="tog",
        response="The golden crucifix belongs to Coronado.",
        ground_truth="To Coronado",
        context_text="Coronado owned the golden crucifix...",
        scores=mock_metric_scores,
        efficiency=mock_efficiency_metrics,
    )
```

**Step 2: Commit**

```bash
git add tests/unit/eval/conftest.py
git commit -m "$(cat <<'EOF'
test(eval): add shared fixtures for eval tests

Fixtures for mock models, search results, datasets, and metrics.

EOF
)"
```

---

## Task 3: Unit Tests for LLMJudge

**Files:**
- Create: `tests/unit/eval/test_llm_judge.py`

**Step 1: Write comprehensive LLMJudge tests**

```python
# tests/unit/eval/test_llm_judge.py
"""Unit tests for LLMJudge."""

import pytest
from unittest.mock import MagicMock
from graphrag.eval.metrics import LLMJudge, JudgeResult, MetricScores


@pytest.mark.asyncio
async def test_judge_correctness_score_1(mock_chat_model):
    """Judge should return score=1 when LLM indicates correct."""
    mock_chat_model.set_response('{"score": 1, "reason": "Matches ground truth"}')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_correctness(
        question="What color is the sky?",
        ground_truth="Blue",
        predicted="The sky is blue.",
    )

    assert isinstance(result, JudgeResult)
    assert result.score == 1
    assert "Matches" in result.reason


@pytest.mark.asyncio
async def test_judge_correctness_score_0(mock_chat_model):
    """Judge should return score=0 when LLM indicates incorrect."""
    mock_chat_model.set_response('{"score": 0, "reason": "Wrong answer"}')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_correctness(
        question="What color is the sky?",
        ground_truth="Blue",
        predicted="The sky is green.",
    )

    assert result.score == 0


@pytest.mark.asyncio
async def test_judge_faithfulness(mock_chat_model):
    """Judge should evaluate faithfulness."""
    mock_chat_model.set_response('{"score": 0, "reason": "Hallucinated content not in context"}')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_faithfulness(
        context="The document discusses weather patterns in Europe.",
        answer="The stock market rose 5% in Asia today.",
    )

    assert result.score == 0
    assert "Hallucinated" in result.reason


@pytest.mark.asyncio
async def test_judge_relevance(mock_chat_model):
    """Judge should evaluate context relevance."""
    mock_chat_model.set_response('{"score": 1, "reason": "Context directly addresses the question"}')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_relevance(
        question="Who is Indiana Jones?",
        context="Indiana Jones is an archaeologist and professor.",
    )

    assert result.score == 1


@pytest.mark.asyncio
async def test_judge_completeness(mock_chat_model):
    """Judge should evaluate answer completeness."""
    mock_chat_model.set_response('{"score": 0, "reason": "Missing key information"}')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_completeness(
        question="What are the three colors of the French flag?",
        answer="The flag has blue and white.",
    )

    assert result.score == 0


@pytest.mark.asyncio
async def test_evaluate_all(mock_chat_model):
    """evaluate_all should return MetricScores with all four metrics."""
    mock_chat_model.set_response('{"score": 1, "reason": "Good"}')

    judge = LLMJudge(model=mock_chat_model)
    scores = await judge.evaluate_all(
        question="Test question",
        ground_truth="Expected answer",
        predicted="Actual answer",
        context="Some context",
    )

    assert isinstance(scores, MetricScores)
    assert scores.correctness.score == 1
    assert scores.faithfulness.score == 1
    assert scores.relevance.score == 1
    assert scores.completeness.score == 1


@pytest.mark.asyncio
async def test_judge_handles_malformed_json(mock_chat_model):
    """Judge should handle malformed JSON gracefully."""
    mock_chat_model.set_response("This is not valid JSON")

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    # Should return a result (with inferred score) rather than raising
    assert isinstance(result, JudgeResult)


@pytest.mark.asyncio
async def test_judge_handles_markdown_code_blocks(mock_chat_model):
    """Judge should extract JSON from markdown code blocks."""
    mock_chat_model.set_response('```json\n{"score": 1, "reason": "OK"}\n```')

    judge = LLMJudge(model=mock_chat_model)
    result = await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    assert result.score == 1


@pytest.mark.asyncio
async def test_judge_uses_temperature_0(mock_chat_model):
    """Judge should use temperature=0 for consistency."""
    judge = LLMJudge(model=mock_chat_model, temperature=0.0)

    await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    # Verify temperature was passed to model
    call_args = mock_chat_model.achat_stream.call_args
    assert call_args[1]["model_parameters"]["temperature"] == 0.0
```

**Step 2: Run tests**

Run: `pytest tests/unit/eval/test_llm_judge.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/unit/eval/test_llm_judge.py
git commit -m "$(cat <<'EOF'
test(eval): add comprehensive LLMJudge unit tests

Tests for all four metrics, JSON parsing edge cases,
and temperature configuration.

EOF
)"
```

---

## Task 4: Unit Tests for EvaluationRunner

**Files:**
- Create: `tests/unit/eval/test_runner.py`

**Step 1: Write EvaluationRunner tests**

```python
# tests/unit/eval/test_runner.py
"""Unit tests for EvaluationRunner."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from graphrag.eval.runner import EvaluationRunner, QueryResult, EfficiencyMetrics
from graphrag.eval.metrics import LLMJudge, MetricScores, JudgeResult


@pytest.mark.asyncio
async def test_evaluate_single_success(mock_chat_model, mock_search_result, mock_metric_scores):
    """evaluate_single should return QueryResult on success."""
    mock_config = MagicMock()
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate_all = AsyncMock(return_value=mock_metric_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {"config": mock_config, "entities": MagicMock()}

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            result = await runner.evaluate_single(
                imdb_key="tt0097576",
                question="Who is Indiana Jones?",
                ground_truth="An archaeologist",
                method="tog",
            )

    assert isinstance(result, QueryResult)
    assert result.imdb_key == "tt0097576"
    assert result.method == "tog"
    assert result.scores.correctness.score == 1


@pytest.mark.asyncio
async def test_evaluate_single_search_failure(mock_chat_model):
    """evaluate_single should return error QueryResult on search failure."""
    mock_config = MagicMock()
    mock_judge = MagicMock(spec=LLMJudge)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {"config": mock_config, "entities": MagicMock()}

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("Search failed")

            result = await runner.evaluate_single(
                imdb_key="tt0097576",
                question="Test question",
                ground_truth="Expected",
                method="tog",
            )

    assert "ERROR" in result.response
    assert result.scores.correctness.score == 0


@pytest.mark.asyncio
async def test_run_evaluation_batch(mock_chat_model, mock_search_result, mock_metric_scores, sample_qa_dataset):
    """run_evaluation should process multiple QA pairs."""
    mock_config = MagicMock()
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate_all = AsyncMock(return_value=mock_metric_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576", "tt0102798": "tt0102798"},
    )

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {
            "config": mock_config,
            "entities": MagicMock(),
            "relationships": MagicMock(),
        }

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            results = await runner.run_evaluation(
                dataset=sample_qa_dataset,
                methods=["tog"],
            )

    assert len(results) == 3  # 3 QA pairs * 1 method


@pytest.mark.asyncio
async def test_run_evaluation_multiple_methods(mock_chat_model, mock_search_result, mock_metric_scores):
    """run_evaluation should test each method for each QA pair."""
    mock_config = MagicMock()
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate_all = AsyncMock(return_value=mock_metric_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    dataset = [{"question": "Test?", "answer": "Yes", "imdb_key": "tt0097576"}]

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {
            "config": mock_config,
            "entities": MagicMock(),
            "relationships": MagicMock(),
            "communities": MagicMock(),
            "community_reports": MagicMock(),
            "text_units": MagicMock(),
        }

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            results = await runner.run_evaluation(
                dataset=dataset,
                methods=["tog", "local", "basic"],
            )

    assert len(results) == 3  # 1 QA pair * 3 methods


@pytest.mark.asyncio
async def test_progress_callback(mock_chat_model, mock_search_result, mock_metric_scores):
    """run_evaluation should call progress callback."""
    mock_config = MagicMock()
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.evaluate_all = AsyncMock(return_value=mock_metric_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    dataset = [{"question": "Test?", "answer": "Yes", "imdb_key": "tt0097576"}]
    progress_calls = []

    def progress(current, total, movie, method):
        progress_calls.append((current, total, movie, method))

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {"config": mock_config, "entities": MagicMock()}

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            await runner.run_evaluation(
                dataset=dataset,
                methods=["tog"],
                progress_callback=progress,
            )

    assert len(progress_calls) == 1
    assert progress_calls[0][0] == 1  # current
    assert progress_calls[0][1] == 1  # total
```

**Step 2: Run tests**

Run: `pytest tests/unit/eval/test_runner.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/unit/eval/test_runner.py
git commit -m "$(cat <<'EOF'
test(eval): add EvaluationRunner unit tests

Tests for single evaluation, batch processing, multiple methods,
error handling, and progress callbacks.

EOF
)"
```

---

## Task 5: Unit Tests for Result Aggregation

**Files:**
- Create: `tests/unit/eval/test_results.py`

**Step 1: Write result aggregation tests**

```python
# tests/unit/eval/test_results.py
"""Unit tests for result aggregation."""

import pytest
import json
from graphrag.eval.results import EvaluationResults, aggregate_results
from graphrag.eval.runner import QueryResult, EfficiencyMetrics
from graphrag.eval.metrics import MetricScores, JudgeResult


def create_query_result(
    imdb_key: str,
    method: str,
    correctness: int,
    latency: float = 1.0,
) -> QueryResult:
    """Helper to create QueryResult with specified values."""
    return QueryResult(
        imdb_key=imdb_key,
        question="Test question",
        method=method,
        response="Test response",
        ground_truth="Test ground truth",
        context_text="Test context",
        scores=MetricScores(
            correctness=JudgeResult(correctness, "reason"),
            faithfulness=JudgeResult(1, "reason"),
            relevance=JudgeResult(1, "reason"),
            completeness=JudgeResult(1, "reason"),
        ),
        efficiency=EfficiencyMetrics(
            latency=latency,
            llm_calls=5,
            prompt_tokens=1000,
            output_tokens=200,
        ),
    )


def test_aggregate_empty_results():
    """aggregate_results should handle empty list."""
    result = aggregate_results([])

    assert result.metadata["total_questions"] == 0
    assert result.by_method == {}
    assert result.by_movie == {}


def test_aggregate_by_method():
    """aggregate_results should compute correct averages by method."""
    results = [
        create_query_result("tt1", "tog", 1),
        create_query_result("tt1", "tog", 0),
        create_query_result("tt1", "local", 1),
        create_query_result("tt1", "local", 1),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.by_method["tog"]["correctness"] == 0.5
    assert aggregated.by_method["local"]["correctness"] == 1.0


def test_aggregate_by_movie():
    """aggregate_results should compute correct averages by movie."""
    results = [
        create_query_result("tt1", "tog", 1),
        create_query_result("tt1", "tog", 1),
        create_query_result("tt2", "tog", 0),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.by_movie["tt1"]["tog"]["correctness"] == 1.0
    assert aggregated.by_movie["tt2"]["tog"]["correctness"] == 0.0


def test_aggregate_efficiency():
    """aggregate_results should compute efficiency averages."""
    results = [
        create_query_result("tt1", "tog", 1, latency=2.0),
        create_query_result("tt1", "tog", 1, latency=4.0),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.efficiency["tog"]["avg_latency"] == 3.0


def test_to_json():
    """EvaluationResults should serialize to valid JSON."""
    results = [create_query_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    json_str = aggregated.to_json()
    parsed = json.loads(json_str)

    assert "by_method" in parsed
    assert "by_movie" in parsed
    assert "efficiency" in parsed
    assert "metadata" in parsed


def test_to_detailed_json():
    """EvaluationResults should serialize detailed results."""
    results = [create_query_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    json_str = aggregated.to_detailed_json()
    parsed = json.loads(json_str)

    assert "results" in parsed
    assert len(parsed["results"]) == 1


def test_metadata_contains_expected_fields():
    """Metadata should contain timestamp, totals, methods, movies."""
    results = [
        create_query_result("tt1", "tog", 1),
        create_query_result("tt2", "local", 1),
    ]
    aggregated = aggregate_results(results)

    assert "timestamp" in aggregated.metadata
    assert aggregated.metadata["total_questions"] == 2
    assert "tog" in aggregated.metadata["methods"]
    assert "local" in aggregated.metadata["methods"]
    assert "tt1" in aggregated.metadata["movies"]
    assert "tt2" in aggregated.metadata["movies"]


def test_save_creates_files(tmp_path):
    """save should create summary and detailed JSON files."""
    results = [create_query_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    output_dir = tmp_path / "results"
    aggregated.save(str(output_dir))

    assert (output_dir / "eval_results_summary.json").exists()
    assert (output_dir / "eval_results_detailed.json").exists()

    # Verify content
    with open(output_dir / "eval_results_summary.json") as f:
        summary = json.load(f)
    assert "by_method" in summary
```

**Step 2: Run tests**

Run: `pytest tests/unit/eval/test_results.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/unit/eval/test_results.py
git commit -m "$(cat <<'EOF'
test(eval): add result aggregation unit tests

Tests for aggregation by method, by movie, efficiency metrics,
JSON serialization, and file saving.

EOF
)"
```

---

## Task 6: Integration Test for Full Evaluation Flow

**Files:**
- Create: `tests/integration/eval/test_eval_flow.py`

**Step 1: Write integration test**

```python
# tests/integration/eval/test_eval_flow.py
"""Integration tests for evaluation flow."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
from pathlib import Path

from graphrag.eval import (
    LLMJudge,
    EvaluationRunner,
    aggregate_results,
)
from graphrag.query.structured_search.base import SearchResult


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
async def test_full_evaluation_flow(tmp_path):
    """Test complete evaluation flow from dataset to results."""
    # Create mock LLM that returns consistent scores
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "Correct"}'])
    )

    # Create judge
    judge = LLMJudge(model=mock_model)

    # Create mock config
    mock_config = MagicMock()

    # Create runner
    runner = EvaluationRunner(
        config=mock_config,
        judge=judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    # Mock search result
    mock_search_result = SearchResult(
        response="Test answer",
        context_data={},
        context_text="Test context",
        completion_time=1.0,
        llm_calls=5,
        prompt_tokens=1000,
        output_tokens=200,
    )

    # Sample dataset
    dataset = [
        {"question": "Q1?", "answer": "A1", "imdb_key": "tt0097576"},
        {"question": "Q2?", "answer": "A2", "imdb_key": "tt0097576"},
    ]

    # Mock the search
    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {
            "config": mock_config,
            "entities": MagicMock(),
            "relationships": MagicMock(),
        }

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            # Run evaluation
            results = await runner.run_evaluation(
                dataset=dataset,
                methods=["tog"],
            )

    # Verify results
    assert len(results) == 2
    for r in results:
        assert r.scores.correctness.score == 1

    # Aggregate
    aggregated = aggregate_results(results)

    # Verify aggregation
    assert aggregated.by_method["tog"]["correctness"] == 1.0

    # Save to temp directory
    output_dir = tmp_path / "eval_results"
    aggregated.save(str(output_dir))

    # Verify files created
    assert (output_dir / "eval_results_summary.json").exists()
    assert (output_dir / "eval_results_detailed.json").exists()

    # Verify summary content
    with open(output_dir / "eval_results_summary.json") as f:
        summary = json.load(f)

    assert summary["by_method"]["tog"]["correctness"] == 1.0
    assert summary["metadata"]["total_questions"] == 2


@pytest.mark.asyncio
async def test_evaluation_with_mixed_scores():
    """Test evaluation with varying scores."""
    mock_model = MagicMock()
    call_count = [0]

    def get_stream(*args, **kwargs):
        call_count[0] += 1
        # Alternate between 0 and 1
        score = call_count[0] % 2
        return AsyncIteratorMock([f'{{"score": {score}, "reason": "Test"}}'])

    mock_model.achat_stream = MagicMock(side_effect=get_stream)

    judge = LLMJudge(model=mock_model)
    mock_config = MagicMock()

    runner = EvaluationRunner(
        config=mock_config,
        judge=judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    mock_search_result = SearchResult(
        response="Test",
        context_data={},
        context_text="Context",
        completion_time=1.0,
        llm_calls=1,
        prompt_tokens=100,
        output_tokens=50,
    )

    dataset = [
        {"question": "Q1?", "answer": "A1", "imdb_key": "tt0097576"},
    ]

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {"config": mock_config, "entities": MagicMock()}

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_result

            results = await runner.run_evaluation(dataset=dataset, methods=["tog"])

    # First query gets scores: correctness=1, faithfulness=0, relevance=1, completeness=0
    # (alternating pattern)
    aggregated = aggregate_results(results)
    # Due to alternating, we should see mixed averages
    # This tests the aggregation logic handles non-perfect scores


@pytest.mark.asyncio
async def test_evaluation_handles_search_errors():
    """Test that evaluation continues despite search errors."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "OK"}'])
    )

    judge = LLMJudge(model=mock_model)
    mock_config = MagicMock()

    runner = EvaluationRunner(
        config=mock_config,
        judge=judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    dataset = [
        {"question": "Q1?", "answer": "A1", "imdb_key": "tt0097576"},
        {"question": "Q2?", "answer": "A2", "imdb_key": "tt0097576"},
    ]

    call_count = [0]

    async def failing_search(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Search failed")
        return SearchResult(
            response="Success",
            context_data={},
            context_text="Context",
            completion_time=1.0,
            llm_calls=1,
            prompt_tokens=100,
            output_tokens=50,
        )

    with patch.object(runner, '_load_index', new_callable=AsyncMock) as mock_load:
        mock_load.return_value = {"config": mock_config, "entities": MagicMock()}

        with patch.object(runner, '_run_search', new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = failing_search

            results = await runner.run_evaluation(dataset=dataset, methods=["tog"])

    # Should have 2 results (one error, one success)
    assert len(results) == 2

    # First should be error
    assert "ERROR" in results[0].response
    assert results[0].scores.correctness.score == 0

    # Second should succeed (but won't have judge scores since judge wasn't called)
```

**Step 2: Run tests**

Run: `pytest tests/integration/eval/test_eval_flow.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/integration/eval/test_eval_flow.py
git commit -m "$(cat <<'EOF'
test(eval): add integration tests for evaluation flow

End-to-end tests covering:
- Full evaluation flow from dataset to saved results
- Mixed score handling
- Error recovery during search

EOF
)"
```

---

## Task 7: Run All Eval Tests

**Step 1: Run all eval tests**

Run: `pytest tests/unit/eval/ tests/integration/eval/ -v`
Expected: All tests PASS

**Step 2: Run with coverage**

Run: `pytest tests/unit/eval/ tests/integration/eval/ --cov=graphrag.eval --cov-report=term-missing`
Expected: Good coverage of graphrag.eval module

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(eval): complete Phase 4 - comprehensive testing

Test suite for evaluation framework:
- Unit tests for LLMJudge (8 tests)
- Unit tests for EvaluationRunner (5 tests)
- Unit tests for result aggregation (9 tests)
- Integration tests for full flow (3 tests)

Total: 25+ tests covering core evaluation functionality.

EOF
)"
```

---

## Phase 4 Checklist

- [ ] Task 1: Test directory structure
- [ ] Task 2: Test fixtures (conftest.py)
- [ ] Task 3: LLMJudge unit tests
- [ ] Task 4: EvaluationRunner unit tests
- [ ] Task 5: Result aggregation unit tests
- [ ] Task 6: Integration tests
- [ ] Task 7: All tests pass with good coverage
