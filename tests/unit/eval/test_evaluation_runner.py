"""
Tests for EvaluationRunner class.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.eval.runner import EvaluationRunner, QueryResult


class AsyncIteratorMock:
    """Mock async iterator for LLM streaming responses."""
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
async def test_runner_processes_single_qa():
    """EvaluationRunner should process a single QA pair."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    # Mock judge to return scores
    from graphrag.eval.metrics import MetricScores, JudgeResult
    mock_scores = MetricScores(
        correctness=JudgeResult(1, "OK"),
        faithfulness=JudgeResult(1, "OK"),
        relevance=JudgeResult(1, "OK"),
        completeness=JudgeResult(1, "OK"),
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

    # Mock both _load_index and _run_search
    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }
    with patch.object(runner, '_load_index', return_value=mock_index_data), \
         patch.object(runner, '_run_search', return_value=mock_result):
        result = await runner.evaluate_single(
            imdb_key="tt0097576",
            question="Who is Indiana Jones?",
            ground_truth="An archaeologist",
            method="tog",
        )

    assert result.imdb_key == "tt0097576"
    assert result.method == "tog"
    assert result.scores.correctness.score == 1


@pytest.mark.asyncio
async def test_evaluate_single_search_failure():
    """evaluate_single should return error QueryResult on search failure."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
    )

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

    with patch.object(runner, '_load_index', return_value=mock_index_data), \
         patch.object(runner, '_run_search', side_effect=Exception("Search failed")):
        result = await runner.evaluate_single(
            imdb_key="tt0097576",
            question="Test question",
            ground_truth="Expected",
            method="tog",
        )

    assert "ERROR" in result.response
    assert result.scores.correctness.score == 0


@pytest.mark.asyncio
async def test_run_evaluation_batch():
    """run_evaluation should process multiple QA pairs."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    # Mock judge to return scores
    from graphrag.eval.metrics import MetricScores, JudgeResult
    mock_scores = MetricScores(
        correctness=JudgeResult(1, "OK"),
        faithfulness=JudgeResult(1, "OK"),
        relevance=JudgeResult(1, "OK"),
        completeness=JudgeResult(1, "OK"),
    )
    mock_judge.evaluate_all = AsyncMock(return_value=mock_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576", "tt0102798": "tt0102798"},
    )

    sample_qa_dataset = [
        {"question": "Q1?", "answer": "A1", "imdb_key": "tt0097576"},
        {"question": "Q2?", "answer": "A2", "imdb_key": "tt0097576"},
        {"question": "Q3?", "answer": "A3", "imdb_key": "tt0102798"},
    ]

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

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

    with patch.object(runner, '_load_index', return_value=mock_index_data), \
         patch.object(runner, '_run_search', return_value=mock_result):
        results = await runner.run_evaluation(
            dataset=sample_qa_dataset,
            methods=["tog"],
        )

    assert len(results) == 3  # 3 QA pairs * 1 method


@pytest.mark.asyncio
async def test_run_evaluation_multiple_methods():
    """run_evaluation should test each method for each QA pair."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    # Mock judge to return scores
    from graphrag.eval.metrics import MetricScores, JudgeResult
    mock_scores = MetricScores(
        correctness=JudgeResult(1, "OK"),
        faithfulness=JudgeResult(1, "OK"),
        relevance=JudgeResult(1, "OK"),
        completeness=JudgeResult(1, "OK"),
    )
    mock_judge.evaluate_all = AsyncMock(return_value=mock_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    dataset = [{"question": "Test?", "answer": "Yes", "imdb_key": "tt0097576"}]

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

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

    with patch.object(runner, '_load_index', return_value=mock_index_data), \
         patch.object(runner, '_run_search', return_value=mock_result):
        results = await runner.run_evaluation(
            dataset=dataset,
            methods=["tog", "local", "basic"],
        )

    assert len(results) == 3  # 1 QA pair * 3 methods


@pytest.mark.asyncio
async def test_progress_callback():
    """run_evaluation should call progress callback."""
    mock_config = MagicMock()
    mock_judge = MagicMock()

    # Mock judge to return scores
    from graphrag.eval.metrics import MetricScores, JudgeResult
    mock_scores = MetricScores(
        correctness=JudgeResult(1, "OK"),
        faithfulness=JudgeResult(1, "OK"),
        relevance=JudgeResult(1, "OK"),
        completeness=JudgeResult(1, "OK"),
    )
    mock_judge.evaluate_all = AsyncMock(return_value=mock_scores)

    runner = EvaluationRunner(
        config=mock_config,
        judge=mock_judge,
        index_roots={"tt0097576": "tt0097576"},
    )

    dataset = [{"question": "Test?", "answer": "Yes", "imdb_key": "tt0097576"}]
    progress_calls = []

    def progress(current, total, movie, method):
        progress_calls.append((current, total, movie, method))

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

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

    with patch.object(runner, '_load_index', return_value=mock_index_data), \
         patch.object(runner, '_run_search', return_value=mock_result):
        await runner.run_evaluation(
            dataset=dataset,
            methods=["tog"],
            progress_callback=progress,
        )

    assert len(progress_calls) == 1
    assert progress_calls[0][0] == 1  # current
    assert progress_calls[0][1] == 1  # total
