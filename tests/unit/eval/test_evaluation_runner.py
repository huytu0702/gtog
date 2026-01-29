"""
Tests for EvaluationRunner class.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphrag.eval.runner import EvaluationRunner


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
