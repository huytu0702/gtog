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
    # Each judge call gets its own response
    mock_model.achat_stream = MagicMock(
        side_effect=lambda *args, **kwargs: AsyncIteratorMock([
            '{"score": 1, "reason": "Correct"}'
        ])
    )

    # Create judge
    judge = LLMJudge(model=mock_model)

    # Create mock config
    mock_config = MagicMock()

    # Create runner
    runner = EvaluationRunner(
        config=mock_config,
        judge=judge,
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

    # Sample dataset (new format without imdb_key)
    dataset = [
        {"question": "Q1?", "ground_truth": "A1", "context": "Context 1"},
        {"question": "Q2?", "ground_truth": "A2", "context": "Context 2"},
    ]

    # Mock the search
    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

    with patch.object(runner, "_run_search", return_value=mock_search_result):
        # Run evaluation
        results = await runner.run_evaluation(
            dataset=dataset,
            methods=["tog"],
            index_data=mock_index_data,
        )

    # Verify results
    assert len(results) == 2
    for r in results:
        assert r.scores is not None
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

    # New format dataset
    dataset = [
        {"question": "Q1?", "ground_truth": "A1", "context": "Context 1"},
    ]

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

    with patch.object(runner, "_run_search", return_value=mock_search_result):
        results = await runner.run_evaluation(
            dataset=dataset,
            methods=["tog"],
            index_data=mock_index_data,
        )

    # First query gets scores: correctness=1, faithfulness=0, relevance=1, completeness=0
    # (alternating pattern)
    aggregated = aggregate_results(results)
    # Due to alternating, we should see mixed averages
    # This tests the aggregation logic handles non-perfect scores
    assert aggregated.by_method["tog"]["correctness"] == 1.0
    assert aggregated.by_method["tog"]["faithfulness"] == 0.0


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
    )

    # New format dataset
    dataset = [
        {"question": "Q1?", "ground_truth": "A1", "context": "Context 1"},
        {"question": "Q2?", "ground_truth": "A2", "context": "Context 2"},
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

    mock_index_data = {
        "config": mock_config,
        "entities": MagicMock(),
        "relationships": MagicMock(),
        "communities": MagicMock(),
        "community_reports": MagicMock(),
        "text_units": MagicMock(),
    }

    with patch.object(runner, "_run_search", side_effect=failing_search):
        results = await runner.run_evaluation(
            dataset=dataset,
            methods=["tog"],
            index_data=mock_index_data,
        )

    # Should have 2 results (one error, one success)
    assert len(results) == 2

    # First should be error
    assert "ERROR" in results[0].response
    assert results[0].scores is not None
    assert results[0].scores.correctness.score == 0

    # Second should succeed
    assert "Success" in results[1].response
