"""
Tests for Results aggregation.
"""
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
