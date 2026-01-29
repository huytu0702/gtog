"""
Tests for Results aggregation.
"""
import pytest
from graphrag.eval.results import EvaluationResults, aggregate_results
from graphrag.eval.runner import QueryResult, EfficiencyMetrics
from graphrag.eval.metrics import MetricScores, JudgeResult


def create_mock_result(imdb_key: str, method: str, correct: int, latency: float = 1.0) -> QueryResult:
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
        create_mock_result("tt1", "tog", 1),
        create_mock_result("tt2", "tog", 0),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.by_movie["tt1"]["tog"]["correctness"] == 1.0
    assert aggregated.by_movie["tt2"]["tog"]["correctness"] == 0.0


def test_aggregate_efficiency():
    """aggregate_results should compute efficiency averages."""
    results = [
        create_mock_result("tt1", "tog", 1, latency=2.0),
        create_mock_result("tt1", "tog", 1, latency=4.0),
    ]

    aggregated = aggregate_results(results)

    assert aggregated.efficiency["tog"]["avg_latency"] == 3.0


def test_to_json():
    """EvaluationResults should serialize to valid JSON."""
    results = [create_mock_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    json_str = aggregated.to_json()
    import json
    parsed = json.loads(json_str)

    assert "by_method" in parsed
    assert "by_movie" in parsed
    assert "efficiency" in parsed
    assert "metadata" in parsed


def test_to_detailed_json():
    """EvaluationResults should serialize detailed results."""
    results = [create_mock_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    json_str = aggregated.to_detailed_json()
    import json
    parsed = json.loads(json_str)

    assert "results" in parsed
    assert len(parsed["results"]) == 1


def test_metadata_contains_expected_fields():
    """Metadata should contain timestamp, totals, methods, movies."""
    results = [
        create_mock_result("tt1", "tog", 1),
        create_mock_result("tt2", "local", 1),
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
    results = [create_mock_result("tt1", "tog", 1)]
    aggregated = aggregate_results(results)

    output_dir = tmp_path / "results"
    aggregated.save(str(output_dir))

    assert (output_dir / "eval_results_summary.json").exists()
    assert (output_dir / "eval_results_detailed.json").exists()

    # Verify content
    import json
    with open(output_dir / "eval_results_summary.json") as f:
        summary = json.load(f)
    assert "by_method" in summary
