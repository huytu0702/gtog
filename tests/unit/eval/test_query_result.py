"""
Tests for QueryResult and EfficiencyMetrics dataclasses.
"""
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
