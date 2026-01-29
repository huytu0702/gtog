"""
Tests for JudgeResult and MetricScores dataclasses.
"""
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
