"""
Tests for LLMJudge class.
"""
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
