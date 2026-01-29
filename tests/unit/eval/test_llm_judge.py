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
async def test_judge_correctness_score_1():
    """Judge should return score=1 when LLM indicates correct."""
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
    assert "Matches" in result.reason


@pytest.mark.asyncio
async def test_judge_correctness_score_0():
    """Judge should return score=0 when LLM indicates incorrect."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 0, "reason": "Wrong answer"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_correctness(
        question="What color is the sky?",
        ground_truth="Blue",
        predicted="The sky is green.",
    )

    assert result.score == 0


@pytest.mark.asyncio
async def test_judge_faithfulness():
    """Judge should evaluate faithfulness."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 0, "reason": "Hallucinated content not in context"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_faithfulness(
        context="The document discusses weather patterns in Europe.",
        answer="The stock market rose 5% in Asia today.",
    )

    assert result.score == 0
    assert "Hallucinated" in result.reason


@pytest.mark.asyncio
async def test_judge_relevance():
    """Judge should evaluate context relevance."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "Context directly addresses the question"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_relevance(
        question="Who is Indiana Jones?",
        context="Indiana Jones is an archaeologist and professor.",
    )

    assert result.score == 1


@pytest.mark.asyncio
async def test_judge_completeness():
    """Judge should evaluate answer completeness."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 0, "reason": "Missing key information"}'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_completeness(
        question="What are the three colors of the French flag?",
        answer="The flag has blue and white.",
    )

    assert result.score == 0


@pytest.mark.asyncio
async def test_evaluate_all():
    """evaluate_all should return MetricScores with all four metrics."""
    mock_model = MagicMock()
    # Each judge call gets its own response
    mock_model.achat_stream = MagicMock(
        side_effect=lambda *args, **kwargs: AsyncIteratorMock(['{"score": 1, "reason": "Good"}'])
    )

    judge = LLMJudge(model=mock_model)
    scores = await judge.evaluate_all(
        question="Test question",
        ground_truth="Expected answer",
        predicted="Actual answer",
        context="Some context",
    )

    from graphrag.eval.metrics import MetricScores
    assert isinstance(scores, MetricScores)
    assert scores.correctness.score == 1
    assert scores.faithfulness.score == 1
    assert scores.relevance.score == 1
    assert scores.completeness.score == 1


@pytest.mark.asyncio
async def test_judge_handles_malformed_json():
    """Judge should handle malformed JSON gracefully."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(["This is not valid JSON"])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    # Should return a result (with inferred score) rather than raising
    assert isinstance(result, JudgeResult)


@pytest.mark.asyncio
async def test_judge_handles_markdown_code_blocks():
    """Judge should extract JSON from markdown code blocks."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['```json\n{"score": 1, "reason": "OK"}\n```'])
    )

    judge = LLMJudge(model=mock_model)
    result = await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    assert result.score == 1


@pytest.mark.asyncio
async def test_judge_uses_temperature_0():
    """Judge should use temperature=0 for consistency."""
    mock_model = MagicMock()
    mock_model.achat_stream = MagicMock(
        return_value=AsyncIteratorMock(['{"score": 1, "reason": "OK"}'])
    )

    judge = LLMJudge(model=mock_model, temperature=0.0)

    await judge.judge_correctness(
        question="Test",
        ground_truth="Expected",
        predicted="Actual",
    )

    # Verify temperature was passed to model
    call_args = mock_model.achat_stream.call_args
    assert call_args[1]["model_parameters"]["temperature"] == 0.0
