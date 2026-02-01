"""Shared fixtures for eval tests."""

import pytest
from unittest.mock import MagicMock


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
        correctness=JudgeResult(1, "Correct answer"),
        faithfulness=JudgeResult(1, "Supported by context"),
        relevance=JudgeResult(1, "Relevant context"),
        completeness=JudgeResult(1, "Complete answer"),
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
