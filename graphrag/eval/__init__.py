"""
GraphRAG Evaluation Framework.

Provides tools for evaluating and comparing search methods:
- LLM-as-Judge metrics (correctness, faithfulness, relevance, completeness)
- Evaluation runner for batch processing
- Result aggregation and reporting
"""

from graphrag.eval.metrics import (
    LLMJudge,
    JudgeResult,
    MetricScores,
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    COMPLETENESS_PROMPT,
)
from graphrag.eval.runner import (
    EvaluationRunner,
    EvaluationConfig,
    QueryResult,
    EfficiencyMetrics,
)
from graphrag.eval.results import (
    EvaluationResults,
    MethodSummary,
    aggregate_results,
)

__all__ = [
    # Metrics
    "LLMJudge",
    "JudgeResult",
    "MetricScores",
    "CORRECTNESS_PROMPT",
    "FAITHFULNESS_PROMPT",
    "RELEVANCE_PROMPT",
    "COMPLETENESS_PROMPT",
    # Runner
    "EvaluationRunner",
    "EvaluationConfig",
    "QueryResult",
    "EfficiencyMetrics",
    # Results
    "EvaluationResults",
    "MethodSummary",
    "aggregate_results",
]
