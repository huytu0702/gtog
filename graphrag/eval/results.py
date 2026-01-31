"""Result aggregation and output formatting."""

from dataclasses import dataclass
from typing import Any
from datetime import datetime
import json
from pathlib import Path

from graphrag.eval.runner import QueryResult


@dataclass
class MethodSummary:
    """Summary statistics for a single method."""

    correctness: float
    faithfulness: float
    relevance: float
    completeness: float
    avg_latency: float
    avg_llm_calls: float
    avg_prompt_tokens: float
    avg_output_tokens: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": round(self.correctness, 3),
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "completeness": round(self.completeness, 3),
            "avg_latency": round(self.avg_latency, 3),
            "avg_llm_calls": round(self.avg_llm_calls, 1),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 0),
            "avg_output_tokens": round(self.avg_output_tokens, 0),
            "count": self.count,
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    metadata: dict[str, Any]
    results: list[QueryResult]
    by_method: dict[str, dict[str, float]]
    by_movie: dict[str, dict[str, dict[str, float]]]
    efficiency: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "by_method": self.by_method,
            "by_movie": self.by_movie,
            "efficiency": self.efficiency,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_detailed_dict(self) -> dict[str, Any]:
        """Full results including per-query details."""
        results_list = []
        for r in self.results:
            result_entry = {
                "question": r.question,
                "ground_truth": r.ground_truth,
                "context": r.context,
                "methods": {
                    r.method: {
                        "response": r.response,
                        "context_text": r.context_text,
                    }
                },
            }
            # Only include scores and efficiency if they exist
            if r.scores is not None and r.efficiency is not None:
                result_entry["methods"][r.method]["scores"] = r.scores.to_dict()
                result_entry["methods"][r.method]["efficiency"] = r.efficiency.to_dict()
            results_list.append(result_entry)

        return {
            "metadata": self.metadata,
            "results": results_list,
        }

    def to_simple_dict(self) -> list[dict[str, Any]]:
        """Simple results without evaluation metrics - just imdb_key, question, ground_truth, context_text, response."""
        return [r.to_simple_dict() for r in self.results]

    def to_simple_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_simple_dict(), indent=indent)

    def to_detailed_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_detailed_dict(), indent=indent)

    def save(self, output_dir: str) -> None:
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path / "eval_results_summary.json"
        with open(summary_path, "w") as f:
            f.write(self.to_json())

        # Save detailed results
        detailed_path = output_path / "eval_results_detailed.json"
        with open(detailed_path, "w") as f:
            f.write(self.to_detailed_json())

    def save_simple(self, output_dir: str) -> None:
        """Save simple results (no evaluation metrics) to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save simple results
        simple_path = output_path / "eval_results_simple.json"
        with open(simple_path, "w") as f:
            f.write(self.to_simple_json())


def aggregate_results(results: list[QueryResult]) -> EvaluationResults:
    """Aggregate QueryResults into summary statistics."""
    if not results:
        return EvaluationResults(
            metadata={"timestamp": datetime.now().isoformat(), "total_questions": 0},
            results=[],
            by_method={},
            by_movie={},
            efficiency={},
        )

    # Collect unique methods
    methods = set(r.method for r in results)

    # Aggregate by method (only if scores exist)
    by_method: dict[str, dict[str, float]] = {}
    for method in methods:
        method_results = [
            r for r in results if r.method == method and r.scores is not None
        ]
        if method_results:
            # type: ignore[union-attr] - filtered for non-None scores above
            by_method[method] = {
                "correctness": sum(r.scores.correctness.score for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
                "faithfulness": sum(r.scores.faithfulness.score for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
                "relevance": sum(r.scores.relevance.score for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
                "completeness": sum(r.scores.completeness.score for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
            }

    # Aggregate efficiency by method (only if efficiency exists)
    efficiency: dict[str, dict[str, float]] = {}
    for method in methods:
        method_results = [
            r for r in results if r.method == method and r.efficiency is not None
        ]
        if method_results:
            # type: ignore[union-attr] - filtered for non-None efficiency above
            efficiency[method] = {
                "avg_latency": sum(r.efficiency.latency for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
                "avg_llm_calls": sum(r.efficiency.llm_calls for r in method_results)  # type: ignore[union-attr]
                / len(method_results),
                "avg_prompt_tokens": sum(
                    r.efficiency.prompt_tokens
                    for r in method_results  # type: ignore[union-attr]
                )
                / len(method_results),
                "avg_output_tokens": sum(
                    r.efficiency.output_tokens
                    for r in method_results  # type: ignore[union-attr]
                )
                / len(method_results),
            }

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(set(r.question for r in results)),
        "methods": list(methods),
    }

    return EvaluationResults(
        metadata=metadata,
        results=results,
        by_method=by_method,
        by_movie={},
        efficiency=efficiency,
    )
