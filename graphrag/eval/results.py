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
        return {
            "metadata": self.metadata,
            "results": [
                {
                    "imdb_key": r.imdb_key,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "methods": {
                        r.method: {
                            "response": r.response,
                            "scores": r.scores.to_dict(),
                            "efficiency": r.efficiency.to_dict(),
                        }
                    },
                }
                for r in self.results
            ],
        }

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

    # Collect unique methods and movies
    methods = set(r.method for r in results)
    movies = set(r.imdb_key for r in results)

    # Aggregate by method
    by_method = {}
    for method in methods:
        method_results = [r for r in results if r.method == method]
        if method_results:
            by_method[method] = {
                "correctness": sum(r.scores.correctness.score for r in method_results) / len(method_results),
                "faithfulness": sum(r.scores.faithfulness.score for r in method_results) / len(method_results),
                "relevance": sum(r.scores.relevance.score for r in method_results) / len(method_results),
                "completeness": sum(r.scores.completeness.score for r in method_results) / len(method_results),
            }

    # Aggregate by movie
    by_movie = {}
    for movie in movies:
        by_movie[movie] = {}
        for method in methods:
            movie_method_results = [r for r in results if r.imdb_key == movie and r.method == method]
            if movie_method_results:
                by_movie[movie][method] = {
                    "correctness": sum(r.scores.correctness.score for r in movie_method_results) / len(movie_method_results),
                    "faithfulness": sum(r.scores.faithfulness.score for r in movie_method_results) / len(movie_method_results),
                    "relevance": sum(r.scores.relevance.score for r in movie_method_results) / len(movie_method_results),
                    "completeness": sum(r.scores.completeness.score for r in movie_method_results) / len(movie_method_results),
                }

    # Aggregate efficiency by method
    efficiency = {}
    for method in methods:
        method_results = [r for r in results if r.method == method]
        if method_results:
            efficiency[method] = {
                "avg_latency": sum(r.efficiency.latency for r in method_results) / len(method_results),
                "avg_llm_calls": sum(r.efficiency.llm_calls for r in method_results) / len(method_results),
                "avg_prompt_tokens": sum(r.efficiency.prompt_tokens for r in method_results) / len(method_results),
                "avg_output_tokens": sum(r.efficiency.output_tokens for r in method_results) / len(method_results),
            }

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(set((r.imdb_key, r.question) for r in results)),
        "methods": list(methods),
        "movies": list(movies),
    }

    return EvaluationResults(
        metadata=metadata,
        results=results,
        by_method=by_method,
        by_movie=by_movie,
        efficiency=efficiency,
    )
