"""Evaluation runner for batch processing queries."""

from dataclasses import dataclass, field
from typing import Any
import json
import logging
import asyncio
from pathlib import Path

from graphrag.eval.metrics import MetricScores, LLMJudge
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.structured_search.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics from a search operation."""
    latency: float  # seconds
    llm_calls: int
    prompt_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": self.latency,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class QueryResult:
    """Result from evaluating a single query with a single method."""
    imdb_key: str
    question: str
    method: str
    response: str
    scores: MetricScores
    efficiency: EfficiencyMetrics
    ground_truth: str = ""
    context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "imdb_key": self.imdb_key,
            "question": self.question,
            "method": self.method,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "scores": self.scores.to_dict(),
            "efficiency": self.efficiency.to_dict(),
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    dataset_path: str
    index_roots: dict[str, str]  # imdb_key -> root path
    methods: list[str] = field(default_factory=lambda: ["tog", "local", "basic"])
    output_dir: str = "eval/results"
    save_intermediate: bool = True


class EvaluationRunner:
    """Runs evaluation across multiple QA pairs and methods."""

    def __init__(
        self,
        config: GraphRagConfig,
        judge: LLMJudge,
        index_roots: dict[str, str] | None = None,
    ):
        self.config = config
        self.judge = judge
        self.index_roots = index_roots or {}
        self._loaded_indexes: dict[str, dict] = {}

    async def _load_index(self, imdb_key: str) -> dict:
        """Load index data for a movie."""
        if imdb_key in self._loaded_indexes:
            return self._loaded_indexes[imdb_key]

        root = self.index_roots.get(imdb_key, imdb_key)
        from graphrag.config.load_config import load_config
        from graphrag.utils.api import create_storage_from_config
        from graphrag.utils.storage import load_table_from_storage

        config = load_config(Path(root))
        storage = create_storage_from_config(config.output)

        # Load required tables
        entities = await load_table_from_storage("entities", storage)
        relationships = await load_table_from_storage("relationships", storage)

        # Load optional tables for local search
        try:
            communities = await load_table_from_storage("communities", storage)
            community_reports = await load_table_from_storage("community_reports", storage)
            text_units = await load_table_from_storage("text_units", storage)
        except Exception:
            communities = None
            community_reports = None
            text_units = None

        self._loaded_indexes[imdb_key] = {
            "config": config,
            "entities": entities,
            "relationships": relationships,
            "communities": communities,
            "community_reports": community_reports,
            "text_units": text_units,
        }

        return self._loaded_indexes[imdb_key]

    async def _run_search(
        self,
        method: str,
        query: str,
        index_data: dict,
    ) -> SearchResult:
        """Run a search method and return SearchResult."""
        import graphrag.api as api

        config = index_data["config"]
        entities = index_data["entities"]
        relationships = index_data["relationships"]

        if method == "tog":
            response, context = await api.tog_search(
                config=config,
                entities=entities,
                relationships=relationships,
                query=query,
            )
            # Wrap in SearchResult if not already
            if isinstance(response, str):
                return SearchResult(
                    response=response,
                    context_data=context,
                    context_text=str(context),
                    completion_time=0,  # Not tracked in current API
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                )
            return response

        elif method == "local":
            communities = index_data["communities"]
            community_reports = index_data["community_reports"]
            text_units = index_data["text_units"]

            if communities is None:
                raise ValueError(f"Local search requires communities table")

            response, context = await api.local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=None,
                community_level=2,
                response_type="Multiple Paragraphs",
                query=query,
            )
            return SearchResult(
                response=response,
                context_data=context,
                context_text=str(context),
                completion_time=0,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

        elif method == "basic":
            text_units = index_data["text_units"]
            if text_units is None:
                raise ValueError(f"Basic search requires text_units table")

            response, context = await api.basic_search(
                config=config,
                text_units=text_units,
                query=query,
            )
            return SearchResult(
                response=response,
                context_data=context,
                context_text=str(context),
                completion_time=0,
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    async def evaluate_single(
        self,
        imdb_key: str,
        question: str,
        ground_truth: str,
        method: str,
    ) -> QueryResult:
        """Evaluate a single query with a single method."""
        import time

        # Load index
        index_data = await self._load_index(imdb_key)

        # Run search
        start_time = time.time()
        try:
            result = await self._run_search(method, question, index_data)
            latency = time.time() - start_time
        except Exception as e:
            logger.error(f"Search failed for {imdb_key}/{method}: {e}")
            # Return failure result
            from graphrag.eval.metrics import JudgeResult
            return QueryResult(
                imdb_key=imdb_key,
                question=question,
                method=method,
                response=f"ERROR: {str(e)}",
                ground_truth=ground_truth,
                context_text="",
                scores=MetricScores(
                    correctness=JudgeResult(0, "Search failed"),
                    faithfulness=JudgeResult(0, "Search failed"),
                    relevance=JudgeResult(0, "Search failed"),
                    completeness=JudgeResult(0, "Search failed"),
                ),
                efficiency=EfficiencyMetrics(
                    latency=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                    output_tokens=0,
                ),
            )

        # Evaluate with judge
        scores = await self.judge.evaluate_all(
            question=question,
            ground_truth=ground_truth,
            predicted=result.response if isinstance(result.response, str) else str(result.response),
            context=result.context_text if isinstance(result.context_text, str) else str(result.context_text),
        )

        return QueryResult(
            imdb_key=imdb_key,
            question=question,
            method=method,
            response=result.response if isinstance(result.response, str) else str(result.response),
            ground_truth=ground_truth,
            context_text=result.context_text if isinstance(result.context_text, str) else str(result.context_text),
            scores=scores,
            efficiency=EfficiencyMetrics(
                latency=latency,
                llm_calls=result.llm_calls,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
            ),
        )

    async def run_evaluation(
        self,
        dataset: list[dict],
        methods: list[str],
        progress_callback: callable = None,
    ) -> list[QueryResult]:
        """Run evaluation on full dataset."""
        results = []
        total = len(dataset) * len(methods)
        current = 0

        for qa in dataset:
            imdb_key = qa["imdb_key"]
            question = qa["question"]
            ground_truth = qa["answer"]

            for method in methods:
                current += 1
                if progress_callback:
                    progress_callback(current, total, imdb_key, method)

                try:
                    result = await self.evaluate_single(
                        imdb_key=imdb_key,
                        question=question,
                        ground_truth=ground_truth,
                        method=method,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")

        return results
