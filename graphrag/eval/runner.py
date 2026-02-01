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

    question: str
    method: str
    response: str
    context: str  # context from dataset
    context_text: str  # context from search
    ground_truth: str
    efficiency: EfficiencyMetrics | None = None
    scores: MetricScores | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "question": self.question,
            "method": self.method,
            "response": self.response,
            "context": self.context,
            "context_text": self.context_text,
            "ground_truth": self.ground_truth,
        }
        if self.scores:
            result["scores"] = self.scores.to_dict()
        if self.efficiency:
            result["efficiency"] = self.efficiency.to_dict()
        return result

    def to_simple_dict(self) -> dict[str, Any]:
        """Return simple dict with required fields for JSON output."""
        result = {
            "question": self.question,
            "response": self.response,
            "context": self.context,
            "context_text": self.context_text,
            "ground_truth": self.ground_truth,
            "method": self.method,
        }
        if self.efficiency:
            result["latency"] = self.efficiency.latency
            result["llm_calls"] = self.efficiency.llm_calls
            result["prompt_tokens"] = self.efficiency.prompt_tokens
            result["output_tokens"] = self.efficiency.output_tokens
        return result


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
            community_reports = await load_table_from_storage(
                "community_reports", storage
            )
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
        from graphrag.config.embeddings import (
            entity_description_embedding,
            text_unit_text_embedding,
        )
        from graphrag.query.factory import (
            get_basic_search_engine,
            get_local_search_engine,
            get_tog_search_engine,
        )
        from graphrag.query.indexer_adapters import (
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.utils.api import get_embedding_store, load_search_prompt

        config = index_data["config"]
        entities = index_data["entities"]
        relationships = index_data["relationships"]

        if method == "tog":
            entities_ = read_indexer_entities(
                entities=entities,
                communities=None,
                community_level=None,
            )
            relationships_ = read_indexer_relationships(relationships)

            vector_store_args = {
                index: store.model_dump()
                for index, store in config.vector_store.items()
            }
            entity_text_embeddings = get_embedding_store(
                config_args=vector_store_args,
                embedding_name=entity_description_embedding,
            )

            search_engine = get_tog_search_engine(
                config=config,
                entities=entities_,
                relationships=relationships_,
                response_type="detailed",
                entity_text_embeddings=entity_text_embeddings,
            )
            return await search_engine.search(query=query)

        elif method == "local":
            communities = index_data["communities"]
            community_reports = index_data["community_reports"]
            text_units = index_data["text_units"]

            if communities is None:
                raise ValueError(f"Local search requires communities table")

            vector_store_args = {
                index: store.model_dump()
                for index, store in config.vector_store.items()
            }
            description_embedding_store = get_embedding_store(
                config_args=vector_store_args,
                embedding_name=entity_description_embedding,
            )
            prompt = load_search_prompt(config.root_dir, config.local_search.prompt)

            search_engine = get_local_search_engine(
                config=config,
                reports=read_indexer_reports(
                    community_reports, communities, community_level=2
                ),
                text_units=read_indexer_text_units(text_units),
                entities=read_indexer_entities(
                    entities=entities, communities=communities, community_level=2
                ),
                relationships=read_indexer_relationships(relationships),
                covariates={"claims": []},
                response_type="Multiple Paragraphs",
                description_embedding_store=description_embedding_store,
                system_prompt=prompt,
            )
            return await search_engine.search(query=query)

        elif method == "basic":
            text_units = index_data["text_units"]
            if text_units is None:
                raise ValueError(f"Basic search requires text_units table")

            vector_store_args = {
                index: store.model_dump()
                for index, store in config.vector_store.items()
            }
            embedding_store = get_embedding_store(
                config_args=vector_store_args,
                embedding_name=text_unit_text_embedding,
            )
            prompt = load_search_prompt(config.root_dir, config.basic_search.prompt)

            search_engine = get_basic_search_engine(
                config=config,
                text_units=read_indexer_text_units(text_units),
                text_unit_embeddings=embedding_store,
                system_prompt=prompt,
            )
            return await search_engine.search(query=query)

        else:
            raise ValueError(f"Unknown method: {method}")

    async def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        context: str,
        method: str,
        index_data: dict,
        skip_evaluation: bool = False,
    ) -> QueryResult:
        """Evaluate a single query with a single method."""
        import time

        # Run search
        start_time = time.time()
        try:
            result = await self._run_search(method, question, index_data)
            latency = time.time() - start_time
        except Exception as e:
            logger.error(f"Search failed for {method}: {e}")
            # Return failure result
            from graphrag.eval.metrics import JudgeResult

            if skip_evaluation:
                return QueryResult(
                    question=question,
                    method=method,
                    response=f"ERROR: {str(e)}",
                    context=context,
                    context_text="",
                    ground_truth=ground_truth,
                )
            else:
                return QueryResult(
                    question=question,
                    method=method,
                    response=f"ERROR: {str(e)}",
                    context=context,
                    context_text="",
                    ground_truth=ground_truth,
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

        response_text = (
            result.response
            if isinstance(result.response, str)
            else str(result.response)
        )
        context_text = (
            result.context_text
            if isinstance(result.context_text, str)
            else str(result.context_text)
        )

        # Skip evaluation if requested
        if skip_evaluation:
            return QueryResult(
                question=question,
                method=method,
                response=response_text,
                context=context,
                context_text=context_text,
                ground_truth=ground_truth,
                efficiency=EfficiencyMetrics(
                    latency=latency,
                    llm_calls=result.llm_calls,
                    prompt_tokens=result.prompt_tokens,
                    output_tokens=result.output_tokens,
                ),
            )

        # Evaluate with judge
        scores = await self.judge.evaluate_all(
            question=question,
            ground_truth=ground_truth,
            predicted=response_text,
            context=context_text,
        )

        return QueryResult(
            question=question,
            method=method,
            response=response_text,
            context=context,
            context_text=context_text,
            ground_truth=ground_truth,
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
        index_data: dict,
        progress_callback: callable = None,
        skip_evaluation: bool = False,
    ) -> list[QueryResult]:
        """Run evaluation on full dataset."""
        results = []
        total = len(dataset) * len(methods)
        current = 0

        for qa in dataset:
            question = qa["question"]
            ground_truth = qa["ground_truth"]
            context = qa.get("context", "")

            for method in methods:
                current += 1
                if progress_callback:
                    progress_callback(current, total, question[:50], method)

                try:
                    result = await self.evaluate_single(
                        question=question,
                        ground_truth=ground_truth,
                        context=context,
                        method=method,
                        index_data=index_data,
                        skip_evaluation=skip_evaluation,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")

        return results
