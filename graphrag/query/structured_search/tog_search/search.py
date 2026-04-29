import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.data_model.text_unit import TextUnit
from graphrag.tokenizer.tokenizer import Tokenizer
from graphrag.vector_stores.base import BaseVectorStore
from graphrag.query.structured_search.base import SearchResult
from .state import ToGSearchState, ExplorationNode
from .exploration import GraphExplorer
from .pruning import PruningStrategy, LLMPruning, SemanticPruning, PruningMetrics
from .reasoning import ToGReasoning, ReasoningMetrics
from graphrag.query.context_builder.conversation_history import ConversationHistory

logger = logging.getLogger(__name__)


@dataclass
class ToGMetrics:
    """Aggregated metrics for ToG search."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    exploration_llm_calls: int = 0
    reasoning_llm_calls: int = 0
    exploration_prompt_tokens: int = 0
    reasoning_prompt_tokens: int = 0
    exploration_output_tokens: int = 0
    reasoning_output_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    def add_pruning(self, m: PruningMetrics) -> None:
        """Add pruning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.exploration_llm_calls += m.llm_calls
        self.exploration_prompt_tokens += m.prompt_tokens
        self.exploration_output_tokens += m.output_tokens
        self.embedding_calls += m.embedding_calls
        self.embedding_tokens += m.embedding_tokens

    def add_reasoning(self, m: ReasoningMetrics) -> None:
        """Add reasoning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.reasoning_llm_calls += m.llm_calls
        self.reasoning_prompt_tokens += m.prompt_tokens
        self.reasoning_output_tokens += m.output_tokens


class ToGSearch:
    """
    ToG (Think-on-Graph) Search Engine for GraphRAG.

    Implements iterative graph exploration with LLM-guided pruning
    and reasoning over discovered paths.

    Uses embedding-based entity linking (like original ToG paper).
    """

    def __init__(
        self,
        model: ChatModel,
        entities: List[Entity],
        relationships: List[Relationship],
        tokenizer: Tokenizer,
        pruning_strategy: PruningStrategy,
        reasoning_module: ToGReasoning,
        text_units: List[TextUnit] | None = None,
        embedding_model: Optional[EmbeddingModel] = None,
        entity_text_embeddings: Optional[BaseVectorStore] = None,
        width: int = 3,
        depth: int = 3,
        num_retain_entity: int = 5,
        callbacks: List[QueryCallbacks] | None = None,
        debug: bool = False,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.explorer = GraphExplorer(
            entities,
            relationships,
            text_units=text_units,
            embedding_model=embedding_model,
            entity_embedding_store=entity_text_embeddings,
        )
        self.tokenizer = tokenizer
        self.pruning_strategy = pruning_strategy
        self.reasoning_module = reasoning_module
        self.width = width
        self.depth = depth
        self.num_retain_entity = num_retain_entity
        self.callbacks = callbacks or []
        self._debug = debug

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> SearchResult:
        """Perform ToG search and return SearchResult with metrics."""
        start_time = time.time()
        metrics = ToGMetrics()

        response_chunks: List[str] = []
        context_paths: List[str] = []
        context_text = ""

        async for (
            chunk,
            paths,
            chunk_metrics,
            ctx_text,
        ) in self._stream_search_with_metrics(query, conversation_history):
            if chunk:
                response_chunks.append(chunk)
            if paths:
                context_paths = paths
            if chunk_metrics:
                if isinstance(chunk_metrics, PruningMetrics):
                    metrics.add_pruning(chunk_metrics)
                elif isinstance(chunk_metrics, ReasoningMetrics):
                    metrics.add_reasoning(chunk_metrics)
            if ctx_text:
                context_text = ctx_text

        response = "".join(response_chunks)
        completion_time = time.time() - start_time

        return SearchResult(
            response=response,
            context_data={"exploration_paths": context_paths},
            context_text=context_text,
            completion_time=completion_time,
            llm_calls=metrics.llm_calls,
            prompt_tokens=metrics.prompt_tokens,
            output_tokens=metrics.output_tokens,
            llm_calls_categories={
                "exploration": metrics.exploration_llm_calls,
                "reasoning": metrics.reasoning_llm_calls,
            },
            prompt_tokens_categories={
                "exploration": metrics.exploration_prompt_tokens,
                "reasoning": metrics.reasoning_prompt_tokens,
            },
            output_tokens_categories={
                "exploration": metrics.exploration_output_tokens,
                "reasoning": metrics.reasoning_output_tokens,
            },
        )

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator[str, None]:
        """Perform ToG search with streaming output (backward compatible)."""
        async for chunk, _, _, _ in self._stream_search_with_metrics(query, conversation_history):
            if chunk:  # Only yield non-empty chunks
                yield chunk

    async def _stream_search_with_metrics(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator[
        Tuple[str, List[str], Union[PruningMetrics, ReasoningMetrics, None], str], None
    ]:
        """Perform ToG search with streaming output."""
        logger.info(
            "[ToG][start] width=%d depth=%d pruning=%s reasoning=%s query_len=%d",
            self.width,
            self.depth,
            type(self.pruning_strategy).__name__,
            type(self.reasoning_module).__name__,
            len(query),
        )

        # Enrich query for entity linking with previous user questions
        effective_query = query
        if conversation_history:
            past_questions = "\n".join(
                conversation_history.get_user_turns(max_user_turns=5)
            )
            if past_questions:
                effective_query = f"{query}\n{past_questions}"

        # Build history context string for reasoning
        history_context = ""
        if conversation_history:
            history_context, _ = conversation_history.build_context(
                include_user_turns_only=False,
                max_qa_turns=5,
                recency_bias=False,
            )

        # Find initial entities using semantic similarity (like ToG paper)
        if self.embedding_model:
            logger.info("[ToG][entity_linking] semantic_start top_k=%d", self.width)
            starting_entities = await self.explorer.find_starting_entities_semantic(
                effective_query, top_k=self.width
            )
        else:
            logger.info("[ToG][entity_linking] lexical_start top_k=%d", self.width)
            starting_entities = self.explorer.find_starting_entities(
                effective_query, top_k=self.width
            )

        logger.info("[ToG][entity_linking] found=%d", len(starting_entities))
        logger.debug("[ToG][entity_linking] ids=%s", starting_entities)

        if not starting_entities:
            available_entities = list(self.explorer.entities.keys())[:10]
            yield (
                f"No relevant entities found for query '{query}'. Available entities: {available_entities}",
                [],
                None,
                "",
            )
            return

        # Initialize search state
        state = ToGSearchState(
            query=query,
            current_depth=0,
            nodes_by_depth={0: []},
            finished_paths=[],
            max_depth=self.depth,
            beam_width=self.width,
        )

        # Create initial nodes from starting entities
        for entity_id in starting_entities:
            entity_info = self.explorer.get_full_entity_info(entity_id)
            if entity_info:
                entity_id_full, name, full_description = entity_info
                initial_node = ExplorationNode(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_description=full_description,
                    depth=0,
                    score=1.0,  # Initial score for starting nodes
                    parent=None,
                    relation_from_parent=None,
                    relation_full_description=None,
                    entity_full_description=full_description,
                )
                state.add_node(initial_node)

        # Check depth-0: can starting entities alone answer the query?
        frontier_nodes = state.get_current_frontier()
        logger.info("[ToG][depth=0] frontier=%d", len(frontier_nodes))
        logger.debug("[ToG][depth=0] entities=%s", [n.entity_name for n in frontier_nodes])
        frontier_text_units = self.explorer.get_text_units_for_nodes(frontier_nodes)
        (
            should_terminate,
            answer,
            early_term_metrics,
        ) = await self.reasoning_module.check_early_termination(
            query, frontier_nodes, conversation_history_context=history_context,
            text_units=frontier_text_units,
        )
        if should_terminate and answer:
            reasoning_paths = self.reasoning_module.get_reasoning_paths(
                state.get_current_frontier()
            )
            logger.info(
                "[ToG][early_terminate][depth=0] triggered paths=%d",
                len(reasoning_paths),
            )
            early_context_text = self.reasoning_module.format_paths(
                state.get_current_frontier(), text_units=frontier_text_units
            )
            yield (answer, reasoning_paths, early_term_metrics, early_context_text)
            return
        yield ("", [], early_term_metrics, "")

        # Pre-compute query embedding once for the entire search session
        query_embedding: np.ndarray | None = None
        if isinstance(self.pruning_strategy, SemanticPruning) and self.embedding_model:
            logger.info("[ToG][embedding] precompute query embedding")
            query_embedding = np.array(
                await self.embedding_model.aembed(text=effective_query)
            )

        # Exploration loop
        while state.current_depth < state.max_depth:
            # Get current frontier
            current_nodes = state.get_current_frontier()

            logger.info(
                "[ToG][explore][depth=%d] frontier=%d",
                state.current_depth,
                len(current_nodes),
            )
            logger.debug(
                "[ToG][explore][depth=%d] nodes=%s",
                state.current_depth,
                [n.entity_name for n in current_nodes],
            )

            if not current_nodes:
                logger.info("[ToG][explore] stop: empty frontier at depth=%d", state.current_depth)
                break  # No more nodes to explore

            # Prepare for next depth
            next_depth = state.current_depth + 1

            # Parallelize per-node processing
            tasks = [
                self._process_node(query, node, query_embedding)
                for node in current_nodes
            ]
            task_results = await asyncio.gather(*tasks)

            # Collect all pruning metrics and next-level nodes
            next_level_nodes: list = []
            all_node_metrics: list = []
            for new_nodes, metrics_list in task_results:
                next_level_nodes.extend(new_nodes)
                all_node_metrics.extend(metrics_list)

            logger.info(
                "[ToG][expand][depth=%d->%d] candidates=%d metrics=%d",
                state.current_depth,
                next_depth,
                len(next_level_nodes),
                len(all_node_metrics),
            )

            # Yield all collected metrics
            for m in all_node_metrics:
                yield ("", [], m, "")

            # Add next level nodes to state
            state.nodes_by_depth[next_depth] = next_level_nodes

            # Prune to beam width
            state.current_depth = next_depth
            state.prune_current_frontier()
            pruned_frontier = state.get_current_frontier()
            logger.info(
                "[ToG][prune][depth=%d] kept=%d",
                state.current_depth,
                len(pruned_frontier),
            )
            logger.debug(
                "[ToG][prune][depth=%d] top_nodes=%s",
                state.current_depth,
                [n.entity_name for n in pruned_frontier],
            )

            # Debug: show exploration steps AFTER pruning (only kept paths)
            # Check for early termination
            frontier_nodes = state.get_current_frontier()
            frontier_text_units = self.explorer.get_text_units_for_nodes(frontier_nodes)
            (
                should_terminate,
                answer,
                early_term_metrics,
            ) = await self.reasoning_module.check_early_termination(
                query, frontier_nodes, conversation_history_context=history_context,
                text_units=frontier_text_units,
            )

            if should_terminate and answer:
                reasoning_paths = self.reasoning_module.get_reasoning_paths(
                    state.get_current_frontier()
                )
                logger.info(
                    "[ToG][early_terminate][depth=%d] triggered paths=%d",
                    state.current_depth,
                    len(reasoning_paths),
                )
                # Generate context_text for early termination
                early_context_text = self.reasoning_module.format_paths(
                    state.get_current_frontier(), text_units=frontier_text_units
                )
                yield (answer, reasoning_paths, early_term_metrics, early_context_text)
                return
            # Yield early termination metrics (non-terminating case)
            yield ("", [], early_term_metrics, "")

        # Generate final answer from explored paths
        all_paths = []
        for depth_nodes in state.nodes_by_depth.values():
            all_paths.extend(depth_nodes)

        if not all_paths:
            logger.info("[ToG][finish] no_paths_generated")
            yield (
                "No exploration paths were generated. The knowledge graph may not contain relevant information for this query.",
                [],
                None,
                "",
            )
            return

        # Generate rich context text with entity and relation descriptions
        all_text_units = self.explorer.get_text_units_for_nodes(all_paths)
        context_text = self.reasoning_module.format_paths(all_paths, text_units=all_text_units)

        logger.info(
            "[ToG][reasoning] total_paths=%d text_units=%d",
            len(all_paths),
            len(all_text_units),
        )

        # Use reasoning module to generate final answer
        try:
            (
                answer,
                reasoning_paths,
                answer_metrics,
            ) = await self.reasoning_module.generate_answer(
                query, all_paths,
                conversation_history_context=history_context,
                text_units=all_text_units,
            )

            logger.info(
                "[ToG][finish] success reasoning_paths=%d answer_chars=%d",
                len(reasoning_paths),
                len(answer),
            )
            # Yield answer metrics with context_text
            yield ("", reasoning_paths, answer_metrics, context_text)
            yield (answer, reasoning_paths, None, context_text)
        except Exception as e:
            logger.exception("[ToG][finish] reasoning_failed: %s", e)
            # Fallback response if reasoning fails
            paths_summary = "\n".join([
                f"- {node.entity_name}: {node.entity_description[:100]}..."
                for node in all_paths[:5]
            ])
            yield (
                f"""Error during reasoning: {str(e)}

However, I found these relevant entities during exploration:
{paths_summary}

Based on the exploration, I found {len(all_paths)} potential paths. Please try rephrasing your query or check if the entities are relevant to your question.""",
                [],
                None,
                "",
            )

    async def _process_node(
        self,
        query: str,
        node: ExplorationNode,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[List[ExplorationNode], List[PruningMetrics]]:
        """Process a single frontier node: score relations, score entities, build new nodes."""
        next_depth = node.depth + 1
        metrics_list: List[PruningMetrics] = []

        relations = self.explorer.get_relations(node.entity_id)
        if not relations:
            logger.debug("[ToG][node=%s][depth=%d] no_relations", node.entity_name, node.depth)
            return [], []

        logger.debug(
            "[ToG][node=%s][depth=%d] relations=%d",
            node.entity_name,
            node.depth,
            len(relations),
        )

        # Score relations
        scored_relations, pruning_metrics = await self.pruning_strategy.score_relations(
            query, node.entity_name, relations, query_embedding=query_embedding
        )
        metrics_list.append(pruning_metrics)

        # Keep relation candidates ordered by relevance before entity retention.
        scored_relations.sort(key=lambda x: x[4], reverse=True)

        # Build entity candidates
        candidate_data = []
        for rel_desc, target_id, direction, weight, rel_score in scored_relations:
            target_info = self.explorer.get_full_entity_info(target_id)
            rel_info = self.explorer.get_full_relation_info(node.entity_id, target_id, rel_desc)
            if target_info:
                _, target_name, target_full_desc = target_info
                rel_full_desc = rel_info[1] if rel_info else rel_desc
                candidate_data.append(
                    (rel_desc, target_id, direction, weight, rel_score,
                     target_name, target_full_desc, rel_full_desc)
                )

        if not candidate_data:
            return [], metrics_list

        logger.debug(
            "[ToG][node=%s][depth=%d] scored_relations=%d candidate_entities=%d",
            node.entity_name,
            node.depth,
            len(scored_relations),
            len(candidate_data),
        )

        if len(candidate_data) > self.num_retain_entity:
            logger.debug(
                "[ToG][node=%s][depth=%d] sample_candidates %d->%d",
                node.entity_name,
                node.depth,
                len(candidate_data),
                self.num_retain_entity,
            )
            candidate_data = random.sample(candidate_data, self.num_retain_entity)

        entity_candidates = [
            (target_id, target_name, target_full_desc)
            for (_, target_id, _, _, _, target_name, target_full_desc, _) in candidate_data
        ]
        current_path = self._node_to_path_string(node)

        entity_scores, entity_metrics = await self.pruning_strategy.score_entities(
            query=query,
            current_path=current_path,
            entities=entity_candidates,
            query_embedding=query_embedding,
        )
        metrics_list.append(entity_metrics)

        # Create new exploration nodes
        new_nodes: List[ExplorationNode] = []
        for idx, (
            rel_desc, target_id, _direction, _weight, rel_score,
            target_name, target_full_desc, rel_full_desc,
        ) in enumerate(candidate_data):
            entity_score = entity_scores[idx] if idx < len(entity_scores) else 5.0
            hop_score = rel_score * (max(entity_score, 0.0) / 10.0)
            combined_score = node.score * hop_score
            new_nodes.append(ExplorationNode(
                entity_id=target_id,
                entity_name=target_name,
                entity_description=target_full_desc,
                depth=next_depth,
                score=combined_score,
                parent=node,
                relation_from_parent=rel_desc,
                relation_full_description=rel_full_desc,
                entity_full_description=target_full_desc,
            ))

        return new_nodes, metrics_list

    def _node_to_path_string(self, node: ExplorationNode) -> str:
        """Build a readable chain string from root to current node."""
        chain: List[str] = []
        current = node
        while current.parent is not None:
            relation = current.relation_from_parent or "related_to"
            chain.append(
                f"{current.parent.entity_name} --[{relation}]--> {current.entity_name}"
            )
            current = current.parent

        if not chain:
            return node.entity_name

        chain.reverse()
        return " | ".join(chain)
