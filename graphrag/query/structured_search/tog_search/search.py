import time
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple, Union
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.tokenizer import Tokenizer
from graphrag.vector_stores.base import BaseVectorStore
from graphrag.query.structured_search.base import SearchResult
from .state import ToGSearchState, ExplorationNode
from .exploration import GraphExplorer
from .pruning import PruningStrategy, LLMPruning, SemanticPruning, PruningMetrics
from .reasoning import ToGReasoning, ReasoningMetrics


@dataclass
class ToGMetrics:
    """Aggregated metrics for ToG search."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    exploration_llm_calls: int = 0
    reasoning_llm_calls: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    def add_pruning(self, m: PruningMetrics) -> None:
        """Add pruning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.exploration_llm_calls += m.llm_calls
        self.embedding_calls += m.embedding_calls
        self.embedding_tokens += m.embedding_tokens

    def add_reasoning(self, m: ReasoningMetrics) -> None:
        """Add reasoning metrics."""
        self.llm_calls += m.llm_calls
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        self.reasoning_llm_calls += m.llm_calls


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

    async def search(self, query: str) -> SearchResult:
        """Perform ToG search and return SearchResult with metrics."""
        start_time = time.time()
        metrics = ToGMetrics()

        response_chunks: List[str] = []
        context_paths: List[str] = []

        async for chunk, paths, chunk_metrics in self._stream_search_with_metrics(query):
            if chunk:
                response_chunks.append(chunk)
            if paths:
                context_paths = paths
            if chunk_metrics:
                if isinstance(chunk_metrics, PruningMetrics):
                    metrics.add_pruning(chunk_metrics)
                elif isinstance(chunk_metrics, ReasoningMetrics):
                    metrics.add_reasoning(chunk_metrics)

        response = "".join(response_chunks)
        completion_time = time.time() - start_time

        context_text = "\n".join(context_paths) if context_paths else ""

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
                "exploration": metrics.prompt_tokens - (metrics.prompt_tokens - metrics.embedding_tokens) if metrics.embedding_tokens else 0,
                "reasoning": metrics.prompt_tokens,
            },
            output_tokens_categories={
                "exploration": metrics.output_tokens,
                "reasoning": metrics.output_tokens,
            },
        )

    async def stream_search(self, query: str) -> AsyncGenerator[str, None]:
        """Perform ToG search with streaming output (backward compatible)."""
        async for chunk, _, _ in self._stream_search_with_metrics(query):
            if chunk:  # Only yield non-empty chunks
                yield chunk

    async def _stream_search_with_metrics(
        self, query: str
    ) -> AsyncGenerator[
        Tuple[str, List[str], Union[PruningMetrics, ReasoningMetrics, None]], None
    ]:
        """Perform ToG search with streaming output."""
        # Find initial entities using semantic similarity (like ToG paper)
        if self.embedding_model:
            starting_entities = await self.explorer.find_starting_entities_semantic(
                query, top_k=self.width
            )
        else:
            starting_entities = self.explorer.find_starting_entities(
                query, top_k=self.width
            )

        if not starting_entities:
            available_entities = list(self.explorer.entities.keys())[:10]
            yield (f"No relevant entities found for query '{query}'. Available entities: {available_entities}", [], None)
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
            name, description = self.explorer.get_entity_info(entity_id)
            initial_node = ExplorationNode(
                entity_id=entity_id,
                entity_name=name,
                entity_description=description,
                depth=0,
                score=1.0,  # Initial score for starting nodes
                parent=None,
                relation_from_parent=None,
            )
            state.add_node(initial_node)

        # Exploration loop
        while state.current_depth < state.max_depth:
            # Get current frontier
            current_nodes = state.get_current_frontier()

            if not current_nodes:
                break  # No more nodes to explore

            # Prepare for next depth
            next_depth = state.current_depth + 1
            next_level_nodes = []

            # Explore each node in current frontier
            for node in current_nodes:
                # Get relations for current entity
                relations = self.explorer.get_relations(node.entity_id)

                if not relations:
                    continue  # No relations to explore from this node

                # Score relations
                scored_relations, pruning_metrics = await self.pruning_strategy.score_relations(
                    query, node.entity_name, relations
                )

                # Yield pruning metrics
                yield ("", [], pruning_metrics)

                # Keep top entities based on scores
                scored_relations.sort(key=lambda x: x[4], reverse=True)  # Sort by score
                top_relations = scored_relations[: self.num_retain_entity]

                # Create new exploration nodes
                for rel_desc, target_id, direction, weight, score in top_relations:
                    target_name, target_desc = self.explorer.get_entity_info(target_id)
                    if target_name:
                        new_node = ExplorationNode(
                            entity_id=target_id,
                            entity_name=target_name,
                            entity_description=target_desc,
                            depth=next_depth,
                            score=score,
                            parent=node,
                            relation_from_parent=rel_desc,
                        )
                        next_level_nodes.append(new_node)

            # Add next level nodes to state
            state.nodes_by_depth[next_depth] = next_level_nodes

            # Prune to beam width
            state.current_depth = next_depth
            state.prune_current_frontier()

            # Debug: show exploration steps AFTER pruning (only kept paths)
            # Disabled to only show final answer
            if False:  # hasattr(self, "_debug") and self._debug:
                kept_nodes = state.get_current_frontier()
                for node in kept_nodes:
                    if node.parent:
                        yield (f"[DEPTH {next_depth}] {node.parent.entity_name} --[{node.relation_from_parent}]--> {node.entity_name} (score: {node.score:.2f})\n", [], None)

            # Check for early termination
            (
                should_terminate,
                answer,
                early_term_metrics,
            ) = await self.reasoning_module.check_early_termination(
                query, state.get_current_frontier()
            )

            # Yield early termination metrics
            yield ("", [], early_term_metrics)

            if should_terminate and answer:
                # Disabled debug output for early termination
                if False:
                    yield (f"=== ToG EARLY TERMINATION ===\n", [], None)
                    yield (f"Terminated at depth {state.current_depth} with {len(state.get_current_frontier())} paths.\n\n", [], None)
                    yield (f"=== ToG REASONING ANSWER ===\n\n", [], None)
                yield (answer, [], None)
                return

        # Generate final answer from explored paths
        all_paths = []
        for depth_nodes in state.nodes_by_depth.values():
            all_paths.extend(depth_nodes)

        if not all_paths:
            yield ("No exploration paths were generated. The knowledge graph may not contain relevant information for this query.", [], None)
            return

        # Use reasoning module to generate final answer
        try:
            answer, reasoning_paths, answer_metrics = await self.reasoning_module.generate_answer(
                query, all_paths
            )

            # Yield answer metrics
            yield ("", reasoning_paths, answer_metrics)

            # Show exploration paths before answer
            # Disabled to only show final answer
            if False:
                yield (f"=== ToG EXPLORATION ANALYSIS ===\n", [], None)
                yield (f"Query: {query}\n", [], None)
                yield (f"Max Depth: {self.depth}, Beam Width: {self.width}\n", [], None)
                yield (f"Total exploration paths found: {len(all_paths)}\n", [], None)
                yield (f"Unique entities explored: {len(set(node.entity_id for node in all_paths))}\n\n", [], None)

                yield (f"=== EXPLORATION PATHS (with scores) ===\n", [], None)
                for i, path in enumerate(reasoning_paths, 1):
                    yield (f"Path {i}: {path}\n", [], None)

                # Show path details with scores
                yield (f"\n=== PATH DETAILS ===\n", [], None)
                for depth in range(self.depth + 1):
                    depth_nodes = [node for node in all_paths if node.depth == depth]
                    if depth_nodes:
                        yield (f"Depth {depth}:\n", [], None)
                        for node in depth_nodes:
                            parent_info = (
                                f" (from: {node.parent.entity_name})"
                                if node.parent
                                else ""
                            )
                            yield (f"  - {node.entity_name} [score: {node.score:.2f}]{parent_info}\n", [], None)
                        yield ("\n", [], None)

                yield (f"=== ToG REASONING ANSWER ===\n\n", [], None)
            yield (answer, reasoning_paths, None)
        except Exception as e:
            # Fallback response if reasoning fails
            paths_summary = "\n".join([
                f"- {node.entity_name}: {node.entity_description[:100]}..."
                for node in all_paths[:5]
            ])
            yield (f"""Error during reasoning: {str(e)}

However, I found these relevant entities during exploration:
{paths_summary}

Based on the exploration, I found {len(all_paths)} potential paths. Please try rephrasing your query or check if the entities are relevant to your question.""", [], None)
