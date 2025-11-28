from typing import AsyncGenerator, List
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.protocol.base import ChatModel
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.tokenizer import Tokenizer
from .state import ToGSearchState, ExplorationNode
from .exploration import GraphExplorer
from .pruning import PruningStrategy, LLMPruning, SemanticPruning
from .reasoning import ToGReasoning

class ToGSearch:
    """
    ToG (Think-on-Graph) Search Engine for GraphRAG.

    Implements iterative graph exploration with LLM-guided pruning
    and reasoning over discovered paths.
    """

    def __init__(
        self,
        model: ChatModel,
        entities: List[Entity],
        relationships: List[Relationship],
        tokenizer: Tokenizer,
        pruning_strategy: PruningStrategy,
        reasoning_module: ToGReasoning,
        width: int = 3,
        depth: int = 3,
        num_retain_entity: int = 5,
        callbacks: List[QueryCallbacks] | None = None,
    ):
        self.model = model
        self.explorer = GraphExplorer(entities, relationships)
        self.tokenizer = tokenizer
        self.pruning_strategy = pruning_strategy
        self.reasoning_module = reasoning_module
        self.width = width
        self.depth = depth
        self.num_retain_entity = num_retain_entity
        self.callbacks = callbacks or []

    async def search(self, query: str) -> str:
        """Perform ToG search and return answer."""
        result = ""
        async for chunk in self.stream_search(query):
            result += chunk
        return result

    async def stream_search(self, query: str) -> AsyncGenerator[str, None]:
        """Perform ToG search with streaming output."""
        # Find initial entities
        starting_entities = self.explorer.find_starting_entities(query, top_k=self.width)
        
        # Initialize search state
        state = ToGSearchState(
            query=query,
            current_depth=0,
            nodes_by_depth={0: []},
            finished_paths=[],
            max_depth=self.depth,
            beam_width=self.width
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
                relation_from_parent=None
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
                scored_relations = await self.pruning_strategy.score_relations(
                    query,
                    node.entity_name,
                    relations
                )

                # Keep top entities based on scores
                scored_relations.sort(key=lambda x: x[4], reverse=True)  # Sort by score
                top_relations = scored_relations[:self.num_retain_entity]

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
                            relation_from_parent=rel_desc
                        )
                        next_level_nodes.append(new_node)

            # Add next level nodes to state
            state.nodes_by_depth[next_depth] = next_level_nodes
            
            # Prune to beam width
            state.current_depth = next_depth
            state.prune_current_frontier()

            # Check for early termination
            should_terminate, answer = await self.reasoning_module.check_early_termination(
                query, state.get_current_frontier()
            )
            
            if should_terminate and answer:
                yield answer
                return

        # Generate final answer from explored paths
        all_paths = []
        for depth_nodes in state.nodes_by_depth.values():
            all_paths.extend(depth_nodes)
        
        # Use reasoning module to generate final answer
        answer, reasoning_paths = await self.reasoning_module.generate_answer(
            query, all_paths
        )
        
        yield answer