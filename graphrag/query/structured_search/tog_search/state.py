from dataclasses import dataclass
from typing import List, Dict, Any, Union


@dataclass
class ExplorationNode:
    """Represents a node in the exploration tree."""

    entity_id: str
    entity_name: str
    entity_description: str
    depth: int
    score: float
    parent: Union["ExplorationNode", None]
    relation_from_parent: Union[str, None]
    relation_full_description: Union[str, None] = None  # Full relationship description
    entity_full_description: Union[str, None] = (
        None  # Full entity description from Entity object
    )

    def get_path(self) -> List[tuple[str, str]]:
        """Returns the path from root to this node as (entity, relation) pairs."""
        path = []
        current = self
        while current.parent is not None:
            path.append((current.entity_name, current.relation_from_parent))
            current = current.parent
        return list(reversed(path))


@dataclass
class ToGSearchState:
    """Maintains the state of ToG search exploration."""

    query: str
    current_depth: int
    nodes_by_depth: Dict[int, List[ExplorationNode]]
    finished_paths: List[ExplorationNode]
    max_depth: int
    beam_width: int

    def add_node(self, node: ExplorationNode):
        """Add a node to the current depth."""
        if node.depth not in self.nodes_by_depth:
            self.nodes_by_depth[node.depth] = []
        self.nodes_by_depth[node.depth].append(node)

    def get_current_frontier(self) -> List[ExplorationNode]:
        """Get nodes at the current exploration depth."""
        return self.nodes_by_depth.get(self.current_depth, [])

    def prune_current_frontier(self):
        """Keep only top-k nodes at current depth."""
        frontier = self.get_current_frontier()
        frontier.sort(key=lambda n: n.score, reverse=True)
        self.nodes_by_depth[self.current_depth] = frontier[: self.beam_width]
