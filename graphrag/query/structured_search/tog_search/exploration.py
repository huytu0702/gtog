from typing import List, Tuple
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from .state import ExplorationNode, ToGSearchState

class GraphExplorer:
    """Handles graph traversal and relation/entity retrieval."""

    def __init__(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
    ):
        self.entities = {e.id: e for e in entities}
        self.relationships = relationships
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency lists for efficient graph traversal."""
        self.outgoing = {}  # entity_id -> [(relation, target_entity_id)]
        self.incoming = {}  # entity_id -> [(relation, source_entity_id)]

        for rel in self.relationships:
            # Outgoing edges
            if rel.source not in self.outgoing:
                self.outgoing[rel.source] = []
            self.outgoing[rel.source].append((rel.description, rel.target, rel.weight))

            # Incoming edges
            if rel.target not in self.incoming:
                self.incoming[rel.target] = []
            self.incoming[rel.target].append((rel.description, rel.source, rel.weight))

    def get_relations(self, entity_id: str, bidirectional: bool = True) -> List[Tuple[str, str, str, float]]:
        """
        Get all relations for an entity.
        Returns: List of (relation_description, target_entity_id, direction, weight)
        """
        relations = []

        # Outgoing relations
        for rel_desc, target_id, weight in self.outgoing.get(entity_id, []):
            relations.append((rel_desc, target_id, "outgoing", weight))

        # Incoming relations (if bidirectional)
        if bidirectional:
            for rel_desc, source_id, weight in self.incoming.get(entity_id, []):
                relations.append((rel_desc, source_id, "incoming", weight))

        return relations

    def get_entity_info(self, entity_id: str) -> Tuple[str, str] | None:
        """Get entity name and description."""
        entity = self.entities.get(entity_id)
        if entity:
            return (entity.title, entity.description or "")
        return None

    def find_starting_entities(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find starting entities for exploration based on query.
        Uses simple keyword matching - can be enhanced with embeddings.
        """
        query_lower = query.lower()
        candidates = []

        for entity_id, entity in self.entities.items():
            title_lower = entity.title.lower()
            desc_lower = (entity.description or "").lower()

            # Simple scoring based on keyword presence
            score = 0.0
            if query_lower in title_lower:
                score += 10.0
            if query_lower in desc_lower:
                score += 5.0

            # Token overlap scoring
            query_tokens = set(query_lower.split())
            title_tokens = set(title_lower.split())
            desc_tokens = set(desc_lower.split())

            score += len(query_tokens & title_tokens) * 2.0
            score += len(query_tokens & desc_tokens) * 1.0

            if score > 0:
                candidates.append((entity_id, score))

        # Return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in candidates[:top_k]]