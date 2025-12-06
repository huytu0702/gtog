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
        print(f"[DEBUG] GraphExplorer loaded {len(entities)} entities")
        print(f"[DEBUG] Entity IDs: {list(self.entities.keys())[:10]}")
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

    def get_relations(
        self, entity_id: str, bidirectional: bool = True
    ) -> List[Tuple[str, str, str, float]]:
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
        Uses improved keyword matching with fuzzy scoring.
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        candidates = []

        for entity_id, entity in self.entities.items():
            title_lower = entity.title.lower()
            desc_lower = (entity.description or "").lower()
            title_tokens = set(title_lower.split())
            desc_tokens = set(desc_lower.split())

            # Enhanced scoring system
            score = 0.0

            # Exact match bonus
            if query_lower in title_lower:
                score += 20.0  # Higher bonus for exact title match
            if query_lower in desc_lower:
                score += 10.0  # Bonus for exact description match

            # Token overlap scoring (more lenient)
            title_overlap = len(query_tokens & title_tokens)
            desc_overlap = len(query_tokens & desc_tokens)
            score += title_overlap * 4.0  # Increased weight for title overlap
            score += desc_overlap * 2.0  # Weight for description overlap

            # Partial word matching (even more lenient)
            for token in query_tokens:
                if len(token) > 2:  # Only consider longer tokens
                    if token in title_lower:
                        score += 2.0
                    elif token in desc_lower:
                        score += 1.0
                else:  # Single character tokens
                    if token in title_lower:
                        score += 1.0
                    elif token in desc_lower:
                        score += 0.5

            # Length penalty for very short matches
            if len(title_lower) < len(query_lower) * 0.3:
                score *= 0.8  # Penalty for very short titles

            if score > 0.5:  # Lower threshold for inclusion
                candidates.append((entity_id, score))

        # Enhanced fallback strategy
        if not candidates and self.entities:
            # Try to find entities with any query token match
            fallback_candidates = []
            for entity_id, entity in self.entities.items():
                title_lower = entity.title.lower()
                desc_lower = (entity.description or "").lower()

                # Very lenient matching for fallback
                for token in query_tokens:
                    if len(token) > 1 and (token in title_lower or token in desc_lower):
                        fallback_score = 2.0  # Minimal score for fallback
                        fallback_candidates.append((entity_id, fallback_score))
                        break

            if fallback_candidates:
                candidates = fallback_candidates
            else:
                # Last resort: return entities with most common words
                common_words = {"search", "method", "technique", "approach", "system"}
                for entity_id, entity in self.entities.items():
                    title_lower = entity.title.lower()
                    for word in common_words:
                        if word in title_lower:
                            candidates.append((entity_id, 1.0))
                            break
                    if len(candidates) >= top_k:
                        break

        # Return top-k (or all if less than k)
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            result = [eid for eid, _ in candidates[:top_k]]
            print(
                f"[DEBUG] Found {len(candidates)} candidates, returning top {len(result)}: {result}"
            )
            return result
        elif self.entities:
            # Absolute fallback: return first entities
            entity_ids = list(self.entities.keys())[:top_k]
            print(f"[DEBUG] No candidates, using fallback: {entity_ids}")
            return entity_ids
        else:
            print("[DEBUG] No entities available")
            return []
