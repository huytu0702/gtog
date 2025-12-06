"""Graph exploration module for ToG search.

Handles graph traversal, entity linking (with embedding similarity like ToG paper),
and relation/entity retrieval.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.language_model.protocol.base import EmbeddingModel
from .state import ExplorationNode, ToGSearchState

logger = logging.getLogger(__name__)


class GraphExplorer:
    """Handles graph traversal and relation/entity retrieval.
    
    Uses embedding similarity for entity linking (like original ToG paper).
    """

    def __init__(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize the GraphExplorer.
        
        Args:
            entities: List of entities in the knowledge graph
            relationships: List of relationships between entities
            embedding_model: Optional embedding model for semantic entity linking
        """
        self.entities = {e.id: e for e in entities}
        self.entity_list = entities  # Keep original list for embedding
        self.relationships = relationships
        self.embedding_model = embedding_model
        
        # Cache for entity embeddings
        self._entity_embeddings: Optional[np.ndarray] = None
        self._entity_texts: Optional[List[str]] = None
        
        logger.debug(f"GraphExplorer loaded {len(entities)} entities")
        logger.debug(f"Entity IDs (first 10): {list(self.entities.keys())[:10]}")
        logger.debug(f"Embedding model available: {embedding_model is not None}")
        
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

    async def _compute_entity_embeddings(self) -> None:
        """Compute and cache embeddings for all entities."""
        if self._entity_embeddings is not None:
            return  # Already computed
        
        if not self.embedding_model or not self.entity_list:
            return
        
        # Create text representations for each entity
        self._entity_texts = [
            f"{e.title}: {e.description or ''}"[:500]  # Limit length
            for e in self.entity_list
        ]
        
        try:
            # Batch embed all entities
            embeddings = await self.embedding_model.aembed_batch(
                text_list=self._entity_texts
            )
            self._entity_embeddings = np.array(embeddings)
            logger.debug(f"Computed embeddings for {len(self._entity_texts)} entities")
        except Exception as e:
            logger.warning(f"Failed to compute entity embeddings: {e}")
            self._entity_embeddings = None

    async def find_starting_entities_semantic(
        self, query: str, top_k: int = 3
    ) -> List[str]:
        """
        Find starting entities using embedding similarity (like ToG paper).
        
        Uses SentenceTransformer-style dot product scoring.
        """
        if not self.embedding_model:
            logger.debug("No embedding model, falling back to keyword matching")
            return self.find_starting_entities_keyword(query, top_k)
        
        # Ensure entity embeddings are computed
        await self._compute_entity_embeddings()
        
        if self._entity_embeddings is None:
            logger.debug("Entity embeddings not available, falling back to keyword")
            return self.find_starting_entities_keyword(query, top_k)
        
        try:
            # Embed the query
            query_embedding = await self.embedding_model.aembed(text=query)
            query_emb = np.array(query_embedding)
            
            # Compute dot product scores (like ToG paper uses util.dot_score)
            scores = np.dot(self._entity_embeddings, query_emb)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Map back to entity IDs
            result = [self.entity_list[i].id for i in top_indices]
            
            logger.debug(
                f"Semantic entity linking: top {top_k} entities with scores "
                f"{[f'{self.entity_list[i].title}: {scores[i]:.3f}' for i in top_indices]}"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Semantic entity linking failed: {e}, falling back to keyword")
            return self.find_starting_entities_keyword(query, top_k)

    def find_starting_entities_keyword(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find starting entities using keyword matching (fallback method).
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
                score += 20.0
            if query_lower in desc_lower:
                score += 10.0

            # Token overlap scoring
            title_overlap = len(query_tokens & title_tokens)
            desc_overlap = len(query_tokens & desc_tokens)
            score += title_overlap * 4.0
            score += desc_overlap * 2.0

            # Partial word matching
            for token in query_tokens:
                if len(token) > 2:
                    if token in title_lower:
                        score += 2.0
                    elif token in desc_lower:
                        score += 1.0

            if score > 0.5:
                candidates.append((entity_id, score))

        # Fallback strategy
        if not candidates and self.entities:
            for entity_id, entity in self.entities.items():
                title_lower = entity.title.lower()
                desc_lower = (entity.description or "").lower()

                for token in query_tokens:
                    if len(token) > 1 and (token in title_lower or token in desc_lower):
                        candidates.append((entity_id, 2.0))
                        break

        # Return top-k
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            result = [eid for eid, _ in candidates[:top_k]]
            logger.debug(f"Keyword matching: found {len(candidates)} candidates")
            return result
        elif self.entities:
            entity_ids = list(self.entities.keys())[:top_k]
            logger.debug(f"No candidates, using fallback: {entity_ids}")
            return entity_ids
        else:
            logger.debug("No entities available")
            return []

    def find_starting_entities(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find starting entities (sync version, uses keyword matching).
        
        For semantic matching, use find_starting_entities_semantic() instead.
        """
        return self.find_starting_entities_keyword(query, top_k)
