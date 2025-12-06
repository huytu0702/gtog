from typing import List, Tuple
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
import numpy as np
from graphrag.prompts.query.tog_relation_scoring_prompt import (
    TOG_RELATION_SCORING_PROMPT,
)
from graphrag.prompts.query.tog_entity_scoring_prompt import TOG_ENTITY_SCORING_PROMPT


class PruningStrategy:
    """Base class for pruning strategies."""

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """
        Score relations for relevance to query.
        Returns: List of (relation, target_id, direction, weight, relevance_score)
        """
        raise NotImplementedError

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> List[float]:
        """
        Score entities for relevance.
        entities: List of (entity_id, entity_name, entity_description)
        Returns: List of scores (same order as input)
        """
        raise NotImplementedError


class LLMPruning(PruningStrategy):
    """Uses LLM to score relations and entities."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.4,
        relation_scoring_prompt: str | None = None,
        entity_scoring_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.relation_scoring_prompt = (
            relation_scoring_prompt or TOG_RELATION_SCORING_PROMPT
        )
        self.entity_scoring_prompt = entity_scoring_prompt or TOG_ENTITY_SCORING_PROMPT

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """Score relations using LLM."""

        if not relations:
            return []

        # Build relations text
        relations_text = "\n".join([
            f"{i + 1}. [{direction}] {rel_desc} (weight: {weight:.2f})"
            for i, (rel_desc, _, direction, weight) in enumerate(relations)
        ])

        prompt = self.relation_scoring_prompt.format(
            query=query, entity_name=entity_name, relations=relations_text
        )

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        # Parse scores
        scores = self._parse_scores(response, len(relations))

        # Combine with relation data
        return [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> List[float]:
        """Score entities using LLM."""

        if not entities:
            return []

        entities_text = "\n".join([
            f"{i + 1}. {name}: {desc[:100]}..."
            for i, (_, name, desc) in enumerate(entities)
        ])

        prompt = self.entity_scoring_prompt.format(
            query=query, current_path=current_path, candidate_entities=entities_text
        )

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        return self._parse_scores(response, len(entities))

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse score list from LLM response."""
        import re

        # Clean response and try to extract list pattern first
        response = response.strip()

        # Try to match list pattern: [1, 2, 3] or 1, 2, 3
        list_match = re.search(r"\[([\d\s,\.]+)\]", response)
        if list_match:
            numbers_str = list_match.group(1)
        else:
            # Look for comma-separated numbers
            numbers_str = response

        # Extract numbers
        numbers = re.findall(r"\d+\.?\d*", numbers_str)
        scores = []

        for num_str in numbers[:expected_count]:
            try:
                score = float(num_str)
                # Clamp to 1-10 range
                score = max(1.0, min(10.0, score))
                scores.append(score)
            except ValueError:
                scores.append(5.0)  # Default score

        # If not enough scores, pad with uniform distribution
        while len(scores) < expected_count:
            scores.append(5.0)

        return scores[:expected_count]


class SemanticPruning(PruningStrategy):
    """Uses embedding similarity for pruning."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """Score relations using embedding similarity."""

        if not relations:
            return []

        # Embed query
        query_emb = np.array(await self.embedding_model.aembed(text=query))

        # Embed relation descriptions
        relation_texts = [rel_desc for rel_desc, _, _, _ in relations]
        relation_embeddings = await self.embedding_model.aembed_batch(
            text_list=relation_texts
        )

        # Compute cosine similarities
        scores = []
        for rel_emb in relation_embeddings:
            rel_emb = np.array(rel_emb)
            similarity = np.dot(query_emb, rel_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(rel_emb)
            )
            # Scale to 1-10 range
            score = (similarity + 1) * 5  # cosine range [-1, 1] -> [0, 10]
            scores.append(score)

        return [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> List[float]:
        """Score entities using embedding similarity."""

        if not entities:
            return []

        # Embed query
        query_embedding = await self.embedding_model.async_generate(inputs=[query])
        query_emb = np.array(query_embedding[0])

        # Embed entity descriptions
        entity_texts = [f"{name}: {desc}" for _, name, desc in entities]
        entity_embeddings = await self.embedding_model.async_generate(
            inputs=entity_texts
        )

        # Compute similarities
        scores = []
        for ent_emb in entity_embeddings:
            ent_emb = np.array(ent_emb)
            similarity = np.dot(query_emb, ent_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(ent_emb)
            )
            score = (similarity + 1) * 5
            scores.append(score)

        return scores


class BM25Pruning(PruningStrategy):
    """Uses BM25 algorithm for lexical matching (from ToG paper)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 pruning.
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        return re.findall(r'\w+', text.lower())

    def _compute_bm25_scores(
        self, query: str, documents: List[str]
    ) -> List[float]:
        """Compute BM25 scores for documents given a query."""
        if not documents:
            return []
        
        # Tokenize
        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(doc) for doc in documents]
        
        # Compute document frequencies
        doc_count = len(documents)
        df = {}  # document frequency for each term
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
        
        # Average document length
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(doc_count, 1)
        
        # Compute BM25 score for each document
        scores = []
        for doc_tokens in doc_tokens_list:
            score = 0.0
            dl = len(doc_tokens)
            term_freq = {}
            for token in doc_tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            for token in query_tokens:
                if token not in term_freq:
                    continue
                
                tf = term_freq[token]
                doc_freq = df.get(token, 0)
                
                # IDF component
                idf = np.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                
                # TF component with saturation
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1))
                )
                
                score += idf * tf_component
            
            scores.append(score)
        
        return scores

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """Score relations using BM25."""

        if not relations:
            return []

        # Create searchable text for each relation
        relation_texts = [
            f"{entity_name} {direction} {rel_desc}"
            for rel_desc, _, direction, _ in relations
        ]

        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query, relation_texts)
        
        # Normalize scores to 1-10 range
        max_score = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        normalized_scores = [
            max(1.0, min(10.0, (s / max_score) * 9 + 1))
            for s in bm25_scores
        ]

        return [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, normalized_scores
            )
        ]

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> List[float]:
        """Score entities using BM25."""

        if not entities:
            return []

        # Create searchable text for each entity
        entity_texts = [
            f"{name} {desc}" for _, name, desc in entities
        ]

        # Combine query with current path context
        search_text = f"{query} {current_path}"
        
        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(search_text, entity_texts)
        
        # Normalize scores to 1-10 range
        max_score = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        normalized_scores = [
            max(1.0, min(10.0, (s / max_score) * 9 + 1))
            for s in bm25_scores
        ]

        return normalized_scores
