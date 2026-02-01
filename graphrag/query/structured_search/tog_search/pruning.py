from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.vector_stores.base import BaseVectorStore
import numpy as np
import logging
from graphrag.prompts.query.tog_relation_scoring_prompt import (
    TOG_RELATION_SCORING_PROMPT,
)
from graphrag.prompts.query.tog_entity_scoring_prompt import TOG_ENTITY_SCORING_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PruningMetrics:
    """Metrics collected during pruning operations."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    def __add__(self, other: "PruningMetrics") -> "PruningMetrics":
        """Combine two PruningMetrics instances."""
        return PruningMetrics(
            llm_calls=self.llm_calls + other.llm_calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
        )


class PruningStrategy:
    """Base class for pruning strategies."""

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """
        Score relations for relevance to query.
        Returns: Tuple of (List of (relation, target_id, direction, weight, relevance_score), PruningMetrics)
        """
        raise NotImplementedError

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """
        Score entities for relevance.
        entities: List of (entity_id, entity_name, entity_description)
        Returns: Tuple of (List of scores (same order as input), PruningMetrics)
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
        
        # Load prompts - if file path is given, read the file content
        self.relation_scoring_prompt = self._load_prompt(
            relation_scoring_prompt, TOG_RELATION_SCORING_PROMPT
        )
        self.entity_scoring_prompt = self._load_prompt(
            entity_scoring_prompt, TOG_ENTITY_SCORING_PROMPT
        )

    def _load_prompt(self, prompt_or_path: str | None, default_prompt: str) -> str:
        """Load prompt from file if path is given, otherwise use directly."""
        import os
        
        if prompt_or_path is None:
            return default_prompt
        
        # Check if it's a file path (ends with .txt or .md)
        if prompt_or_path.endswith(('.txt', '.md')):
            if os.path.exists(prompt_or_path):
                try:
                    with open(prompt_or_path, 'r', encoding='utf-8') as f:
                        logger.debug(f"Loaded prompt from file: {prompt_or_path}")
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read prompt file {prompt_or_path}: {e}")
                    return default_prompt
            else:
                logger.warning(f"Prompt file not found: {prompt_or_path}, using default")
                return default_prompt
        
        # Otherwise, use the string directly as prompt
        return prompt_or_path

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using LLM."""
        metrics = PruningMetrics()

        if not relations:
            return [], metrics

        # Build relations text
        relations_text = "\n".join([
            f"{i + 1}. [{direction}] {rel_desc[:100]}... (weight: {weight:.2f})"
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

        # Update metrics
        metrics.llm_calls = 1
        # Estimate tokens (rough approximation: 4 chars per token)
        metrics.prompt_tokens = len(prompt) // 4
        metrics.output_tokens = len(response) // 4

        # Parse scores
        scores = self._parse_scores(response, len(relations))

        # Combine with relation data
        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]
        return scored_relations, metrics

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """Score entities using LLM."""
        metrics = PruningMetrics()

        if not entities:
            return [], metrics

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

        # Update metrics
        metrics.llm_calls = 1
        # Estimate tokens (rough approximation: 4 chars per token)
        metrics.prompt_tokens = len(prompt) // 4
        metrics.output_tokens = len(response) // 4

        return self._parse_scores(response, len(entities)), metrics

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

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        entity_embedding_store: Optional[BaseVectorStore] = None,
    ):
        self.embedding_model = embedding_model
        self.entity_embedding_store = entity_embedding_store
        self._entity_embeddings: Optional[np.ndarray] = None
        self._entity_texts: Optional[List[str]] = None

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using embedding similarity."""
        metrics = PruningMetrics()

        if not relations:
            return [], metrics

        # Embed query
        query_emb = np.array(await self.embedding_model.aembed(text=query))

        # Embed relation descriptions
        relation_texts = [rel_desc for rel_desc, _, _, _ in relations]
        relation_embeddings = await self.embedding_model.aembed_batch(
            text_list=relation_texts
        )

        # Update metrics for embedding calls
        metrics.embedding_calls = 2  # One for query, one batch for relations
        metrics.embedding_tokens = len(query) // 4 + sum(len(t) // 4 for t in relation_texts)

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

        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, scores
            )
        ]
        return scored_relations, metrics

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """Score entities using embedding similarity."""
        metrics = PruningMetrics()

        if not entities:
            return [], metrics

        # Load entity embeddings (pre-computed or computed)
        await self._load_entity_embeddings(entities)

        # Embed query
        query_emb = np.array(await self.embedding_model.aembed(text=query))

        # Update metrics for embedding call
        metrics.embedding_calls = 1
        metrics.embedding_tokens = len(query) // 4

        # Compute similarities using pre-computed embeddings
        scores = []
        for i, ent_emb in enumerate(self._entity_embeddings):
            similarity = np.dot(query_emb, ent_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(ent_emb)
            )
            score = (similarity + 1) * 5  # Scale to 1-10 range
            scores.append(score)

        return scores, metrics

    async def _load_entity_embeddings(
        self, entities: List[Tuple[str, str, str]]
    ) -> None:
        """Load pre-computed embeddings for entities."""
        if self._entity_embeddings is not None:
            return  # Already cached

        # Priority 1: Use pre-computed embeddings from vector store
        if self.entity_embedding_store:
            try:
                embeddings: List[Optional[List[float]]] = [None for _ in entities]
                self._entity_texts = [f"{name}: {desc}" for _, name, desc in entities]
                missing_indices: List[int] = []
                missing_texts: List[str] = []

                for i, (entity_id, _name, _desc) in enumerate(entities):
                    # Try to get pre-computed embedding from vector store
                    doc = self.entity_embedding_store.search_by_id(entity_id)
                    if doc and doc.vector:
                        embeddings[i] = doc.vector
                    else:
                        missing_indices.append(i)
                        missing_texts.append(self._entity_texts[i])

                if missing_indices:
                    batch_embeddings = await self.embedding_model.aembed_batch(
                        text_list=missing_texts
                    )
                    for idx, emb in zip(missing_indices, batch_embeddings):
                        embeddings[idx] = emb

                self._entity_embeddings = np.array(embeddings)
                logger.debug(
                    f"Loaded embeddings for {len(self._entity_texts)} entities from vector store"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load pre-computed embeddings: {e}, falling back to computing"
                )

        # Priority 2: Compute all embeddings
        entity_texts = [f"{name}: {desc}" for _, name, desc in entities]
        embeddings = await self.embedding_model.aembed_batch(text_list=entity_texts)
        self._entity_embeddings = np.array(embeddings)
        self._entity_texts = entity_texts
        logger.debug(f"Computed embeddings for {len(self._entity_texts)} entities")


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
        return re.findall(r"\w+", text.lower())

    def _compute_bm25_scores(self, query: str, documents: List[str]) -> List[float]:
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
    ) -> Tuple[List[Tuple[str, str, str, float, float]], PruningMetrics]:
        """Score relations using BM25."""
        metrics = PruningMetrics()

        if not relations:
            return [], metrics

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
            max(1.0, min(10.0, (s / max_score) * 9 + 1)) for s in bm25_scores
        ]

        scored_relations = [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(
                relations, normalized_scores
            )
        ]
        return scored_relations, metrics

    async def score_entities(
        self,
        query: str,
        current_path: str,
        entities: List[Tuple[str, str, str]],
    ) -> Tuple[List[float], PruningMetrics]:
        """Score entities using BM25."""
        metrics = PruningMetrics()

        if not entities:
            return [], metrics

        # Create searchable text for each entity
        entity_texts = [f"{name}: {desc}" for _, name, desc in entities]

        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query, entity_texts)

        # Normalize scores to 1-10 range
        max_score = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        normalized_scores = [
            max(1.0, min(10.0, (s / max_score) * 9 + 1)) for s in bm25_scores
        ]

        return normalized_scores, metrics
