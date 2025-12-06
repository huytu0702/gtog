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
