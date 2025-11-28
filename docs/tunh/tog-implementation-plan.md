# ToG (Think-on-Graph) Implementation Plan for GraphRAG

## Executive Summary

This document outlines a detailed plan to integrate **ToG (Think-on-Graph)** as a new search method into the GraphRAG framework. ToG is a deep reasoning approach that combines LLMs with knowledge graph traversal, published in ICLR 2024. This implementation will add a fifth search method alongside Global, Local, DRIFT, and Basic search.

---

## 1. Background

### 1.1 What is ToG?

**ToG (Think-on-Graph)** is a knowledge graph reasoning system that enables LLMs to perform deep and responsible reasoning by grounding them in structured knowledge graphs. It uses a two-stage approach:

1. **Exploration Stage**: Iteratively expands through the knowledge graph using beam search
   - Retrieves relations from entities at current depth
   - Scores and prunes candidate entities using LLM or similarity methods
   - Tracks exploration history through the graph

2. **Reasoning Stage**: Generates final answers based on explored paths
   - Evaluates if sufficient information has been gathered
   - Uses chain-of-thought reasoning to produce answers
   - Returns entity chains showing reasoning paths

### 1.2 Key Differences from Existing GraphRAG Search Methods

| Feature | Global | Local | DRIFT | Basic | **ToG (Proposed)** |
|---------|--------|-------|-------|-------|-------------------|
| **Approach** | Map-reduce over communities | Entity-centric retrieval | Local + community context | Vector similarity | **Iterative graph traversal** |
| **Reasoning** | Aggregated summaries | Related entities/relations | Mixed local+community | Raw text chunks | **Multi-hop exploration with LLM scoring** |
| **Best For** | High-level themes | Specific entity details | Balanced queries | Simple lookups | **Complex multi-step reasoning questions** |
| **Graph Usage** | Community structure | Entity neighborhood | Entity + communities | None | **Dynamic relation-based traversal** |
| **Iteration** | Single pass | Single pass | Single pass | Single pass | **Depth-controlled beam search** |

### 1.3 Value Proposition

ToG will enable GraphRAG users to:
- **Answer complex reasoning questions** requiring multi-hop inference
- **Trace reasoning paths** through the knowledge graph with explainability
- **Control exploration depth** for computational efficiency
- **Customize pruning strategies** (LLM-based, BM25, semantic similarity)
- **Handle questions that existing methods struggle with**, such as:
  - "What is the connection between X and Y?"
  - "How does A influence B through intermediary factors?"
  - "What are the paths from entity X to concept Y?"

---

## 2. Technical Architecture

### 2.1 Core Components

```
ToG Search System
│
├── 1. ToG Search Engine (graphrag/query/structured_search/tog_search/)
│   ├── search.py              # Main ToGSearch class
│   ├── tog_context.py         # Context builder for ToG
│   ├── exploration.py         # Graph exploration logic
│   ├── pruning.py             # Entity/relation pruning strategies
│   ├── reasoning.py           # Final reasoning and answer generation
│   └── state.py               # Search state management
│
├── 2. Configuration (graphrag/config/models/)
│   └── tog_search_config.py   # ToG configuration parameters
│
├── 3. Factory Integration (graphrag/query/)
│   └── factory.py             # get_tog_search_engine()
│
├── 4. API Integration (graphrag/api/)
│   └── query.py               # tog_search() and tog_search_streaming()
│
├── 5. CLI Integration (graphrag/cli/)
│   └── query.py               # run_tog_search()
│
├── 6. Prompts (graphrag/prompts/query/)
│   ├── tog_relation_scoring_prompt.py
│   ├── tog_entity_scoring_prompt.py
│   └── tog_reasoning_prompt.py
│
└── 7. Documentation
    └── docs/tunh/tog_search_guide.md
```

### 2.2 Data Model Mapping

ToG operates on a different graph structure than traditional KG systems (Freebase/Wikidata). Here's how GraphRAG's data maps to ToG's requirements:

| ToG Requirement | GraphRAG Source | Adaptation Strategy |
|-----------------|-----------------|---------------------|
| **Entities** | `entities.parquet` | Direct mapping: use `title`, `description` |
| **Relations** | `relationships.parquet` | Extract `source`, `target`, `description` |
| **Entity IDs** | `entities.id` | Use as node identifiers |
| **Relation Types** | `relationships.description` | Parse or use as-is for relation semantics |
| **Entity Descriptions** | `entities.description` | Use for LLM scoring and reasoning |
| **Relation Scores** | `relationships.weight` | Use for initial pruning |

**Key Difference**: GraphRAG's relationships are semantic and described in natural language, while ToG expects structured relation types (e.g., `born_in`, `works_for`). We'll adapt by:
- Using relationship descriptions as relation identifiers
- Implementing semantic relation matching instead of exact type matching
- Leveraging relationship weights for initial filtering

---

## 3. Implementation Plan

### Phase 1: Core ToG Search Components (Priority: HIGH)

#### 3.1 Configuration Model

**File**: `graphrag/config/models/tog_search_config.py`

```python
from pydantic import BaseModel, Field

class ToGSearchConfig(BaseModel):
    """Configuration for ToG (Think-on-Graph) search."""

    # Model Configuration
    chat_model_id: str = Field(
        description="The model ID to use for ToG search.",
        default="default_chat_model"
    )
    embedding_model_id: str = Field(
        description="The model ID to use for embeddings.",
        default="default_embedding_model"
    )

    # Exploration Parameters
    width: int = Field(
        description="Beam width for exploration (number of entities to keep per level).",
        default=3
    )
    depth: int = Field(
        description="Maximum depth of graph traversal.",
        default=3
    )

    # Pruning Configuration
    prune_strategy: str = Field(
        description="Pruning strategy: 'llm', 'bm25', or 'semantic'.",
        default="llm"
    )
    num_retain_entity: int = Field(
        description="Number of entities to retain during pruning.",
        default=5
    )

    # Temperature Settings
    temperature_exploration: float = Field(
        description="Temperature for exploration phase.",
        default=0.4
    )
    temperature_reasoning: float = Field(
        description="Temperature for reasoning phase.",
        default=0.0
    )

    # Context and Token Limits
    max_context_tokens: int = Field(
        description="Maximum tokens for context.",
        default=8000
    )
    max_exploration_paths: int = Field(
        description="Maximum number of exploration paths to maintain.",
        default=10
    )

    # Prompts
    relation_scoring_prompt: str | None = Field(
        description="Custom prompt for relation scoring.",
        default=None
    )
    entity_scoring_prompt: str | None = Field(
        description="Custom prompt for entity scoring.",
        default=None
    )
    reasoning_prompt: str | None = Field(
        description="Custom prompt for final reasoning.",
        default=None
    )
```

**Integration**: Add to `graphrag/config/models/graph_rag_config.py`:
```python
from graphrag.config.models.tog_search_config import ToGSearchConfig

class GraphRagConfig(BaseModel):
    # ... existing fields ...
    tog_search: ToGSearchConfig = Field(default_factory=ToGSearchConfig)
```

#### 3.2 Search State Management

**File**: `graphrag/query/structured_search/tog_search/state.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExplorationNode:
    """Represents a node in the exploration tree."""
    entity_id: str
    entity_name: str
    entity_description: str
    depth: int
    score: float
    parent: 'ExplorationNode' | None
    relation_from_parent: str | None

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
        self.nodes_by_depth[self.current_depth] = frontier[:self.beam_width]
```

#### 3.3 Graph Exploration Logic

**File**: `graphrag/query/structured_search/tog_search/exploration.py`

```python
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
```

#### 3.4 Pruning Strategies

**File**: `graphrag/query/structured_search/tog_search/pruning.py`

```python
from typing import List, Tuple
from graphrag.language_model.models import ChatModel, EmbeddingModel
import numpy as np

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
        self.relation_scoring_prompt = relation_scoring_prompt
        self.entity_scoring_prompt = entity_scoring_prompt

    async def score_relations(
        self,
        query: str,
        entity_name: str,
        relations: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, float]]:
        """Score relations using LLM."""

        if not relations:
            return []

        # Build prompt
        relations_text = "\n".join([
            f"{i+1}. [{direction}] {rel_desc} (weight: {weight:.2f})"
            for i, (rel_desc, _, direction, weight) in enumerate(relations)
        ])

        prompt = f"""Given the question: "{query}"
Currently exploring entity: {entity_name}

Available relations:
{relations_text}

Score each relation (1-10) based on how likely it leads to answering the question.
Output format: [score1, score2, score3, ...]
Only output the list of numbers.
"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.model.async_generate(
            messages=messages,
            temperature=self.temperature,
        )

        # Parse scores
        scores = self._parse_scores(response, len(relations))

        # Combine with relation data
        return [
            (rel_desc, target_id, direction, weight, score)
            for (rel_desc, target_id, direction, weight), score in zip(relations, scores)
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
            f"{i+1}. {name}: {desc[:100]}..."
            for i, (_, name, desc) in enumerate(entities)
        ])

        prompt = f"""Given the question: "{query}"
Current exploration path: {current_path}

Candidate entities to explore:
{entities_text}

Score each entity (1-10) based on relevance to the question.
Output format: [score1, score2, score3, ...]
Only output the list of numbers.
"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.model.async_generate(
            messages=messages,
            temperature=self.temperature,
        )

        return self._parse_scores(response, len(entities))

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse score list from LLM response."""
        import re

        # Try to extract list of numbers
        numbers = re.findall(r'\d+\.?\d*', response)
        scores = [float(n) for n in numbers[:expected_count]]

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
        query_embedding = await self.embedding_model.async_generate(
            inputs=[query]
        )
        query_emb = np.array(query_embedding[0])

        # Embed relation descriptions
        relation_texts = [rel_desc for rel_desc, _, _, _ in relations]
        relation_embeddings = await self.embedding_model.async_generate(
            inputs=relation_texts
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
            for (rel_desc, target_id, direction, weight), score in zip(relations, scores)
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
        query_embedding = await self.embedding_model.async_generate(
            inputs=[query]
        )
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
```

#### 3.5 Reasoning Module

**File**: `graphrag/query/structured_search/tog_search/reasoning.py`

```python
from typing import List
from graphrag.language_model.models import ChatModel
from .state import ExplorationNode

class ToGReasoning:
    """Handles final reasoning and answer generation."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.0,
        reasoning_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_prompt = reasoning_prompt

    async def generate_answer(
        self,
        query: str,
        exploration_paths: List[ExplorationNode],
    ) -> Tuple[str, List[str]]:
        """
        Generate final answer from exploration paths.
        Returns: (answer, reasoning_paths)
        """

        # Format exploration paths
        paths_text = self._format_paths(exploration_paths)

        prompt = f"""Based on the following exploration paths through the knowledge graph, answer the question.

Question: {query}

Exploration Paths:
{paths_text}

Provide a comprehensive answer based on the information discovered through graph exploration.
Include specific entities and relationships in your reasoning.

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        answer = await self.model.async_generate(
            messages=messages,
            temperature=self.temperature,
        )

        # Extract reasoning paths for transparency
        reasoning_paths = [self._path_to_string(node) for node in exploration_paths]

        return answer, reasoning_paths

    def _format_paths(self, nodes: List[ExplorationNode]) -> str:
        """Format exploration paths as readable text."""
        paths = []
        for i, node in enumerate(nodes, 1):
            path_str = self._path_to_string(node)
            paths.append(f"Path {i}: {path_str}")
            paths.append(f"  Final entity: {node.entity_name}")
            paths.append(f"  Description: {node.entity_description[:200]}...")
            paths.append("")

        return "\n".join(paths)

    def _path_to_string(self, node: ExplorationNode) -> str:
        """Convert exploration path to string."""
        path = node.get_path()
        if not path:
            return node.entity_name

        path_parts = [node.entity_name]
        current = node
        while current.parent is not None:
            path_parts.insert(0, f"{current.relation_from_parent}")
            path_parts.insert(0, current.parent.entity_name)
            current = current.parent

        return " -> ".join(path_parts)

    async def check_early_termination(
        self,
        query: str,
        current_nodes: List[ExplorationNode],
    ) -> Tuple[bool, str | None]:
        """
        Check if exploration can terminate early with an answer.
        Returns: (should_terminate, answer_or_none)
        """

        paths_text = self._format_paths(current_nodes[:3])  # Check top 3 paths

        prompt = f"""Question: {query}

Current exploration paths:
{paths_text}

Can you answer the question with high confidence based on these paths?
Respond with:
- "YES: [answer]" if you can answer confidently
- "NO: [reason]" if more exploration is needed

Response:"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.model.async_generate(
            messages=messages,
            temperature=0.0,
        )

        if response.strip().upper().startswith("YES:"):
            answer = response[4:].strip()
            return True, answer

        return False, None
```

#### 3.6 Main ToG Search Engine

**File**: `graphrag/query/structured_search/tog_search/search.py`

```python
from typing import AsyncGenerator, List
from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.language_model.models import ChatModel
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.base import Tokenizer
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
        """Perform ToG search with streaming response."""

        # Initialize search state
        starting_entities = self.explorer.find_starting_entities(query, top_k=self.width)

        state = ToGSearchState(
            query=query,
            current_depth=0,
            nodes_by_depth={},
            finished_paths=[],
            max_depth=self.depth,
            beam_width=self.width,
        )

        # Create root nodes
        for entity_id in starting_entities:
            entity_info = self.explorer.get_entity_info(entity_id)
            if entity_info:
                name, desc = entity_info
                node = ExplorationNode(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_description=desc,
                    depth=0,
                    score=10.0,  # Starting entities get max score
                    parent=None,
                    relation_from_parent=None,
                )
                state.add_node(node)

        # Exploration loop
        for depth in range(self.depth):
            state.current_depth = depth
            frontier = state.get_current_frontier()

            if not frontier:
                break

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_context({"depth": depth, "frontier_size": len(frontier)})

            # Check for early termination
            should_stop, early_answer = await self.reasoning_module.check_early_termination(
                query, frontier
            )
            if should_stop and early_answer:
                yield early_answer
                return

            # Expand each node in frontier
            new_nodes = []
            for node in frontier:
                expanded = await self._expand_node(query, node, state)
                new_nodes.extend(expanded)

            # Prune to keep top entities
            if new_nodes:
                new_nodes.sort(key=lambda n: n.score, reverse=True)
                for node in new_nodes[:self.width * self.num_retain_entity]:
                    state.add_node(node)

        # Generate final answer from all explored paths
        all_paths = []
        for depth_nodes in state.nodes_by_depth.values():
            all_paths.extend(depth_nodes)

        # Take top paths by score
        all_paths.sort(key=lambda n: n.score, reverse=True)
        top_paths = all_paths[:self.width * 2]

        answer, reasoning_paths = await self.reasoning_module.generate_answer(
            query, top_paths
        )

        # Include reasoning paths in context
        context = {
            "reasoning_paths": reasoning_paths,
            "num_explored_entities": len(all_paths),
            "max_depth_reached": max(state.nodes_by_depth.keys()) if state.nodes_by_depth else 0,
        }
        for callback in self.callbacks:
            callback.on_context(context)

        yield answer

    async def _expand_node(
        self,
        query: str,
        node: ExplorationNode,
        state: ToGSearchState,
    ) -> List[ExplorationNode]:
        """Expand a node by exploring its relations."""

        # Get relations for this entity
        relations = self.explorer.get_relations(node.entity_id, bidirectional=True)

        if not relations:
            return []

        # Score relations
        scored_relations = await self.pruning_strategy.score_relations(
            query=query,
            entity_name=node.entity_name,
            relations=relations,
        )

        # Keep top relations
        scored_relations.sort(key=lambda r: r[4], reverse=True)  # Sort by score
        top_relations = scored_relations[:self.num_retain_entity]

        # Get candidate entities
        candidate_entities = []
        for rel_desc, target_id, direction, weight, rel_score in top_relations:
            entity_info = self.explorer.get_entity_info(target_id)
            if entity_info:
                name, desc = entity_info
                candidate_entities.append((target_id, name, desc, rel_desc, rel_score))

        if not candidate_entities:
            return []

        # Score entities
        entity_data = [(eid, name, desc) for eid, name, desc, _, _ in candidate_entities]
        current_path = " -> ".join([n.entity_name for n in self._get_path_to_node(node)])

        entity_scores = await self.pruning_strategy.score_entities(
            query=query,
            current_path=current_path,
            entities=entity_data,
        )

        # Create new nodes
        new_nodes = []
        for (entity_id, name, desc, relation, rel_score), ent_score in zip(
            candidate_entities, entity_scores
        ):
            combined_score = (rel_score + ent_score) / 2.0

            new_node = ExplorationNode(
                entity_id=entity_id,
                entity_name=name,
                entity_description=desc,
                depth=node.depth + 1,
                score=combined_score,
                parent=node,
                relation_from_parent=relation,
            )
            new_nodes.append(new_node)

        return new_nodes

    def _get_path_to_node(self, node: ExplorationNode) -> List[ExplorationNode]:
        """Get path from root to node."""
        path = []
        current = node
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path
```

---

### Phase 2: Integration and Configuration (Priority: HIGH)

#### 3.7 Factory Integration

**File**: `graphrag/query/factory.py`

Add the following function:

```python
def get_tog_search_engine(
    config: GraphRagConfig,
    entities: list[Entity],
    relationships: list[Relationship],
    response_type: str,
    callbacks: list[QueryCallbacks] | None = None,
) -> ToGSearch:
    """Create a ToG search engine based on data + configuration."""

    chat_model_settings = config.get_language_model_config(
        config.tog_search.chat_model_id
    )

    chat_model = ModelManager().get_or_create_chat_model(
        name="tog_search_chat",
        model_type=chat_model_settings.type,
        config=chat_model_settings,
    )

    embedding_model_settings = config.get_language_model_config(
        config.tog_search.embedding_model_id
    )

    embedding_model = ModelManager().get_or_create_embedding_model(
        name="tog_search_embedding",
        model_type=embedding_model_settings.type,
        config=embedding_model_settings,
    )

    tokenizer = get_tokenizer(model_config=chat_model_settings)

    # Create pruning strategy
    if config.tog_search.prune_strategy == "llm":
        pruning_strategy = LLMPruning(
            model=chat_model,
            temperature=config.tog_search.temperature_exploration,
            relation_scoring_prompt=config.tog_search.relation_scoring_prompt,
            entity_scoring_prompt=config.tog_search.entity_scoring_prompt,
        )
    elif config.tog_search.prune_strategy == "semantic":
        pruning_strategy = SemanticPruning(
            embedding_model=embedding_model
        )
    else:
        raise ValueError(f"Unknown pruning strategy: {config.tog_search.prune_strategy}")

    # Create reasoning module
    reasoning_module = ToGReasoning(
        model=chat_model,
        temperature=config.tog_search.temperature_reasoning,
        reasoning_prompt=config.tog_search.reasoning_prompt,
    )

    return ToGSearch(
        model=chat_model,
        entities=entities,
        relationships=relationships,
        tokenizer=tokenizer,
        pruning_strategy=pruning_strategy,
        reasoning_module=reasoning_module,
        width=config.tog_search.width,
        depth=config.tog_search.depth,
        num_retain_entity=config.tog_search.num_retain_entity,
        callbacks=callbacks,
    )
```

#### 3.8 API Integration

**File**: `graphrag/api/query.py`

Add the following functions:

```python
@validate_call(config={"arbitrary_types_allowed": True})
async def tog_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame],
]:
    """Perform a ToG search and return the context data and response.

    Parameters
    ----------
    - config (GraphRagConfig): A graphrag configuration (from settings.yaml)
    - entities (pd.DataFrame): A DataFrame containing the final entities (from entities.parquet)
    - relationships (pd.DataFrame): A DataFrame containing the final relationships (from relationships.parquet)
    - query (str): The user query to search for.

    Returns
    -------
    TODO: Document the search response type and format.
    """
    init_loggers(config=config, verbose=verbose, filename="query.log")

    callbacks = callbacks or []
    full_response = ""
    context_data = {}

    def on_context(context: Any) -> None:
        nonlocal context_data
        context_data = context

    local_callbacks = NoopQueryCallbacks()
    local_callbacks.on_context = on_context
    callbacks.append(local_callbacks)

    logger.debug("Executing ToG search query: %s", query)
    async for chunk in tog_search_streaming(
        config=config,
        entities=entities,
        relationships=relationships,
        query=query,
        callbacks=callbacks,
    ):
        full_response += chunk
    logger.debug("Query response: %s", truncate(full_response, 400))
    return full_response, context_data


@validate_call(config={"arbitrary_types_allowed": True})
def tog_search_streaming(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> AsyncGenerator:
    """Perform a ToG search and return the context data and response via a generator.

    Parameters
    ----------
    - config (GraphRagConfig): A graphrag configuration (from settings.yaml)
    - entities (pd.DataFrame): A DataFrame containing the final entities (from entities.parquet)
    - relationships (pd.DataFrame): A DataFrame containing the final relationships (from relationships.parquet)
    - query (str): The user query to search for.

    Returns
    -------
    TODO: Document the search response type and format.
    """
    init_loggers(config=config, verbose=verbose, filename="query.log")

    entities_ = read_indexer_entities(entities, communities=None, community_level=None)
    relationships_ = read_indexer_relationships(relationships)

    logger.debug("Executing streaming ToG search query: %s", query)
    search_engine = get_tog_search_engine(
        config=config,
        entities=entities_,
        relationships=relationships_,
        response_type="detailed",  # ToG always provides detailed responses
        callbacks=callbacks,
    )
    return search_engine.stream_search(query=query)
```

#### 3.9 CLI Integration

**File**: `graphrag/cli/query.py`

Add the following function:

```python
def run_tog_search(
    config_filepath: Path | None,
    data_dir: Path | None,
    root_dir: Path,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a ToG search with a given query.

    Loads index files required for ToG search and calls the Query API.
    """
    root = root_dir.resolve()
    cli_overrides = {}
    if data_dir:
        cli_overrides["output.base_dir"] = str(data_dir)
    config = load_config(root, config_filepath, cli_overrides)

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "entities",
            "relationships",
        ],
    )

    final_entities: pd.DataFrame = dataframe_dict["entities"]
    final_relationships: pd.DataFrame = dataframe_dict["relationships"]

    if streaming:
        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.tog_search_streaming(
                config=config,
                entities=final_entities,
                relationships=final_relationships,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())

    # not streaming
    response, context_data = asyncio.run(
        api.tog_search(
            config=config,
            entities=final_entities,
            relationships=final_relationships,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data
```

Update the main CLI argument parser to include ToG:

```python
# In graphrag/cli/main.py or wherever the query subcommand is defined
query_parser.add_argument(
    "--method",
    type=str,
    choices=["local", "global", "drift", "basic", "tog"],  # Add "tog"
    default="local",
    help="The search method to use",
)
```

---

### Phase 3: Prompts and Documentation (Priority: MEDIUM)

#### 3.10 System Prompts

**File**: `graphrag/prompts/query/tog_relation_scoring_prompt.py`

```python
"""Default relation scoring prompt for ToG search."""

TOG_RELATION_SCORING_PROMPT = """
You are an expert at analyzing knowledge graph relations to determine their relevance for answering questions.

Given:
- A question to answer
- The current entity being explored
- A list of relations connected to this entity

Your task:
Rate each relation on a scale of 1-10 based on how likely following that relation will lead to information needed to answer the question.

Consider:
- Semantic relevance of the relation to the question
- Direction of the relation (incoming vs outgoing)
- The strength/weight of the relation
- How this relation fits into the overall exploration path

Output only a list of numerical scores in order, e.g.: [8, 6, 9, 3, 7]
"""
```

**File**: `graphrag/prompts/query/tog_entity_scoring_prompt.py`

```python
"""Default entity scoring prompt for ToG search."""

TOG_ENTITY_SCORING_PROMPT = """
You are an expert at analyzing entities in a knowledge graph to determine their relevance for answering questions.

Given:
- A question to answer
- The current exploration path through the graph
- A list of candidate entities to explore next

Your task:
Rate each entity on a scale of 1-10 based on how likely exploring that entity will help answer the question.

Consider:
- Semantic relevance of the entity to the question
- How the entity fits into the current exploration path
- The entity's description and its potential to provide key information
- Whether this entity represents a promising direction for continued exploration

Output only a list of numerical scores in order, e.g.: [8, 6, 9, 3, 7]
"""
```

**File**: `graphrag/prompts/query/tog_reasoning_prompt.py`

```python
"""Default reasoning prompt for ToG search."""

TOG_REASONING_PROMPT = """
You are an expert at synthesizing information from knowledge graph exploration to answer questions.

You will be given:
- A question to answer
- Multiple exploration paths through a knowledge graph
- Entities and relationships discovered along these paths

Your task:
1. Analyze all the exploration paths provided
2. Identify the most relevant information for answering the question
3. Synthesize this information into a comprehensive answer
4. Explain your reasoning, citing specific entities and relationships

Requirements:
- Base your answer ONLY on the provided graph exploration results
- Cite specific entities and relationships in your answer
- If the exploration paths don't contain sufficient information, acknowledge this
- Provide a clear, well-structured response

Structure your response as:
1. Direct answer to the question
2. Supporting evidence from the graph exploration
3. Key relationships that support your answer
"""
```

#### 3.11 User Documentation

**File**: `docs/tunh/tog_search_guide.md`

```markdown
# ToG (Think-on-Graph) Search Guide

## Overview

ToG (Think-on-Graph) is an advanced search method in GraphRAG that enables deep, multi-hop reasoning over knowledge graphs. Unlike other search methods, ToG iteratively explores the graph using beam search guided by LLM scoring, making it ideal for complex questions requiring multi-step inference.

## When to Use ToG Search

### Best Use Cases

ToG search excels at:

1. **Multi-hop reasoning questions**
   - "What is the connection between entity A and entity B?"
   - "How does X influence Y through intermediary factors?"

2. **Path-finding queries**
   - "What are the paths from X to Y?"
   - "How are these two concepts related?"

3. **Complex causal chains**
   - "What factors led to outcome Z?"
   - "Trace the chain of events from A to B"

4. **Exploratory analysis**
   - "What are the indirect relationships between X and Y?"
   - "Find unexpected connections in the data"

### When NOT to Use ToG

- **Simple factual lookups**: Use Local search instead
- **High-level summaries**: Use Global search instead
- **Quick overviews**: Use Basic search instead
- **Specific entity details**: Use Local search instead

## Configuration

Add ToG configuration to your `settings.yaml`:

```yaml
tog_search:
  # Model Configuration
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model

  # Exploration Parameters
  width: 3                    # Beam width - number of paths to explore
  depth: 3                    # Maximum depth of graph traversal

  # Pruning Configuration
  prune_strategy: llm         # Options: "llm", "semantic"
  num_retain_entity: 5        # Entities to keep at each step

  # Temperature Settings
  temperature_exploration: 0.4  # Randomness during exploration
  temperature_reasoning: 0.0    # Deterministic final reasoning

  # Resource Limits
  max_context_tokens: 8000
  max_exploration_paths: 10
```

### Configuration Parameters Explained

**width**: Controls the breadth of exploration
- Lower (1-2): Focused, faster, may miss alternative paths
- Higher (5+): Comprehensive, slower, more complete exploration
- Recommended: 3 for balanced exploration

**depth**: Controls how far to traverse the graph
- Lower (1-2): Quick, surface-level connections
- Higher (4+): Deep, multi-hop reasoning
- Recommended: 3 for most questions

**prune_strategy**:
- `llm`: Uses language model to score relevance (slower, more accurate)
- `semantic`: Uses embedding similarity (faster, good quality)

**num_retain_entity**: How many candidate entities to evaluate at each step
- Higher values = more thorough but slower
- Recommended: 5 for balanced performance

## Usage

### Command Line

```bash
# Basic ToG search
graphrag query --root ./my-project --method tog --query "What connects entity A to entity B?"

# With streaming output
graphrag query --root ./my-project --method tog --streaming --query "Your question here"
```

### Python API

```python
import asyncio
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.api import tog_search

# Load configuration and data
config = load_config(root_dir="./my-project")
entities = pd.read_parquet("./my-project/output/create_final_entities.parquet")
relationships = pd.read_parquet("./my-project/output/create_final_relationships.parquet")

# Run ToG search
response, context = asyncio.run(
    tog_search(
        config=config,
        entities=entities,
        relationships=relationships,
        query="What is the connection between X and Y?"
    )
)

print(response)
print(f"Explored {context['num_explored_entities']} entities")
print(f"Reasoning paths: {context['reasoning_paths']}")
```

## Understanding ToG Results

ToG returns both an answer and exploration metadata:

```python
{
    "reasoning_paths": [
        "EntityA -> relationX -> EntityB -> relationY -> EntityC",
        "EntityA -> relationZ -> EntityD -> relationY -> EntityC"
    ],
    "num_explored_entities": 45,
    "max_depth_reached": 3
}
```

**reasoning_paths**: The paths through the graph that led to the answer
**num_explored_entities**: Total entities considered during exploration
**max_depth_reached**: How deep the search went

## Performance Considerations

### Speed vs. Quality Trade-offs

| Setting | Fast | Balanced | Thorough |
|---------|------|----------|----------|
| width | 1-2 | 3 | 5+ |
| depth | 1-2 | 3 | 4+ |
| prune_strategy | semantic | semantic | llm |
| num_retain_entity | 3 | 5 | 10 |

### Cost Optimization

ToG makes multiple LLM calls during exploration:
- Relation scoring: ~(width × depth × num_retain_entity) calls
- Entity scoring: Similar magnitude
- Final reasoning: 1-2 calls

**Tips to reduce cost:**
1. Use faster/cheaper models for exploration
2. Use semantic pruning instead of LLM pruning
3. Reduce width and depth for simpler questions
4. Cache exploration results for similar queries

## Advanced Usage

### Custom Prompts

Override default prompts in `settings.yaml`:

```yaml
tog_search:
  relation_scoring_prompt: "./prompts/my_relation_prompt.txt"
  entity_scoring_prompt: "./prompts/my_entity_prompt.txt"
  reasoning_prompt: "./prompts/my_reasoning_prompt.txt"
```

### Programmatic Control

```python
from graphrag.query.structured_search.tog_search import ToGSearch
from graphrag.query.factory import get_tog_search_engine

# Create custom search engine
search_engine = get_tog_search_engine(
    config=config,
    entities=entities_list,
    relationships=relationships_list,
    response_type="detailed",
)

# Run with custom callbacks
class MyCallback:
    def on_context(self, context):
        print(f"Depth {context.get('depth')}: exploring {context.get('frontier_size')} entities")

result = await search_engine.search(
    query="Your question",
)
```

## Troubleshooting

### Issue: Search takes too long

**Solutions:**
- Reduce `width` and `depth` parameters
- Switch to `semantic` pruning strategy
- Reduce `num_retain_entity`

### Issue: Answer quality is poor

**Solutions:**
- Increase `width` to explore more paths
- Increase `depth` for deeper reasoning
- Switch to `llm` pruning strategy
- Check if entities and relationships were extracted well during indexing

### Issue: Out of memory errors

**Solutions:**
- Reduce `max_exploration_paths`
- Process smaller graph subsets
- Use semantic pruning (lighter weight)

## Comparison with Other Methods

| Aspect | ToG | Local | Global | DRIFT |
|--------|-----|-------|--------|-------|
| Question Complexity | High | Medium | Low-Medium | Medium |
| Reasoning Depth | Multi-hop | Single-hop | Aggregated | Mixed |
| Speed | Slower | Fast | Medium | Medium |
| Explainability | High (paths) | Medium | Low | Medium |
| Best For | Complex reasoning | Entity details | Themes | Balanced |
```

---

## 4. Testing Strategy

### 4.1 Unit Tests

**File**: `tests/unit/query/structured_search/tog_search/test_exploration.py`

```python
import pytest
from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.query.structured_search.tog_search.exploration import GraphExplorer

def test_graph_explorer_initialization():
    entities = [
        Entity(id="1", title="Entity A", description="Description A"),
        Entity(id="2", title="Entity B", description="Description B"),
    ]
    relationships = [
        Relationship(source="1", target="2", description="related_to", weight=0.8),
    ]

    explorer = GraphExplorer(entities, relationships)

    assert len(explorer.entities) == 2
    assert "1" in explorer.outgoing
    assert "2" in explorer.incoming

def test_get_relations():
    # ... test relation retrieval
    pass

def test_find_starting_entities():
    # ... test starting entity identification
    pass
```

### 4.2 Integration Tests

**File**: `tests/integration/test_tog_search.py`

```python
import pytest
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.factory import get_tog_search_engine

@pytest.mark.asyncio
async def test_tog_search_end_to_end(sample_entities, sample_relationships, sample_config):
    """Test full ToG search pipeline."""

    search_engine = get_tog_search_engine(
        config=sample_config,
        entities=sample_entities,
        relationships=sample_relationships,
        response_type="detailed",
    )

    result = await search_engine.search(
        query="What is the connection between Entity A and Entity C?"
    )

    assert result is not None
    assert len(result) > 0
```

### 4.3 Performance Benchmarks

Create benchmark suite comparing ToG with other methods on:
- Multi-hop reasoning questions
- Path-finding queries
- Response time
- LLM token usage
- Answer quality (human evaluation)

---

## 5. Implementation Timeline

### Week 1-2: Core Components
- [ ] Implement configuration models
- [ ] Implement search state management
- [ ] Implement graph exploration logic
- [ ] Unit tests for core components

### Week 3-4: Pruning and Reasoning
- [ ] Implement LLM pruning strategy
- [ ] Implement semantic pruning strategy
- [ ] Implement reasoning module
- [ ] Unit tests for pruning and reasoning

### Week 5-6: Search Engine and Integration
- [ ] Implement main ToGSearch class
- [ ] Integrate with factory
- [ ] Integrate with API
- [ ] Integrate with CLI
- [ ] Integration tests

### Week 7: Prompts and Documentation
- [ ] Create default prompts
- [ ] Write user documentation
- [ ] Create example notebooks
- [ ] Performance benchmarking

### Week 8: Testing and Refinement
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Bug fixes
- [ ] Code review and refinement

---

## 6. Risks and Mitigations

### Risk 1: Performance - ToG is computationally expensive

**Mitigations:**
- Implement semantic pruning as a faster alternative
- Add caching for relation/entity scores
- Provide configuration guidance for speed vs. quality
- Implement early termination when answer is found

### Risk 2: GraphRAG's relationships are different from traditional KG

**Mitigations:**
- Use semantic matching instead of exact relation types
- Leverage relationship descriptions for scoring
- Allow fuzzy relation matching
- Document adaptation strategies clearly

### Risk 3: Complex prompts may be model-dependent

**Mitigations:**
- Provide multiple prompt templates
- Allow custom prompt overrides
- Test with multiple LLM providers
- Document prompt engineering guidelines

### Risk 4: User confusion about when to use ToG

**Mitigations:**
- Provide clear use case documentation
- Include decision tree for method selection
- Provide example queries for each method
- Add CLI hints or recommendations

---

## 7. Future Enhancements

### Phase 2 (Post-MVP)

1. **BM25 Pruning Strategy**
   - Implement BM25-based relation/entity scoring
   - Lighter weight than LLM, better than pure semantic

2. **Hybrid Pruning**
   - Combine multiple strategies
   - Use fast methods for initial filtering, LLM for final ranking

3. **Interactive Exploration**
   - Allow users to guide exploration interactively
   - Visualize exploration paths in real-time

4. **Path Caching**
   - Cache commonly explored paths
   - Reuse exploration results across queries

5. **Adaptive Depth/Width**
   - Automatically adjust parameters based on query complexity
   - Early stopping when confidence is high

6. **Multi-modal Support**
   - Support image nodes in graph
   - Handle document-level entities

---

## 8. Success Metrics

### Technical Metrics
- [ ] ToG search executes without errors on standard test queries
- [ ] Response time < 30s for depth=3, width=3 on moderate-sized graphs
- [ ] Memory usage stays within 2GB for exploration
- [ ] All unit tests pass (>95% coverage)
- [ ] Integration tests pass

### Quality Metrics
- [ ] Answers complex multi-hop questions that other methods fail on
- [ ] Provides explainable reasoning paths
- [ ] Human evaluation: ToG answers rated higher quality than alternatives for complex questions
- [ ] Successfully identifies connections between distant entities

### Usability Metrics
- [ ] Clear documentation with examples
- [ ] Users can configure and run ToG within 5 minutes
- [ ] Configuration errors provide helpful messages
- [ ] CLI and API match existing patterns

---

## 9. Dependencies

### Required
- GraphRAG core components (entities, relationships, data models)
- Language model infrastructure (ChatModel, EmbeddingModel)
- Configuration system
- Existing query infrastructure

### Optional
- Vector stores (for semantic pruning)
- BM25 implementation (for BM25 pruning)
- Visualization tools (for path display)

---

## 10. Appendix: Code Style and Conventions

### Follow GraphRAG Patterns
- Use Pydantic models for configuration
- Follow existing naming conventions
- Use async/await consistently
- Implement streaming where possible
- Include type hints
- Add comprehensive docstrings

### Error Handling
```python
class ToGSearchError(Exception):
    """Base exception for ToG search errors."""
    pass

class GraphExplorationError(ToGSearchError):
    """Error during graph exploration."""
    pass

class PruningError(ToGSearchError):
    """Error during pruning."""
    pass
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Use throughout implementation
logger.debug(f"Exploring entity {entity_name} at depth {depth}")
logger.info(f"ToG search completed: explored {num_entities} entities")
logger.warning(f"No relations found for entity {entity_id}")
logger.error(f"Pruning failed: {error}")
```

---

## Conclusion

This plan provides a comprehensive roadmap for implementing ToG as a new search method in GraphRAG. The implementation will:

1. **Add significant value** by enabling complex multi-hop reasoning
2. **Integrate seamlessly** with existing GraphRAG architecture
3. **Maintain consistency** with current search method patterns
4. **Provide flexibility** through multiple pruning strategies
5. **Ensure quality** through comprehensive testing
6. **Enable adoption** through clear documentation

The modular design allows for incremental development and testing, with clear milestones and success criteria at each phase.
