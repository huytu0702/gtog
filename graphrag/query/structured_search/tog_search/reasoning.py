from typing import List
from graphrag.language_model.protocol.base import ChatModel
from .state import ExplorationNode
from graphrag.prompts.query.tog_reasoning_prompt import TOG_REASONING_PROMPT


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
        self.reasoning_prompt = reasoning_prompt or TOG_REASONING_PROMPT

    async def generate_answer(
        self,
        query: str,
        exploration_paths: List[ExplorationNode],
    ) -> tuple[str, List[str]]:
        """
        Generate final answer from exploration paths.
        Returns: (answer, reasoning_paths)
        """

        # Format exploration paths
        paths_text = self._format_paths(exploration_paths)

        # Replace placeholders in the prompt
        try:
            # If reasoning_prompt is a file path, read it
            if hasattr(
                self.reasoning_prompt, "endswith"
            ) and self.reasoning_prompt.endswith(".txt"):
                import os

                if os.path.exists(self.reasoning_prompt):
                    with open(self.reasoning_prompt, "r", encoding="utf-8") as f:
                        prompt_template = f.read()
                else:
                    prompt_template = TOG_REASONING_PROMPT
            else:
                prompt_template = self.reasoning_prompt

            prompt = prompt_template.format(query=query, exploration_paths=paths_text)
        except KeyError as e:
            # Fallback if prompt has different placeholders
            prompt = f"""
You are an expert at synthesizing information from knowledge graph exploration to answer questions.

Question: {query}

Exploration Paths:
{paths_text}

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

        answer = ""
        try:
            async for chunk in self.model.achat_stream(
                prompt=prompt,
                history=[],
                model_parameters={"temperature": self.temperature},
            ):
                answer += chunk

        except Exception as e:
            # Fallback response if LLM call fails
            answer = f"Error generating answer: {str(e)}\n\nBased on the exploration paths, I found {len(exploration_paths)} potential paths to explore."

        # Extract reasoning paths for transparency
        reasoning_paths = [self._path_to_string(node) for node in exploration_paths]

        return answer, reasoning_paths

    def _format_paths(self, nodes: List[ExplorationNode]) -> str:
        """Format exploration paths as knowledge triplets (ToG paper format)."""
        triplets = []
        seen_triplets = set()
        
        for node in nodes:
            # Extract triplets from the path
            node_triplets = self._extract_triplets(node)
            for triplet in node_triplets:
                triplet_str = f'("{triplet[0]}", {triplet[1]}, "{triplet[2]}")'
                if triplet_str not in seen_triplets:
                    triplets.append(triplet_str)
                    seen_triplets.add(triplet_str)
        
        if not triplets:
            # Fallback: just list entities with descriptions
            paths = []
            for i, node in enumerate(nodes, 1):
                desc = node.entity_description[:150] if node.entity_description else "No description"
                paths.append(f"- Entity {i}: {node.entity_name} - {desc}...")
            return "\n".join(paths)
        
        return "\n".join([f"- {t}" for t in triplets])
    
    def _extract_triplets(self, node: ExplorationNode) -> List[tuple]:
        """Extract knowledge triplets from a path."""
        triplets = []
        current = node
        
        while current.parent is not None:
            # Create triplet: (parent_entity, relation, current_entity)
            triplet = (
                current.parent.entity_name,
                current.relation_from_parent or "related_to",
                current.entity_name
            )
            triplets.append(triplet)
            current = current.parent
        
        return list(reversed(triplets))

    def _path_to_string(self, node: ExplorationNode) -> str:
        """Convert exploration path to triplet-based string."""
        triplets = self._extract_triplets(node)
        
        if not triplets:
            return node.entity_name
        
        # Format as chain of triplets
        parts = []
        for source, relation, target in triplets:
            parts.append(f"{source} --[{relation}]--> {target}")
        
        return " | ".join(parts)

    async def check_early_termination(
        self,
        query: str,
        current_nodes: List[ExplorationNode],
    ) -> tuple[bool, str | None]:
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

        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": 0.0},
        ):
            response += chunk

        if response.strip().upper().startswith("YES:"):
            answer = response[4:].strip()
            return True, answer

        return False, None
