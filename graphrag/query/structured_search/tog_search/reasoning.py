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
        prompt = self.reasoning_prompt.format(query=query, exploration_paths=paths_text)

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

        messages = [{"role": "user", "content": prompt}]
        response = await self.model.async_generate(
            messages=messages,
            temperature=0.0,
        )

        if response.strip().upper().startswith("YES:"):
            answer = response[4:].strip()
            return True, answer

        return False, None