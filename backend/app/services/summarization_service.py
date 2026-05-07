"""Summarization service for compressing conversation history."""

import logging
from pathlib import Path

from litellm import acompletion

from ..config import settings
from ..models.schemas import ConversationTurn

logger = logging.getLogger(__name__)

SUMMARIZATION_KEEP_TURNS = 3  # user turns to keep after summarization


class SummarizationService:
    """Compresses conversation history into a routing-relevant summary."""

    def __init__(self):
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = (
            Path(__file__).parent.parent.parent / "prompts" / "summarization_prompt.txt"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        return (
            "Summarize the following conversation in 2-4 sentences, focusing on topics, "
            "entities, and user intent:\n{existing_summary_block}\n{conversation_text}"
        )

    async def _call_llm(self, prompt: str) -> str:
        response = await acompletion(
            model=settings.default_chat_model_litellm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            api_key=settings.default_chat_model_api_key,
        )
        return response.choices[0].message.content or ""

    def _format_turns(self, turns: list[ConversationTurn]) -> str:
        lines = []
        for turn in turns:
            if turn.role == "user":
                q = turn.rewritten_query or turn.content
                lines.append(f"User: {q}")
            else:
                content = (
                    turn.content[:200] + "..."
                    if len(turn.content) > 200
                    else turn.content
                )
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)

    def get_trimmed_history(
        self,
        conversation_history: list[ConversationTurn],
        keep_turns: int = SUMMARIZATION_KEEP_TURNS,
    ) -> list[ConversationTurn]:
        """Return the last `keep_turns` user turns and their assistant pairs."""
        user_count = 0
        cutoff = 0
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].role == "user":
                user_count += 1
                if user_count == keep_turns:
                    cutoff = i
                    break
        return conversation_history[cutoff:]

    async def summarize(
        self,
        conversation_history: list[ConversationTurn],
        existing_summary: str | None = None,
    ) -> str:
        """
        Compress conversation turns into a routing-relevant summary.

        Args:
            conversation_history: Turns to summarize
            existing_summary: Prior summary to incorporate

        Returns
        -------
            New summary string. Falls back to basic concatenation on LLM error.
        """
        existing_summary_block = ""
        if existing_summary:
            existing_summary_block = (
                f"Previous summary:\n{existing_summary}\n\nNew turns to incorporate:"
            )

        conversation_text = self._format_turns(conversation_history)

        prompt = self.prompt_template.format(
            existing_summary_block=existing_summary_block,
            conversation_text=conversation_text,
        )

        try:
            return await self._call_llm(prompt)
        except Exception as e:
            logger.warning("Summarization LLM call failed: %s. Using fallback.", e)
            # Fallback: join user questions as plain text
            user_questions = [
                t.rewritten_query or t.content
                for t in conversation_history
                if t.role == "user"
            ]
            base = existing_summary or ""
            return (base + " " + "; ".join(user_questions)).strip()


# Global instance
summarization_service = SummarizationService()
