"""Router Agent service for intelligent query routing."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from litellm import acompletion
from litellm.exceptions import RateLimitError

from ..config import settings
from ..models.schemas import ConversationTurn

logger = logging.getLogger(__name__)

SearchMethodType = Literal["local", "global", "tog", "drift", "web"]

RECENT_TURNS_IN_PROMPT = 3   # user turns to include in prompt (after summary)


@dataclass
class RouteDecision:
    """Result of router agent decision."""

    method: SearchMethodType
    confidence: float
    reasoning: str
    rewritten_query: str = field(default="")


class RouterAgent:
    """Agent that routes queries to the optimal search method."""

    def __init__(self):
        """Initialize the router agent."""
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the router prompt template."""
        prompt_path = (
            Path(__file__).parent.parent.parent / "prompts" / "router_prompt.txt"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        return self._default_prompt()

    def _default_prompt(self) -> str:
        """Return default prompt if file not found."""
        return """Analyze the query and return JSON with rewritten_query, method, confidence, reasoning.
Methods: local, global, tog, drift, web
Query: {query}
Collection: {collection_context}
{conversation_history_block}"""

    def _format_history_block(
        self,
        conversation_history: list[ConversationTurn],
        conversation_summary: str | None,
    ) -> str:
        """Format summary + recent turns into a single prompt block."""
        if not conversation_history and not conversation_summary:
            return ""

        sections = []

        if conversation_summary:
            sections.append(
                f"Past conversation summary:\n{conversation_summary}"
            )

        if conversation_history:
            # Keep last RECENT_TURNS_IN_PROMPT user turns + their assistant pairs
            user_count = 0
            cutoff = 0
            for i in range(len(conversation_history) - 1, -1, -1):
                if conversation_history[i].role == "user":
                    user_count += 1
                    if user_count == RECENT_TURNS_IN_PROMPT:
                        cutoff = i
                        break

            recent = conversation_history[cutoff:]
            label = "Recent conversation (most recent last):" if conversation_summary else "Conversation history (most recent last):"
            lines = [label]

            for turn in recent:
                try:
                    if turn.role == "user":
                        meta = ""
                        if turn.rewritten_query:
                            meta += f'  →  rewritten: "{turn.rewritten_query}"'
                        if turn.method_used:
                            meta += f"  →  method: {turn.method_used}"
                        lines.append(f"[User] {turn.content}{meta}")
                    else:
                        content = turn.content[:300] + "..." if len(turn.content) > 300 else turn.content
                        lines.append(f"[Assistant] {content}")
                except Exception:
                    logger.warning("Skipping malformed conversation turn")
                    continue

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    async def _call_llm(self, prompt: str):
        """Call LLM API using litellm with exponential backoff on rate limits."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = await acompletion(
                    model=settings.default_chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,  # Increased for more complete responses
                    api_key=settings.google_api_key,
                    response_format={"type": "json_object"},  # Force JSON output
                )
                return response
            except RateLimitError as e:
                if attempt == max_retries:
                    logger.error(f"Rate limit exceeded after {max_retries} retries: {e}")
                    raise

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit on router agent (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # If response_format not supported, try without it
                if "response_format" in str(e):
                    logger.warning("response_format not supported, falling back to standard completion")
                    return await acompletion(
                        model=settings.default_chat_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=500,
                        api_key=settings.google_api_key,
                    )
                raise

    async def route(
        self,
        query: str,
        collection_context: str = "",
        conversation_history: list[ConversationTurn] | None = None,
        conversation_summary: str | None = None,
    ) -> RouteDecision:
        """
        Analyze query and determine optimal search method.

        Args:
            query: The user's search query
            collection_context: Description of the collection's content
            conversation_history: Recent conversation turns
            conversation_summary: Compressed summary of earlier turns

        Returns:
            RouteDecision with method, confidence, reasoning, and rewritten_query
        """
        history_block = self._format_history_block(
            conversation_history or [],
            conversation_summary,
        )

        prompt = self.prompt_template.format(
            query=query,
            collection_context=collection_context or "No collection context available",
            conversation_history_block=history_block,
        )

        try:
            response = await self._call_llm(prompt)
            content = response.choices[0].message.content

            # Log the raw response for debugging
            logger.debug(f"Router LLM raw response: {content}")

            if not content or not content.strip():
                logger.warning("Router received empty response from LLM")
                return RouteDecision(
                    method="local",
                    confidence=0.3,
                    reasoning="Default to LOCAL - empty LLM response",
                    rewritten_query=query,
                )

            # Try to extract JSON if wrapped in markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                # Extract JSON from markdown code block
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace("```json", "").replace("```", "").strip()

            # Parse JSON response
            decision = json.loads(content)

            method = decision.get("method", "local").lower()
            if method not in ("local", "global", "tog", "drift", "web"):
                logger.warning(f"Invalid method '{method}' returned, defaulting to 'local'")
                method = "local"

            rewritten_query = decision.get("rewritten_query") or query

            return RouteDecision(
                method=method,
                confidence=float(decision.get("confidence", 0.5)),
                reasoning=decision.get("reasoning", "No reasoning provided"),
                rewritten_query=rewritten_query,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse router response. Error: {e}. Content: {content[:200] if 'content' in locals() else 'N/A'}")
            return RouteDecision(
                method="local",
                confidence=0.5,
                reasoning=f"Default to LOCAL due to parse error: {e}",
                rewritten_query=query,
            )
        except Exception as e:
            logger.error(f"Router agent error: {e}", exc_info=True)
            return RouteDecision(
                method="local",
                confidence=0.3,
                reasoning=f"Default to LOCAL due to error: {e}",
                rewritten_query=query,
            )


# Global router agent instance
router_agent = RouterAgent()
