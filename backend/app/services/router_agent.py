"""Router Agent service for intelligent query routing."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from litellm import acompletion
from litellm.exceptions import RateLimitError

from ..config import settings
from ..models.schemas import ConversationTurn

logger = logging.getLogger(__name__)

SearchMethodType = Literal["local", "global", "tog", "drift"]

RECENT_TURNS_IN_PROMPT = 3  # user turns to include in prompt (after summary)
_LLM_MAX_TOKENS = 500
_LLM_TEMPERATURE = 0.1
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 1.0
_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"\b(it|its|he|him|his|she|her|hers|they|them|their|theirs|this|that|these|those)\b",
    re.IGNORECASE,
)


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
Methods: local, global, tog, drift
Query: {query}
Collection: {collection_context}
{conversation_history_block}"""

    def _should_preserve_standalone_query(
        self,
        query: str,
        conversation_history: list[ConversationTurn] | None,
        conversation_summary: str | None,
    ) -> bool:
        has_context = bool(conversation_history or conversation_summary)
        return not has_context and not _CONTEXT_REFERENCE_PATTERN.search(query)

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
            sections.append(f"Past conversation summary:\n{conversation_summary}")

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
            label = (
                "Recent conversation (most recent last):"
                if conversation_summary
                else "Conversation history (most recent last):"
            )
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
                        content = (
                            turn.content[:300] + "..."
                            if len(turn.content) > 300
                            else turn.content
                        )
                        lines.append(f"[Assistant] {content}")
                except Exception:
                    logger.warning(
                        "Skipping malformed conversation turn", exc_info=True
                    )
                    continue

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    async def _call_llm(self, prompt: str) -> object:
        """Call LLM API using litellm with exponential backoff on rate limits."""
        max_retries = _LLM_MAX_RETRIES
        base_delay = _LLM_RETRY_BASE_DELAY

        for attempt in range(max_retries + 1):
            try:
                response = await acompletion(
                    model=settings.query_chat_model_litellm,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=_LLM_TEMPERATURE,
                    max_tokens=_LLM_MAX_TOKENS,  # Increased for more complete responses
                    api_key=settings.query_chat_model_api_key,
                    response_format={"type": "json_object"},  # Force JSON output
                )
                return response
            except RateLimitError as e:
                if attempt == max_retries:
                    logger.error(
                        "Rate limit exceeded after %s retries: %s", max_retries, e
                    )
                    raise

                delay = base_delay * (2**attempt)
                logger.warning(
                    "Rate limit hit on router agent (attempt %d/%d). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # If response_format not supported, try without it
                if "response_format" in str(e):
                    logger.warning(
                        "response_format not supported, falling back to standard completion"
                    )
                    return await acompletion(
                        model=settings.query_chat_model_litellm,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=_LLM_TEMPERATURE,
                        max_tokens=_LLM_MAX_TOKENS,
                        api_key=settings.query_chat_model_api_key,
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

        Returns
        -------
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

        content = ""
        try:
            response: Any = await self._call_llm(prompt)
            content = response.choices[0].message.content

            # Log the raw response for debugging
            logger.debug("Router LLM raw response: %s", content)

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
            if method not in ("local", "global", "tog", "drift"):
                logger.warning(
                    "Invalid method '%s' returned, defaulting to 'local'", method
                )
                method = "local"

            rewritten_query = decision.get("rewritten_query") or query
            if self._should_preserve_standalone_query(
                query,
                conversation_history,
                conversation_summary,
            ):
                rewritten_query = query

            return RouteDecision(
                method=method,
                confidence=float(decision.get("confidence", 0.5)),
                reasoning=decision.get("reasoning", "No reasoning provided"),
                rewritten_query=rewritten_query,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raw = content[:200] if "content" in dir() else "N/A"
            logger.warning(
                "Failed to parse router response. Error: %s. Content: %s", e, raw
            )
            return RouteDecision(
                method="local",
                confidence=0.5,
                reasoning=f"Default to LOCAL due to parse error: {e}",
                rewritten_query=query,
            )
        except Exception as e:
            logger.error("Router agent error: %s", e, exc_info=True)
            return RouteDecision(
                method="local",
                confidence=0.3,
                reasoning=f"Default to LOCAL due to error: {e}",
                rewritten_query=query,
            )


# Global router agent instance
router_agent = RouterAgent()
