"""Router Agent service for intelligent query routing."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI

from ..config import settings

logger = logging.getLogger(__name__)

SearchMethodType = Literal["local", "global", "tog", "drift", "web"]


@dataclass
class RouteDecision:
    """Result of router agent decision."""

    method: SearchMethodType
    confidence: float
    reasoning: str


class RouterAgent:
    """Agent that routes queries to the optimal search method."""

    def __init__(self):
        """Initialize the router agent."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
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
        return """Analyze the query and return JSON with method, confidence, reasoning.
Methods: local, global, tog, drift, web
Query: {query}
Collection: {collection_context}"""

    async def _call_llm(self, prompt: str):
        """Call OpenAI API."""
        return await self.client.chat.completions.create(
            model=settings.default_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

    async def route(self, query: str, collection_context: str = "") -> RouteDecision:
        """
        Analyze query and determine optimal search method.

        Args:
            query: The user's search query
            collection_context: Description of the collection's content

        Returns:
            RouteDecision with method, confidence, and reasoning
        """
        prompt = self.prompt_template.format(
            query=query,
            collection_context=collection_context or "No collection context available",
        )

        try:
            response = await self._call_llm(prompt)
            content = response.choices[0].message.content

            # Parse JSON response
            decision = json.loads(content)

            method = decision.get("method", "local").lower()
            if method not in ("local", "global", "tog", "drift", "web"):
                method = "local"

            return RouteDecision(
                method=method,
                confidence=float(decision.get("confidence", 0.5)),
                reasoning=decision.get("reasoning", "No reasoning provided"),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse router response: {e}")
            return RouteDecision(
                method="local",
                confidence=0.5,
                reasoning=f"Default to LOCAL due to parse error: {e}",
            )
        except Exception as e:
            logger.error(f"Router agent error: {e}")
            return RouteDecision(
                method="local",
                confidence=0.3,
                reasoning=f"Default to LOCAL due to error: {e}",
            )


# Global router agent instance
router_agent = RouterAgent()
