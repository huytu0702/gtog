"""Web Search service using Tavily API."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from ..config import settings
from ..models.events import Source

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Result of web search with LLM synthesis."""

    response: str
    sources: list[Source] = field(default_factory=list)


class WebSearchService:
    """Service for web search using Tavily API with LLM synthesis."""

    def __init__(self):
        """Initialize the web search service."""
        self.tavily = AsyncTavilyClient(api_key=settings.tavily_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the synthesis prompt template."""
        prompt_path = (
            Path(__file__).parent.parent.parent / "prompts" / "web_synthesis_prompt.txt"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        return self._default_prompt()

    def _default_prompt(self) -> str:
        """Return default prompt if file not found."""
        return """Synthesize these web search results to answer the query.
Query: {query}
Results: {search_results}
Include [N] citations."""

    async def _call_llm(self, prompt: str):
        """Call OpenAI API for synthesis."""
        return await self.openai.chat.completions.create(
            model=settings.default_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

    async def search(self, query: str) -> WebSearchResult:
        """
        Perform web search and synthesize results.

        Args:
            query: The search query

        Returns:
            WebSearchResult with synthesized response and sources
        """
        try:
            # Call Tavily API
            tavily_response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )

            results = tavily_response.get("results", [])

            if not results:
                return WebSearchResult(
                    response="No relevant web search results found for your query.",
                    sources=[],
                )

            # Build sources list
            sources = [
                Source(
                    id=i + 1,
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    text_unit_id=None,
                )
                for i, r in enumerate(results)
            ]

            # Format results for LLM
            formatted_results = "\n\n".join([
                f"[{i + 1}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
                for i, r in enumerate(results)
            ])

            # Synthesize with LLM
            prompt = self.prompt_template.format(
                query=query, search_results=formatted_results
            )

            llm_response = await self._call_llm(prompt)
            synthesized = llm_response.choices[0].message.content or ""

            return WebSearchResult(response=synthesized, sources=sources)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return WebSearchResult(
                response=f"Error performing web search: {e}", sources=[]
            )

    async def search_streaming(self, query: str) -> AsyncIterator[str]:
        """
        Perform web search with streaming response.

        Args:
            query: The search query

        Yields:
            Chunks of the synthesized response
        """
        try:
            # Call Tavily API (non-streaming)
            tavily_response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )

            results = tavily_response.get("results", [])

            if not results:
                yield "No relevant web search results found for your query."
                return

            # Format results for LLM
            formatted_results = "\n\n".join([
                f"[{i + 1}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
                for i, r in enumerate(results)
            ])

            prompt = self.prompt_template.format(
                query=query, search_results=formatted_results
            )

            # Stream LLM response
            stream = await self.openai.chat.completions.create(
                model=settings.default_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Web search streaming error: {e}")
            yield f"Error performing web search: {e}"


# Global web search service instance
web_search_service = WebSearchService()
