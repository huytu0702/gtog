# Multi-Agent Search System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Router Agent and Web Search fallback to intelligently route queries to optimal search method.

**Architecture:** A Router Agent (LLM-based) analyzes queries and routes to one of five methods: local, global, tog, drift, or web. Web search uses Tavily API with LLM synthesis. All implementation is backend-only to keep graphrag/ pure as library.

**Tech Stack:** FastAPI, Pydantic, Tavily API, OpenAI (for LLM routing/synthesis), SSE streaming

---

## Task 1: Add Tavily Dependency

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Add tavily-python package**

Edit `backend/requirements.txt` to add:

```
tavily-python==0.3.9
sse-starlette==1.8.2
```

**Step 2: Verify the change**

Run: `type backend\requirements.txt`
Expected: File contains tavily-python and sse-starlette

**Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "feat: add tavily-python and sse-starlette dependencies"
```

---

## Task 2: Add TAVILY_API_KEY to Config

**Files:**
- Modify: `backend/app/config.py`

**Step 1: Read current config**

Read `backend/app/config.py` to understand current structure.

**Step 2: Add TAVILY_API_KEY setting**

Edit `backend/app/config.py` to add after line 18 (`openai_api_key`):

```python
    tavily_api_key: str = ""
```

**Step 3: Verify the change**

Run: `grep -n "tavily" backend/app/config.py`
Expected: Line with `tavily_api_key: str = ""`

**Step 4: Commit**

```bash
git add backend/app/config.py
git commit -m "feat: add TAVILY_API_KEY configuration"
```

---

## Task 3: Add WEB to SearchMethod Enum

**Files:**
- Modify: `backend/app/models/enums.py`

**Step 1: Add WEB enum value**

Edit `backend/app/models/enums.py` to add after line 12 (`DRIFT = "drift"`):

```python
    WEB = "web"
```

**Step 2: Verify the change**

Run: `type backend\app\models\enums.py`
Expected: SearchMethod includes WEB = "web"

**Step 3: Commit**

```bash
git add backend/app/models/enums.py
git commit -m "feat: add WEB to SearchMethod enum"
```

---

## Task 4: Create SSE Event Models

**Files:**
- Create: `backend/app/models/events.py`
- Modify: `backend/app/models/__init__.py`

**Step 1: Create events.py**

Create `backend/app/models/events.py`:

```python
"""SSE event models for streaming search responses."""

from typing import Any, Literal, Optional
from pydantic import BaseModel


class StatusEvent(BaseModel):
    """Status update event during search."""

    event: Literal["status"] = "status"
    step: Literal["routing", "routed", "searching", "generating"]
    message: str
    method: Optional[str] = None


class ContentEvent(BaseModel):
    """Content chunk event for streaming response."""

    event: Literal["content"] = "content"
    delta: str


class Source(BaseModel):
    """Citation source."""

    id: int
    title: str
    url: Optional[str] = None
    text_unit_id: Optional[str] = None


class DoneEvent(BaseModel):
    """Completion event with final metadata."""

    event: Literal["done"] = "done"
    method_used: str
    sources: list[Source] = []
    router_reasoning: Optional[str] = None


class ErrorEvent(BaseModel):
    """Error event."""

    event: Literal["error"] = "error"
    message: str
    code: Optional[str] = None
```

**Step 2: Export from __init__.py**

Edit `backend/app/models/__init__.py` to add import after line 18:

```python
from .events import (
    ContentEvent,
    DoneEvent,
    ErrorEvent,
    Source,
    StatusEvent,
)
```

And add to `__all__` after line 42 (`"HealthResponse",`):

```python
    # SSE Events
    "StatusEvent",
    "ContentEvent",
    "DoneEvent",
    "ErrorEvent",
    "Source",
```

**Step 3: Verify the change**

Run: `python -c "from backend.app.models import StatusEvent, Source; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add backend/app/models/events.py backend/app/models/__init__.py
git commit -m "feat: add SSE event models for streaming responses"
```

---

## Task 5: Create Agent Search Request/Response Models

**Files:**
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/app/models/__init__.py`

**Step 1: Add AgentSearchRequest model**

Edit `backend/app/models/schemas.py` to add after ToGSearchRequest class (after line 125):

```python


class AgentSearchRequest(BaseModel):
    """Request model for agent-routed search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True


class WebSearchRequest(BaseModel):
    """Request model for direct web search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True


class AgentSearchResponse(BaseModel):
    """Response model for agent-routed search."""

    method_used: str
    router_reasoning: str
    response: str
    sources: list = Field(default_factory=list)
```

**Step 2: Export from __init__.py**

Edit `backend/app/models/__init__.py` to add imports after line 17 (`ToGSearchRequest,`):

```python
    AgentSearchRequest,
    AgentSearchResponse,
    WebSearchRequest,
```

And add to `__all__` after line 41 (`"SearchResponse",`):

```python
    "AgentSearchRequest",
    "AgentSearchResponse",
    "WebSearchRequest",
```

**Step 3: Verify the change**

Run: `python -c "from backend.app.models import AgentSearchRequest, AgentSearchResponse, WebSearchRequest; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add backend/app/models/schemas.py backend/app/models/__init__.py
git commit -m "feat: add AgentSearchRequest and WebSearchRequest models"
```

---

## Task 6: Create Router Prompt File

**Files:**
- Create: `backend/prompts/router_prompt.txt`

**Step 1: Create prompts directory and router prompt**

Create `backend/prompts/router_prompt.txt`:

```
You are a query routing assistant. Analyze the user's query and determine which search method is most appropriate.

Available search methods:
- GLOBAL: Questions requiring understanding of the dataset as a whole. Overview, trends, summaries across entire corpus.
- LOCAL: Questions about specific entities mentioned in documents. Focused queries on particular topics/names/concepts.
- TOG: Questions about relationships between entities. Multi-hop reasoning through entity connections.
- DRIFT: Local search + community context for broader variety. Expands query into detailed follow-up questions.
- WEB: External information not in documents. Real-time/current events, topics outside indexed data.

Collection context:
{collection_context}

User query: {query}

Respond with a JSON object:
{
  "method": "local|global|tog|drift|web",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this method was chosen"
}

Important:
- If the query asks about current events, news, or real-time information, choose WEB.
- If the query asks about relationships or connections between concepts, choose TOG.
- If the query asks for broad overviews or trends across many topics, choose GLOBAL.
- If the query asks about specific entities or focused topics, choose LOCAL.
- Use DRIFT only when LOCAL might miss important related context.
- Default to LOCAL if uncertain between LOCAL and other methods.
```

**Step 2: Verify the file**

Run: `type backend\prompts\router_prompt.txt`
Expected: Router prompt content displayed

**Step 3: Commit**

```bash
git add backend/prompts/router_prompt.txt
git commit -m "feat: add router agent prompt template"
```

---

## Task 7: Create Web Synthesis Prompt File

**Files:**
- Create: `backend/prompts/web_synthesis_prompt.txt`

**Step 1: Create web synthesis prompt**

Create `backend/prompts/web_synthesis_prompt.txt`:

```
You are a helpful assistant that synthesizes web search results into a clear, accurate answer.

User query: {query}

Web search results:
{search_results}

Instructions:
1. Synthesize the information from the search results to answer the query.
2. Include inline citations using [N] format where N is the source number.
3. Only use information from the provided search results.
4. If the results don't contain enough information, say so honestly.
5. Keep the response focused and relevant to the query.

Provide your synthesized answer:
```

**Step 2: Verify the file**

Run: `type backend\prompts\web_synthesis_prompt.txt`
Expected: Web synthesis prompt content displayed

**Step 3: Commit**

```bash
git add backend/prompts/web_synthesis_prompt.txt
git commit -m "feat: add web search synthesis prompt template"
```

---

## Task 8: Create Router Agent Service

**Files:**
- Create: `backend/app/services/router_agent.py`
- Test: `backend/tests/unit/test_router_agent.py`

**Step 1: Write failing test**

Create `backend/tests/unit/test_router_agent.py`:

```python
"""Tests for Router Agent service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.services.router_agent import RouterAgent, RouteDecision


class TestRouterAgent:
    """Test RouterAgent class."""

    @pytest.fixture
    def router_agent(self):
        """Create RouterAgent instance with mocked LLM."""
        return RouterAgent()

    @pytest.mark.asyncio
    async def test_route_returns_route_decision(self, router_agent):
        """route() should return a RouteDecision object."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"method": "local", "confidence": 0.85, "reasoning": "Query asks about specific entity"}'

        with patch.object(router_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route("What is chamomile used for?", "herbs collection")

            assert isinstance(result, RouteDecision)
            assert result.method == "local"
            assert result.confidence == 0.85
            assert "specific entity" in result.reasoning

    @pytest.mark.asyncio
    async def test_route_defaults_to_local_on_parse_error(self, router_agent):
        """route() should default to LOCAL if LLM response can't be parsed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'invalid json'

        with patch.object(router_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route("test query", "test collection")

            assert result.method == "local"
            assert "default" in result.reasoning.lower() or "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_identifies_web_search_query(self, router_agent):
        """route() should identify queries needing web search."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"method": "web", "confidence": 0.92, "reasoning": "Query asks about current FDA regulations"}'

        with patch.object(router_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await router_agent.route("What are the latest FDA regulations?", "herbs collection")

            assert result.method == "web"
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_router_agent.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write RouterAgent implementation**

Create `backend/app/services/router_agent.py`:

```python
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
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "router_prompt.txt"
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

    async def route(
        self,
        query: str,
        collection_context: str = ""
    ) -> RouteDecision:
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
            collection_context=collection_context or "No collection context available"
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
                reasoning=decision.get("reasoning", "No reasoning provided")
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse router response: {e}")
            return RouteDecision(
                method="local",
                confidence=0.5,
                reasoning=f"Default to LOCAL due to parse error: {e}"
            )
        except Exception as e:
            logger.error(f"Router agent error: {e}")
            return RouteDecision(
                method="local",
                confidence=0.3,
                reasoning=f"Default to LOCAL due to error: {e}"
            )


# Global router agent instance
router_agent = RouterAgent()
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/test_router_agent.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add backend/app/services/router_agent.py backend/tests/unit/test_router_agent.py
git commit -m "feat: implement RouterAgent service with tests"
```

---

## Task 9: Create Web Search Service

**Files:**
- Create: `backend/app/services/web_search.py`
- Test: `backend/tests/unit/test_web_search.py`

**Step 1: Write failing test**

Create `backend/tests/unit/test_web_search.py`:

```python
"""Tests for Web Search service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.services.web_search import WebSearchService, WebSearchResult


class TestWebSearchService:
    """Test WebSearchService class."""

    @pytest.fixture
    def web_search_service(self):
        """Create WebSearchService instance with mocked clients."""
        with patch('backend.app.services.web_search.TavilyClient'):
            return WebSearchService()

    @pytest.mark.asyncio
    async def test_search_returns_web_search_result(self, web_search_service):
        """search() should return a WebSearchResult object."""
        # Mock Tavily response
        mock_tavily_result = {
            "results": [
                {"title": "Test Article", "url": "https://example.com", "content": "Test content"}
            ]
        }
        web_search_service.tavily.search = MagicMock(return_value=mock_tavily_result)

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "Synthesized response [1]"

        with patch.object(web_search_service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            result = await web_search_service.search("test query")

            assert isinstance(result, WebSearchResult)
            assert "Synthesized response" in result.response
            assert len(result.sources) >= 1

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, web_search_service):
        """search() should handle empty Tavily results gracefully."""
        web_search_service.tavily.search = MagicMock(return_value={"results": []})

        result = await web_search_service.search("obscure query")

        assert isinstance(result, WebSearchResult)
        assert "no relevant results" in result.response.lower() or result.response != ""

    @pytest.mark.asyncio
    async def test_search_handles_tavily_error(self, web_search_service):
        """search() should handle Tavily API errors gracefully."""
        web_search_service.tavily.search = MagicMock(side_effect=Exception("API Error"))

        result = await web_search_service.search("test query")

        assert isinstance(result, WebSearchResult)
        assert "error" in result.response.lower() or result.sources == []
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_web_search.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write WebSearchService implementation**

Create `backend/app/services/web_search.py`:

```python
"""Web Search service using Tavily API."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from openai import AsyncOpenAI
from tavily import TavilyClient

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
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the synthesis prompt template."""
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "web_synthesis_prompt.txt"
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
                    sources=[]
                )

            # Build sources list
            sources = [
                Source(
                    id=i + 1,
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    text_unit_id=None
                )
                for i, r in enumerate(results)
            ]

            # Format results for LLM
            formatted_results = "\n\n".join([
                f"[{i+1}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
                for i, r in enumerate(results)
            ])

            # Synthesize with LLM
            prompt = self.prompt_template.format(
                query=query,
                search_results=formatted_results
            )

            llm_response = await self._call_llm(prompt)
            synthesized = llm_response.choices[0].message.content

            return WebSearchResult(
                response=synthesized,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return WebSearchResult(
                response=f"Error performing web search: {e}",
                sources=[]
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
                f"[{i+1}] {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
                for i, r in enumerate(results)
            ])

            prompt = self.prompt_template.format(
                query=query,
                search_results=formatted_results
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
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/test_web_search.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add backend/app/services/web_search.py backend/tests/unit/test_web_search.py
git commit -m "feat: implement WebSearchService with Tavily integration"
```

---

## Task 10: Export New Services from __init__.py

**Files:**
- Modify: `backend/app/services/__init__.py`

**Step 1: Add new service exports**

Edit `backend/app/services/__init__.py` to add imports after line 5:

```python
from .router_agent import router_agent, RouterAgent, RouteDecision
from .web_search import web_search_service, WebSearchService, WebSearchResult
```

And update `__all__` to:

```python
__all__ = [
    "storage_service",
    "StorageService",
    "indexing_service",
    "IndexingService",
    "query_service",
    "QueryService",
    "router_agent",
    "RouterAgent",
    "RouteDecision",
    "web_search_service",
    "WebSearchService",
    "WebSearchResult",
]
```

**Step 2: Verify the change**

Run: `python -c "from backend.app.services import router_agent, web_search_service; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add backend/app/services/__init__.py
git commit -m "feat: export RouterAgent and WebSearchService from services"
```

---

## Task 11: Add Agent Search Endpoint

**Files:**
- Modify: `backend/app/routers/search.py`
- Test: `backend/tests/unit/test_search_router.py`

**Step 1: Write failing test**

Create `backend/tests/unit/test_search_router.py`:

```python
"""Tests for search router endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app


class TestAgentSearchEndpoint:
    """Test /agent search endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_agent_search_returns_200(self, client):
        """POST /agent should return 200 with valid response."""
        # Mock router agent
        mock_route_decision = MagicMock()
        mock_route_decision.method = "local"
        mock_route_decision.confidence = 0.85
        mock_route_decision.reasoning = "Specific entity query"

        # Mock query service
        mock_search_response = MagicMock()
        mock_search_response.query = "test"
        mock_search_response.response = "Test response"
        mock_search_response.context_data = None
        mock_search_response.method = "local"

        with patch('backend.app.routers.search.router_agent') as mock_router:
            with patch('backend.app.routers.search.query_service') as mock_query:
                mock_router.route = AsyncMock(return_value=mock_route_decision)
                mock_query.local_search = AsyncMock(return_value=mock_search_response)

                response = client.post(
                    "/api/collections/test-collection/search/agent",
                    json={"query": "What is chamomile?", "stream": False}
                )

                assert response.status_code == 200
                data = response.json()
                assert "method_used" in data
                assert "router_reasoning" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_search_router.py::TestAgentSearchEndpoint::test_agent_search_returns_200 -v`
Expected: FAIL (endpoint doesn't exist)

**Step 3: Add agent search endpoint**

Edit `backend/app/routers/search.py`. Add imports at top after line 12:

```python
from ..services import router_agent, web_search_service
from ..models import AgentSearchRequest, AgentSearchResponse
```

Add endpoint at end of file after drift_search:

```python


@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    The router agent analyzes the query and selects the optimal search method.
    """
    try:
        # Get collection context (simplified - just use collection_id for now)
        collection_context = f"Collection: {collection_id}"

        # Route the query
        route_decision = await router_agent.route(request.query, collection_context)
        logger.info(f"Router decision: {route_decision.method} (confidence: {route_decision.confidence})")

        # Execute the appropriate search
        if route_decision.method == "web":
            from ..services import web_search_service
            result = await web_search_service.search(request.query)
            return AgentSearchResponse(
                method_used="web",
                router_reasoning=route_decision.reasoning,
                response=result.response,
                sources=[s.model_dump() for s in result.sources],
            )

        # For GraphRAG methods, call the appropriate service
        if route_decision.method == "global":
            result = await query_service.global_search(
                collection_id=collection_id,
                query=request.query,
            )
        elif route_decision.method == "tog":
            result = await query_service.tog_search(
                collection_id=collection_id,
                query=request.query,
            )
        elif route_decision.method == "drift":
            result = await query_service.drift_search(
                collection_id=collection_id,
                query=request.query,
            )
        else:  # default to local
            result = await query_service.local_search(
                collection_id=collection_id,
                query=request.query,
            )

        return AgentSearchResponse(
            method_used=route_decision.method,
            router_reasoning=route_decision.reasoning,
            response=result.response,
            sources=[],
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Error performing agent search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/test_search_router.py::TestAgentSearchEndpoint::test_agent_search_returns_200 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/routers/search.py backend/tests/unit/test_search_router.py
git commit -m "feat: add /agent search endpoint with router integration"
```

---

## Task 12: Add Web Search Endpoint

**Files:**
- Modify: `backend/app/routers/search.py`
- Test: `backend/tests/unit/test_search_router.py`

**Step 1: Write failing test**

Add to `backend/tests/unit/test_search_router.py`:

```python


class TestWebSearchEndpoint:
    """Test /web search endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_web_search_returns_200(self, client):
        """POST /web should return 200 with valid response."""
        mock_result = MagicMock()
        mock_result.response = "Web search result"
        mock_result.sources = []

        with patch('backend.app.routers.search.web_search_service') as mock_web:
            mock_web.search = AsyncMock(return_value=mock_result)

            response = client.post(
                "/api/collections/test-collection/search/web",
                json={"query": "What are latest FDA regulations?", "stream": False}
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_search_router.py::TestWebSearchEndpoint::test_web_search_returns_200 -v`
Expected: FAIL (endpoint doesn't exist)

**Step 3: Add web search endpoint**

Edit `backend/app/routers/search.py`. Add import at top if not already present:

```python
from ..models import WebSearchRequest
```

Add endpoint at end of file:

```python


@router.post("/web")
async def web_search(collection_id: str, request: WebSearchRequest):
    """
    Perform a direct web search, bypassing the router agent.

    Uses Tavily API for web search with LLM synthesis.
    """
    try:
        result = await web_search_service.search(request.query)

        return {
            "query": request.query,
            "response": result.response,
            "sources": [s.model_dump() for s in result.sources],
            "method": "web",
        }

    except Exception as e:
        logger.exception("Error performing web search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest backend/tests/unit/test_search_router.py::TestWebSearchEndpoint::test_web_search_returns_200 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/routers/search.py backend/tests/unit/test_search_router.py
git commit -m "feat: add /web direct search endpoint"
```

---

## Task 13: Add Streaming Agent Search Endpoint

**Files:**
- Modify: `backend/app/routers/search.py`

**Step 1: Add SSE streaming imports**

Edit `backend/app/routers/search.py` to add import at top:

```python
import json
from sse_starlette.sse import EventSourceResponse
```

**Step 2: Add streaming endpoint**

Add after the `/agent` endpoint:

```python


@router.post("/agent/stream")
async def agent_search_stream(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search with SSE streaming.

    Streams status updates and response content.
    """
    async def event_generator():
        try:
            # Send routing status
            yield {
                "event": "status",
                "data": json.dumps({"step": "routing", "message": "Analyzing query..."})
            }

            # Route the query
            collection_context = f"Collection: {collection_id}"
            route_decision = await router_agent.route(request.query, collection_context)

            # Send routed status
            yield {
                "event": "status",
                "data": json.dumps({
                    "step": "routed",
                    "method": route_decision.method,
                    "message": f"Using {route_decision.method.upper()} search"
                })
            }

            # Send searching status
            yield {
                "event": "status",
                "data": json.dumps({"step": "searching", "message": "Searching..."})
            }

            # Execute search
            if route_decision.method == "web":
                async for chunk in web_search_service.search_streaming(request.query):
                    yield {
                        "event": "content",
                        "data": json.dumps({"delta": chunk})
                    }
                sources = []
            else:
                # For GraphRAG methods, get full response (non-streaming for now)
                if route_decision.method == "global":
                    result = await query_service.global_search(collection_id, request.query)
                elif route_decision.method == "tog":
                    result = await query_service.tog_search(collection_id, request.query)
                elif route_decision.method == "drift":
                    result = await query_service.drift_search(collection_id, request.query)
                else:
                    result = await query_service.local_search(collection_id, request.query)

                # Stream the response in chunks
                chunk_size = 50
                for i in range(0, len(result.response), chunk_size):
                    yield {
                        "event": "content",
                        "data": json.dumps({"delta": result.response[i:i+chunk_size]})
                    }
                sources = []

            # Send done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "method_used": route_decision.method,
                    "sources": sources,
                    "router_reasoning": route_decision.reasoning
                })
            }

        except Exception as e:
            logger.exception("Error in streaming agent search")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())
```

**Step 3: Verify endpoint works**

Run: `python -c "from backend.app.routers.search import router; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add backend/app/routers/search.py
git commit -m "feat: add streaming /agent/stream endpoint with SSE"
```

---

## Task 14: Create Backend Tests Directory Structure

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/unit/__init__.py`
- Create: `backend/tests/conftest.py`

**Step 1: Create test directory structure**

Create `backend/tests/__init__.py`:

```python
"""Backend tests package."""
```

Create `backend/tests/unit/__init__.py`:

```python
"""Unit tests package."""
```

Create `backend/tests/conftest.py`:

```python
"""Pytest configuration for backend tests."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch('backend.app.config.settings') as mock:
        mock.openai_api_key = "test-key"
        mock.tavily_api_key = "test-tavily-key"
        mock.default_chat_model = "gpt-4o-mini"
        mock.collections_dir = MagicMock()
        yield mock
```

**Step 2: Verify structure**

Run: `dir /b backend\tests`
Expected: __init__.py, unit, conftest.py

**Step 3: Commit**

```bash
git add backend/tests/
git commit -m "feat: add backend test directory structure with conftest"
```

---

## Task 15: Run All Tests and Verify

**Files:**
- None (verification only)

**Step 1: Run all backend tests**

Run: `pytest backend/tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run type check**

Run: `pyright backend/app/`
Expected: No errors (or only pre-existing ones)

**Step 3: Run linting**

Run: `ruff check backend/app/ --fix`
Expected: No errors

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address linting and type issues"
```

---

## Task 16: Integration Test - End to End

**Files:**
- Create: `backend/tests/integration/test_agent_search.py`

**Step 1: Create integration test**

Create `backend/tests/integration/__init__.py`:

```python
"""Integration tests package."""
```

Create `backend/tests/integration/test_agent_search.py`:

```python
"""Integration tests for agent search."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from backend.app.main import app


class TestAgentSearchIntegration:
    """Integration tests for agent search flow."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_full_agent_search_flow(self, client):
        """Test complete agent search from request to response."""
        # This test verifies the full flow works end-to-end
        # with mocked external services

        mock_route = MagicMock()
        mock_route.method = "web"
        mock_route.confidence = 0.9
        mock_route.reasoning = "External information needed"

        mock_web_result = MagicMock()
        mock_web_result.response = "The FDA regulations..."
        mock_web_result.sources = []

        with patch('backend.app.services.router_agent.RouterAgent.route', new_callable=AsyncMock) as mock_router:
            with patch('backend.app.services.web_search.WebSearchService.search', new_callable=AsyncMock) as mock_web:
                mock_router.return_value = mock_route
                mock_web.return_value = mock_web_result

                response = client.post(
                    "/api/collections/test/search/agent",
                    json={"query": "What are latest FDA regulations?", "stream": False}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["method_used"] == "web"
                assert "FDA" in data["response"] or "regulations" in data["response"].lower()
```

**Step 2: Run integration test**

Run: `pytest backend/tests/integration/test_agent_search.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add backend/tests/integration/
git commit -m "feat: add integration test for agent search flow"
```

---

## Task 17: Update Design Document Status

**Files:**
- Modify: `docs/plans/2026-02-05-multi-agent-web-search-design.md`

**Step 1: Update status to Implemented**

Edit `docs/plans/2026-02-05-multi-agent-web-search-design.md` line 4:

Change:
```
**Status:** Approved
```

To:
```
**Status:** Implemented
```

**Step 2: Commit**

```bash
git add docs/plans/2026-02-05-multi-agent-web-search-design.md
git commit -m "docs: mark multi-agent web search design as implemented"
```

---

## Summary

This plan implements the Multi-Agent Search System with:

1. **RouterAgent** - LLM-based query routing to optimal search method
2. **WebSearchService** - Tavily integration with LLM synthesis
3. **SSE Streaming** - Real-time status updates and content streaming
4. **New Endpoints**:
   - `POST /api/collections/{id}/search/agent` - Auto-routed search
   - `POST /api/collections/{id}/search/agent/stream` - Streaming version
   - `POST /api/collections/{id}/search/web` - Direct web search

Total: 17 tasks, each with TDD approach (test first, implement, commit).
