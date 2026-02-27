# Router Agent Conversation History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conversation history to the router agent so it resolves pronouns and implicit references in multi-turn queries, using a single LLM call that both rewrites the query and selects the search method.

**Architecture:** The frontend carries session state as a list of `ConversationTurn` objects and sends it with each request. The router agent formats the last 5 turns into the prompt and asks the LLM to produce both a standalone `rewritten_query` and a routing decision in one JSON response. The rewritten query is used for all downstream search calls and returned in the response so the frontend can store it.

**Tech Stack:** FastAPI, Pydantic v2, litellm, pytest, pytest-asyncio

---

## Task 1: Add `ConversationTurn` model and update schemas

**Files:**
- Modify: `backend/app/models/schemas.py`

**Step 1: Write the failing test**

Add to `backend/tests/unit/test_router_agent.py` (at the top of the file, after imports):

```python
from backend.app.models.schemas import ConversationTurn, AgentSearchRequest, AgentSearchResponse
```

Add this test class after the existing `TestRouterAgent` class:

```python
class TestConversationTurnModel:
    """Test ConversationTurn Pydantic model."""

    def test_user_turn_with_metadata(self):
        turn = ConversationTurn(
            role="user",
            content="Who directed it?",
            rewritten_query="Who directed Inception?",
            method_used="local",
        )
        assert turn.role == "user"
        assert turn.rewritten_query == "Who directed Inception?"
        assert turn.method_used == "local"

    def test_assistant_turn_no_metadata(self):
        turn = ConversationTurn(
            role="assistant",
            content="Christopher Nolan directed Inception.",
        )
        assert turn.role == "assistant"
        assert turn.rewritten_query is None
        assert turn.method_used is None

    def test_agent_search_request_has_conversation_history(self):
        req = AgentSearchRequest(
            query="Who directed it?",
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ],
        )
        assert len(req.conversation_history) == 2

    def test_agent_search_request_empty_history_by_default(self):
        req = AgentSearchRequest(query="hello")
        assert req.conversation_history == []

    def test_agent_search_response_has_rewritten_query(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="specific entity",
            rewritten_query="Who directed Inception?",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query == "Who directed Inception?"

    def test_agent_search_response_rewritten_query_optional(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="specific entity",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query is None
```

**Step 2: Run test to verify it fails**

```bash
cd F:/KL/gtog
pytest backend/tests/unit/test_router_agent.py::TestConversationTurnModel -v
```

Expected: FAIL — `ConversationTurn` does not exist yet.

**Step 3: Implement the models**

In `backend/app/models/schemas.py`, add after the imports block (after `from typing import Any, Dict, List, Optional`, add `Literal`):

```python
from typing import Any, Dict, List, Literal, Optional
```

Add `ConversationTurn` before the `AgentSearchRequest` class (around line 136):

```python
class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: Literal["user", "assistant"]
    content: str
    rewritten_query: str | None = None  # user turns only
    method_used: str | None = None      # user turns only
```

Update `AgentSearchRequest` (replace existing):

```python
class AgentSearchRequest(BaseModel):
    """Request model for agent-routed search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
```

Update `AgentSearchResponse` (replace existing):

```python
class AgentSearchResponse(BaseModel):
    """Response model for agent-routed search."""

    method_used: str
    router_reasoning: str
    rewritten_query: str | None = None
    response: str
    sources: list = Field(default_factory=list)
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_router_agent.py::TestConversationTurnModel -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add backend/app/models/schemas.py backend/tests/unit/test_router_agent.py
git commit -m "feat: add ConversationTurn model and update AgentSearch schemas"
```

---

## Task 2: Update `RouteDecision` and history formatter in `RouterAgent`

**Files:**
- Modify: `backend/app/services/router_agent.py`

**Step 1: Write the failing tests**

Add to `backend/tests/unit/test_router_agent.py` inside `TestRouterAgent`:

```python
def test_format_history_empty(self, router_agent):
    """_format_history() returns empty string when no history."""
    result = router_agent._format_history([])
    assert result == ""

def test_format_history_single_user_turn(self, router_agent):
    from backend.app.models.schemas import ConversationTurn
    turns = [
        ConversationTurn(role="user", content="Tell me about Inception",
                         rewritten_query="Tell me about Inception", method_used="local"),
    ]
    result = router_agent._format_history(turns)
    assert "[User]" in result
    assert "Tell me about Inception" in result
    assert "method: local" in result

def test_format_history_limits_to_five_turns(self, router_agent):
    from backend.app.models.schemas import ConversationTurn
    turns = []
    for i in range(7):
        turns.append(ConversationTurn(role="user", content=f"Question {i}",
                                      rewritten_query=f"Question {i}", method_used="local"))
        turns.append(ConversationTurn(role="assistant", content=f"Answer {i}"))
    result = router_agent._format_history(turns)
    # Only last 5 user turns (10 messages) should appear
    assert "Question 2" not in result
    assert "Question 6" in result

@pytest.mark.asyncio
async def test_route_returns_rewritten_query(self, router_agent):
    """route() should return rewritten_query in RouteDecision."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"rewritten_query": "Who directed Inception?", '
        '"method": "local", "confidence": 0.9, "reasoning": "entity query"}'
    )

    with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        result = await router_agent.route("Who directed it?", "movies collection")
        assert result.rewritten_query == "Who directed Inception?"

@pytest.mark.asyncio
async def test_route_falls_back_to_original_query_when_rewrite_missing(self, router_agent):
    """route() uses original query if LLM omits rewritten_query."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"method": "local", "confidence": 0.8, "reasoning": "entity query"}'
    )

    with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        result = await router_agent.route("Who directed it?", "movies collection")
        assert result.rewritten_query == "Who directed it?"

@pytest.mark.asyncio
async def test_route_passes_history_to_prompt(self, router_agent):
    """route() formats history into the prompt sent to LLM."""
    from backend.app.models.schemas import ConversationTurn
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"rewritten_query": "Who directed Inception?", '
        '"method": "local", "confidence": 0.9, "reasoning": "entity"}'
    )

    history = [
        ConversationTurn(role="user", content="Tell me about Inception",
                         rewritten_query="Tell me about Inception", method_used="local"),
        ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
    ]

    with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        await router_agent.route("Who directed it?", "movies", history)
        call_prompt = mock_llm.call_args[0][0]
        assert "Tell me about Inception" in call_prompt
        assert "Inception is a 2010 film" in call_prompt
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/unit/test_router_agent.py -k "format_history or rewritten_query or passes_history" -v
```

Expected: FAIL — `_format_history` does not exist, `RouteDecision` has no `rewritten_query`, `route()` takes no `history` param.

**Step 3: Implement the changes**

Replace the contents of `backend/app/services/router_agent.py` with:

```python
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

HISTORY_MAX_TURNS = 5  # number of user turns to include


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

    def _format_history(self, conversation_history: list[ConversationTurn]) -> str:
        """Format last N turns of conversation history for the prompt."""
        if not conversation_history:
            return ""

        # Keep last HISTORY_MAX_TURNS user turns (and their paired assistant turns)
        # Walk backwards collecting user turns until we have enough
        user_turn_count = 0
        cutoff_index = 0
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].role == "user":
                user_turn_count += 1
                if user_turn_count == HISTORY_MAX_TURNS:
                    cutoff_index = i
                    break

        recent = conversation_history[cutoff_index:]

        lines = ["Conversation history (most recent last):"]
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
                    # Truncate long assistant responses to keep prompt small
                    content = turn.content[:300] + "..." if len(turn.content) > 300 else turn.content
                    lines.append(f"[Assistant] {content}")
            except Exception:
                logger.warning("Skipping malformed conversation turn")
                continue

        return "\n".join(lines)

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
                    max_tokens=500,
                    api_key=settings.google_api_key,
                    response_format={"type": "json_object"},
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
    ) -> RouteDecision:
        """
        Analyze query and determine optimal search method.

        Args:
            query: The user's search query
            collection_context: Description of the collection's content
            conversation_history: Prior conversation turns for context

        Returns:
            RouteDecision with method, confidence, reasoning, and rewritten_query
        """
        history_block = self._format_history(conversation_history or [])

        prompt = self.prompt_template.format(
            query=query,
            collection_context=collection_context or "No collection context available",
            conversation_history_block=history_block,
        )

        try:
            response = await self._call_llm(prompt)
            content = response.choices[0].message.content

            logger.debug(f"Router LLM raw response: {content}")

            if not content or not content.strip():
                logger.warning("Router received empty response from LLM")
                return RouteDecision(
                    method="local",
                    confidence=0.3,
                    reasoning="Default to LOCAL - empty LLM response",
                    rewritten_query=query,
                )

            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace("```json", "").replace("```", "").strip()

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
```

**Step 4: Run tests to verify they pass**

```bash
pytest backend/tests/unit/test_router_agent.py -v
```

Expected: All tests PASS (existing 3 + new 6 = 9 tests)

**Step 5: Commit**

```bash
git add backend/app/services/router_agent.py backend/tests/unit/test_router_agent.py
git commit -m "feat: add conversation history and rewritten_query to RouterAgent"
```

---

## Task 3: Update router prompt template

**Files:**
- Modify: `backend/prompts/router_prompt.txt`

**Step 1: No test needed** — prompt content is validated by Task 2 tests that check the history block appears in the prompt string.

**Step 2: Update the prompt file**

Replace the entire contents of `backend/prompts/router_prompt.txt` with:

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

{conversation_history_block}

Current query: {query}

YOU MUST respond with ONLY a valid JSON object (no markdown, no code blocks, no explanations):
{{
  "rewritten_query": "Standalone version of the query with all pronouns and implicit references resolved. If the query is already standalone, repeat it unchanged.",
  "method": "local|global|tog|drift|web",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this method was chosen"
}}

Rules:
- If conversation history is present, resolve pronouns and implicit references in rewritten_query (e.g. "it", "he", "she", "they", "that film", "his work").
- If the query asks about current events, news, or real-time information, choose WEB.
- If the query asks about relationships or connections between concepts, choose TOG.
- If the query asks for broad overviews or trends across many topics, choose GLOBAL.
- If the query asks about specific entities or focused topics, choose LOCAL.
- Use DRIFT only when LOCAL might miss important related context.
- Default to LOCAL if uncertain between LOCAL and other methods.
- Consider method_used values in history — prefer consistency unless the query type clearly differs.
```

**Step 3: Verify existing tests still pass**

```bash
pytest backend/tests/unit/test_router_agent.py -v
```

Expected: All 9 tests PASS (prompt format changes are backward-compatible since `{conversation_history_block}` is optional — empty string when no history).

**Step 4: Commit**

```bash
git add backend/prompts/router_prompt.txt
git commit -m "feat: update router prompt to support conversation history and rewritten_query"
```

---

## Task 4: Thread conversation history through the search endpoint

**Files:**
- Modify: `backend/app/routers/search.py`
- Modify: `backend/tests/integration/test_agent_search.py`

**Step 1: Write the failing tests**

Add to `backend/tests/integration/test_agent_search.py` inside `TestAgentSearchIntegration`:

```python
def test_agent_search_passes_history_to_router(self, client):
    """agent_search endpoint passes conversation_history to router.route()."""
    mock_route = MagicMock()
    mock_route.method = "local"
    mock_route.confidence = 0.9
    mock_route.reasoning = "entity query"
    mock_route.rewritten_query = "Who directed Inception?"

    mock_result = MagicMock()
    mock_result.response = "Christopher Nolan."

    with patch(
        "backend.app.services.router_agent.RouterAgent.route",
        new_callable=AsyncMock,
    ) as mock_router:
        with patch(
            "backend.app.services.query_service.QueryService.local_search",
            new_callable=AsyncMock,
        ) as mock_local:
            mock_router.return_value = mock_route
            mock_local.return_value = mock_result

            response = client.post(
                "/api/collections/test/search/agent",
                json={
                    "query": "Who directed it?",
                    "stream": False,
                    "conversation_history": [
                        {
                            "role": "user",
                            "content": "Tell me about Inception",
                            "rewritten_query": "Tell me about Inception",
                            "method_used": "local",
                        },
                        {
                            "role": "assistant",
                            "content": "Inception is a 2010 film...",
                        },
                    ],
                },
            )

            assert response.status_code == 200
            # Verify router was called with history
            call_kwargs = mock_router.call_args
            assert call_kwargs is not None
            # history should be passed (positional or keyword)
            args, kwargs = call_kwargs
            history_passed = kwargs.get("conversation_history") or (args[2] if len(args) > 2 else None)
            assert history_passed is not None
            assert len(history_passed) == 2

def test_agent_search_response_includes_rewritten_query(self, client):
    """AgentSearchResponse includes rewritten_query field."""
    mock_route = MagicMock()
    mock_route.method = "local"
    mock_route.confidence = 0.9
    mock_route.reasoning = "entity query"
    mock_route.rewritten_query = "Who directed Inception?"

    mock_result = MagicMock()
    mock_result.response = "Christopher Nolan."

    with patch(
        "backend.app.services.router_agent.RouterAgent.route",
        new_callable=AsyncMock,
    ) as mock_router:
        with patch(
            "backend.app.services.query_service.QueryService.local_search",
            new_callable=AsyncMock,
        ) as mock_local:
            mock_router.return_value = mock_route
            mock_local.return_value = mock_result

            response = client.post(
                "/api/collections/test/search/agent",
                json={"query": "Who directed it?", "stream": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert "rewritten_query" in data
            assert data["rewritten_query"] == "Who directed Inception?"

def test_agent_search_uses_rewritten_query_for_search(self, client):
    """agent_search calls search methods with rewritten_query, not original query."""
    mock_route = MagicMock()
    mock_route.method = "local"
    mock_route.confidence = 0.9
    mock_route.reasoning = "entity query"
    mock_route.rewritten_query = "Who directed Inception?"

    mock_result = MagicMock()
    mock_result.response = "Christopher Nolan."

    with patch(
        "backend.app.services.router_agent.RouterAgent.route",
        new_callable=AsyncMock,
    ) as mock_router:
        with patch(
            "backend.app.services.query_service.QueryService.local_search",
            new_callable=AsyncMock,
        ) as mock_local:
            mock_router.return_value = mock_route
            mock_local.return_value = mock_result

            client.post(
                "/api/collections/test/search/agent",
                json={"query": "Who directed it?", "stream": False},
            )

            # local_search should be called with the rewritten query
            call_kwargs = mock_local.call_args
            args, kwargs = call_kwargs
            query_used = kwargs.get("query") or args[1]
            assert query_used == "Who directed Inception?"
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/integration/test_agent_search.py -v
```

Expected: New 3 tests FAIL — endpoint doesn't pass history or rewritten_query yet.

**Step 3: Update the agent_search endpoint**

In `backend/app/routers/search.py`, replace the `agent_search` function (lines 149-215):

```python
@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    The router agent analyzes the query and selects the optimal search method.
    Supports multi-turn conversations via conversation_history.
    """
    try:
        collection_context = f"Collection: {collection_id}"

        route_decision = await router_agent.route(
            request.query,
            collection_context,
            conversation_history=request.conversation_history or None,
        )
        logger.info(
            f"Router decision: {route_decision.method} (confidence: {route_decision.confidence}) "
            f"rewritten: '{route_decision.rewritten_query}'"
        )

        search_query = route_decision.rewritten_query or request.query

        if route_decision.method == "web":
            from ..services import web_search_service

            result = await web_search_service.search(search_query)
            return AgentSearchResponse(
                method_used="web",
                router_reasoning=route_decision.reasoning,
                rewritten_query=route_decision.rewritten_query,
                response=result.response,
                sources=[s.model_dump() for s in result.sources],
            )

        if route_decision.method == "global":
            result = await query_service.global_search(
                collection_id=collection_id,
                query=search_query,
            )
        elif route_decision.method == "tog":
            result = await query_service.tog_search(
                collection_id=collection_id,
                query=search_query,
            )
        elif route_decision.method == "drift":
            result = await query_service.drift_search(
                collection_id=collection_id,
                query=search_query,
            )
        else:
            result = await query_service.local_search(
                collection_id=collection_id,
                query=search_query,
            )

        return AgentSearchResponse(
            method_used=route_decision.method,
            router_reasoning=route_decision.reasoning,
            rewritten_query=route_decision.rewritten_query,
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

Also update the streaming endpoint `agent_search_stream` — replace the `route` call and downstream search calls in `event_generator()`:

```python
# Replace:
route_decision = await router_agent.route(request.query, collection_context)
# With:
route_decision = await router_agent.route(
    request.query,
    collection_context,
    conversation_history=request.conversation_history or None,
)

# Replace all occurrences of:
request.query
# With (for the search method calls only, NOT the SSE status messages):
route_decision.rewritten_query or request.query
```

And add `rewritten_query` to the `done` event data:

```python
yield {
    "event": "done",
    "data": json.dumps({
        "method_used": route_decision.method,
        "rewritten_query": route_decision.rewritten_query,
        "sources": sources,
        "router_reasoning": route_decision.reasoning,
    }),
}
```

**Step 4: Run all tests to verify they pass**

```bash
pytest backend/tests/ -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add backend/app/routers/search.py backend/tests/integration/test_agent_search.py
git commit -m "feat: thread conversation_history and rewritten_query through agent search endpoint"
```

---

## Task 5: Update existing unit tests broken by RouteDecision change

**Files:**
- Modify: `backend/tests/unit/test_router_agent.py`
- Modify: `backend/tests/integration/test_agent_search.py`

**Step 1: Run all tests to find breakage**

```bash
pytest backend/tests/ -v
```

Look for failures in existing tests that mock `RouteDecision` or check its fields — specifically `test_full_agent_search_flow` which creates a `MagicMock()` for route without `rewritten_query`.

**Step 2: Fix the integration test mock**

In `backend/tests/integration/test_agent_search.py`, update `test_full_agent_search_flow` — add `rewritten_query` to the mock:

```python
mock_route = MagicMock()
mock_route.method = "web"
mock_route.confidence = 0.9
mock_route.reasoning = "External information needed"
mock_route.rewritten_query = "What are the latest FDA regulations?"  # add this line
```

**Step 3: Run all tests again**

```bash
pytest backend/tests/ -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add backend/tests/
git commit -m "fix: update test mocks for RouteDecision.rewritten_query field"
```

---

## Task 6: Final verification

**Step 1: Run full test suite**

```bash
pytest backend/tests/ -v --tb=short
```

Expected: All tests PASS, no warnings about missing fields.

**Step 2: Verify the models export correctly**

```bash
cd F:/KL/gtog
python -c "from backend.app.models.schemas import ConversationTurn, AgentSearchRequest, AgentSearchResponse; print('OK')"
```

Expected: `OK`

**Step 3: Verify router agent imports cleanly**

```bash
python -c "from backend.app.services.router_agent import RouterAgent, RouteDecision; r = RouterAgent(); print('OK')"
```

Expected: `OK`

**Step 4: Final commit if any cleanup needed, then push**

```bash
git log --oneline -6
```

Verify the 5 commits from this plan are present in order.
