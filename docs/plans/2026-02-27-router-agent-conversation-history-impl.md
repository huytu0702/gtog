# Router Agent Conversation History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conversation history with hierarchical summarization to the router agent so it resolves pronouns and implicit references in multi-turn queries at bounded token cost, regardless of conversation length.

**Architecture:** The frontend carries `conversation_summary` (compressed past) + `conversation_history` (recent turns) and sends both with each request. A single LLM call rewrites the query and selects the search method. A separate `/agent/summarize` endpoint compresses history when it exceeds a threshold — the frontend calls this before sending the next query. Backend stays fully stateless.

**Tech Stack:** FastAPI, Pydantic v2, litellm, pytest, pytest-asyncio

---

## Task 1: Add schema models

**Files:**
- Modify: `backend/app/models/schemas.py`

**Step 1: Write the failing tests**

Add to `backend/tests/unit/test_router_agent.py` after the imports:

```python
from backend.app.models.schemas import (
    ConversationTurn,
    AgentSearchRequest,
    AgentSearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
```

Add this test class after the existing `TestRouterAgent` class:

```python
class TestSchemaModels:
    """Test new conversation schema models."""

    def test_conversation_turn_user_with_metadata(self):
        turn = ConversationTurn(
            role="user",
            content="Who directed it?",
            rewritten_query="Who directed Inception?",
            method_used="local",
        )
        assert turn.role == "user"
        assert turn.rewritten_query == "Who directed Inception?"
        assert turn.method_used == "local"

    def test_conversation_turn_assistant_no_metadata(self):
        turn = ConversationTurn(role="assistant", content="Christopher Nolan directed Inception.")
        assert turn.rewritten_query is None
        assert turn.method_used is None

    def test_agent_search_request_defaults(self):
        req = AgentSearchRequest(query="hello")
        assert req.conversation_history == []
        assert req.conversation_summary is None

    def test_agent_search_request_accepts_summary_and_history(self):
        req = AgentSearchRequest(
            query="Who directed it?",
            conversation_summary="User asked about Inception.",
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ],
        )
        assert req.conversation_summary == "User asked about Inception."
        assert len(req.conversation_history) == 2

    def test_agent_search_response_has_rewritten_query(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="entity query",
            rewritten_query="Who directed Inception?",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query == "Who directed Inception?"

    def test_agent_search_response_rewritten_query_optional(self):
        resp = AgentSearchResponse(
            method_used="local",
            router_reasoning="entity query",
            response="Christopher Nolan.",
        )
        assert resp.rewritten_query is None

    def test_summarize_request_model(self):
        req = SummarizeRequest(
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ],
            existing_summary="Previous summary.",
        )
        assert len(req.conversation_history) == 2
        assert req.existing_summary == "Previous summary."

    def test_summarize_request_no_existing_summary(self):
        req = SummarizeRequest(
            conversation_history=[
                ConversationTurn(role="user", content="Tell me about Inception"),
            ]
        )
        assert req.existing_summary is None

    def test_summarize_response_model(self):
        resp = SummarizeResponse(
            summary="User asked about Inception.",
            trimmed_history=[
                ConversationTurn(role="user", content="Who directed it?",
                                 rewritten_query="Who directed Inception?", method_used="local"),
                ConversationTurn(role="assistant", content="Christopher Nolan."),
            ],
        )
        assert resp.summary == "User asked about Inception."
        assert len(resp.trimmed_history) == 2
```

**Step 2: Run test to verify it fails**

```bash
cd F:/KL/gtog
pytest backend/tests/unit/test_router_agent.py::TestSchemaModels -v
```

Expected: FAIL — `ConversationTurn`, `SummarizeRequest`, `SummarizeResponse` do not exist yet.

**Step 3: Implement the models**

In `backend/app/models/schemas.py`:

1. Update the imports line to add `Literal`:

```python
from typing import Any, Dict, List, Literal, Optional
```

2. Add these models before `AgentSearchRequest` (around line 136):

```python
class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: Literal["user", "assistant"]
    content: str
    rewritten_query: str | None = None  # user turns only
    method_used: str | None = None      # user turns only


class SummarizeRequest(BaseModel):
    """Request model for conversation summarization."""

    conversation_history: list[ConversationTurn]
    existing_summary: str | None = None


class SummarizeResponse(BaseModel):
    """Response model for conversation summarization."""

    summary: str
    trimmed_history: list[ConversationTurn]
```

3. Replace `AgentSearchRequest`:

```python
class AgentSearchRequest(BaseModel):
    """Request model for agent-routed search."""

    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    conversation_summary: str | None = None
```

4. Replace `AgentSearchResponse`:

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
pytest backend/tests/unit/test_router_agent.py::TestSchemaModels -v
```

Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add backend/app/models/schemas.py backend/tests/unit/test_router_agent.py
git commit -m "feat: add ConversationTurn, SummarizeRequest/Response and update AgentSearch schemas"
```

---

## Task 2: Update `RouterAgent` with history + summary support

**Files:**
- Modify: `backend/app/services/router_agent.py`

**Step 1: Write the failing tests**

Add inside `TestRouterAgent` in `backend/tests/unit/test_router_agent.py`:

```python
def test_format_history_block_empty(self, router_agent):
    """Returns empty string when no history and no summary."""
    result = router_agent._format_history_block([], None)
    assert result == ""

def test_format_history_block_summary_only(self, router_agent):
    """Shows summary section when only summary provided."""
    result = router_agent._format_history_block([], "User asked about Inception.")
    assert "Past conversation summary" in result
    assert "User asked about Inception." in result

def test_format_history_block_turns_only(self, router_agent):
    from backend.app.models.schemas import ConversationTurn
    turns = [
        ConversationTurn(role="user", content="Tell me about Inception",
                         rewritten_query="Tell me about Inception", method_used="local"),
        ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
    ]
    result = router_agent._format_history_block(turns, None)
    assert "[User]" in result
    assert "Tell me about Inception" in result
    assert "method: local" in result
    assert "[Assistant]" in result

def test_format_history_block_summary_and_turns(self, router_agent):
    from backend.app.models.schemas import ConversationTurn
    turns = [
        ConversationTurn(role="user", content="Who starred in it?",
                         rewritten_query="Who starred in Inception?", method_used="local"),
        ConversationTurn(role="assistant", content="Leonardo DiCaprio starred..."),
    ]
    result = router_agent._format_history_block(turns, "User asked about Inception (2010).")
    assert "Past conversation summary" in result
    assert "User asked about Inception (2010)." in result
    assert "Recent conversation" in result
    assert "Who starred in it?" in result

def test_format_history_block_limits_to_three_recent_turns(self, router_agent):
    from backend.app.models.schemas import ConversationTurn
    turns = []
    for i in range(5):
        turns.append(ConversationTurn(role="user", content=f"Question {i}",
                                      rewritten_query=f"Question {i}", method_used="local"))
        turns.append(ConversationTurn(role="assistant", content=f"Answer {i}"))
    result = router_agent._format_history_block(turns, None)
    assert "Question 1" not in result
    assert "Question 4" in result

@pytest.mark.asyncio
async def test_route_returns_rewritten_query(self, router_agent):
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
async def test_route_injects_summary_into_prompt(self, router_agent):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"rewritten_query": "Who directed Inception?", '
        '"method": "local", "confidence": 0.9, "reasoning": "entity"}'
    )
    with patch.object(router_agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        await router_agent.route(
            "Who directed it?", "movies",
            conversation_summary="User asked about Inception (2010)."
        )
        call_prompt = mock_llm.call_args[0][0]
        assert "User asked about Inception (2010)." in call_prompt

@pytest.mark.asyncio
async def test_route_injects_history_into_prompt(self, router_agent):
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
        await router_agent.route("Who directed it?", "movies", conversation_history=history)
        call_prompt = mock_llm.call_args[0][0]
        assert "Tell me about Inception" in call_prompt
        assert "Inception is a 2010 film" in call_prompt
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/unit/test_router_agent.py -k "format_history_block or rewritten_query or injects" -v
```

Expected: FAIL — `_format_history_block` does not exist, `RouteDecision` has no `rewritten_query`, `route()` takes no history/summary params.

**Step 3: Replace `backend/app/services/router_agent.py`**

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
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = (
            Path(__file__).parent.parent.parent / "prompts" / "router_prompt.txt"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        return self._default_prompt()

    def _default_prompt(self) -> str:
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
        """Call LLM with exponential backoff on rate limits."""
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
                    logger.warning("response_format not supported, falling back")
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
            logger.warning(
                f"Failed to parse router response. Error: {e}. "
                f"Content: {content[:200] if 'content' in locals() else 'N/A'}"
            )
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

**Step 4: Run all router agent tests**

```bash
pytest backend/tests/unit/test_router_agent.py -v
```

Expected: All tests PASS (existing 3 + new schema tests + new router tests)

**Step 5: Commit**

```bash
git add backend/app/services/router_agent.py backend/tests/unit/test_router_agent.py
git commit -m "feat: add history block formatter and conversation_summary support to RouterAgent"
```

---

## Task 3: Add `SummarizationService` and prompt

**Files:**
- Create: `backend/app/services/summarization_service.py`
- Create: `backend/prompts/summarization_prompt.txt`
- Create: `backend/tests/unit/test_summarization_service.py`

**Step 1: Write the failing tests**

Create `backend/tests/unit/test_summarization_service.py`:

```python
"""Tests for SummarizationService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.summarization_service import SummarizationService
from backend.app.models.schemas import ConversationTurn


class TestSummarizationService:
    """Test SummarizationService."""

    @pytest.fixture
    def service(self):
        return SummarizationService()

    @pytest.mark.asyncio
    async def test_summarize_returns_string(self, service):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "User asked about Inception (2010 film)."

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            turns = [
                ConversationTurn(role="user", content="Tell me about Inception",
                                 rewritten_query="Tell me about Inception", method_used="local"),
                ConversationTurn(role="assistant", content="Inception is a 2010 film..."),
            ]
            result = await service.summarize(turns)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_includes_existing_summary_in_prompt(self, service):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Updated summary."

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            turns = [
                ConversationTurn(role="user", content="Who directed it?",
                                 rewritten_query="Who directed Inception?", method_used="local"),
                ConversationTurn(role="assistant", content="Christopher Nolan."),
            ]
            await service.summarize(turns, existing_summary="User was asking about Inception.")
            call_prompt = mock_llm.call_args[0][0]
            assert "User was asking about Inception." in call_prompt

    @pytest.mark.asyncio
    async def test_summarize_falls_back_on_llm_error(self, service):
        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")
            turns = [
                ConversationTurn(role="user", content="Tell me about Inception"),
            ]
            result = await service.summarize(turns)
            # Falls back to a basic concatenation rather than crashing
            assert isinstance(result, str)

    def test_get_trimmed_history_keeps_recent_turns(self, service):
        from backend.app.models.schemas import ConversationTurn
        turns = []
        for i in range(5):
            turns.append(ConversationTurn(role="user", content=f"Q{i}",
                                          rewritten_query=f"Q{i}", method_used="local"))
            turns.append(ConversationTurn(role="assistant", content=f"A{i}"))

        trimmed = service.get_trimmed_history(turns, keep_turns=3)
        # Should keep last 3 user turns = 6 messages
        user_turns = [t for t in trimmed if t.role == "user"]
        assert len(user_turns) == 3
        assert user_turns[-1].content == "Q4"
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/unit/test_summarization_service.py -v
```

Expected: FAIL — `SummarizationService` does not exist.

**Step 3: Create the summarization prompt**

Create `backend/prompts/summarization_prompt.txt`:

```
You are a conversation summarizer. Your task is to compress the conversation below into a concise summary that helps a routing agent understand the topic, entities discussed, and user intent.

Focus on:
- Main topics and entities mentioned (people, films, places, concepts)
- The type of questions the user is asking (relationships, properties, comparisons)
- Any explicit focus or scope the user has established

Keep the summary to 2-4 sentences maximum. Be specific about named entities.

{existing_summary_block}

Conversation to summarize:
{conversation_text}

Write only the summary, no preamble or labels.
```

**Step 4: Create `backend/app/services/summarization_service.py`**

```python
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
            model=settings.default_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            api_key=settings.google_api_key,
        )
        return response.choices[0].message.content or ""

    def _format_turns(self, turns: list[ConversationTurn]) -> str:
        lines = []
        for turn in turns:
            if turn.role == "user":
                q = turn.rewritten_query or turn.content
                lines.append(f"User: {q}")
            else:
                content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
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

        Returns:
            New summary string. Falls back to basic concatenation on LLM error.
        """
        existing_summary_block = ""
        if existing_summary:
            existing_summary_block = f"Previous summary:\n{existing_summary}\n\nNew turns to incorporate:"

        conversation_text = self._format_turns(conversation_history)

        prompt = self.prompt_template.format(
            existing_summary_block=existing_summary_block,
            conversation_text=conversation_text,
        )

        try:
            return await self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Summarization LLM call failed: {e}. Using fallback.")
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
```

**Step 5: Export from services `__init__.py`**

Open `backend/app/services/__init__.py` and add:

```python
from .summarization_service import summarization_service
```

**Step 6: Run tests to verify they pass**

```bash
pytest backend/tests/unit/test_summarization_service.py -v
```

Expected: PASS (4 tests)

**Step 7: Commit**

```bash
git add backend/app/services/summarization_service.py \
        backend/prompts/summarization_prompt.txt \
        backend/app/services/__init__.py \
        backend/tests/unit/test_summarization_service.py
git commit -m "feat: add SummarizationService for hierarchical conversation compression"
```

---

## Task 4: Update router prompt template

**Files:**
- Modify: `backend/prompts/router_prompt.txt`

**Step 1: No new test needed** — Task 2 tests already verify the history block content appears in the prompt.

**Step 2: Replace `backend/prompts/router_prompt.txt`**

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
- If history or summary is present, resolve pronouns and implicit references in rewritten_query (e.g. "it", "he", "she", "they", "that film", "his work").
- If the query asks about current events, news, or real-time information, choose WEB.
- If the query asks about relationships or connections between concepts, choose TOG.
- If the query asks for broad overviews or trends across many topics, choose GLOBAL.
- If the query asks about specific entities or focused topics, choose LOCAL.
- Use DRIFT only when LOCAL might miss important related context.
- Default to LOCAL if uncertain between LOCAL and other methods.
- Consider method_used values in history — prefer consistency unless the query type clearly differs.
```

**Step 3: Verify all router agent tests still pass**

```bash
pytest backend/tests/unit/test_router_agent.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add backend/prompts/router_prompt.txt
git commit -m "feat: update router prompt with summary + history block and rewritten_query output"
```

---

## Task 5: Add `/agent/summarize` endpoint and thread history through agent search

**Files:**
- Modify: `backend/app/routers/search.py`
- Modify: `backend/tests/integration/test_agent_search.py`

**Step 1: Write the failing tests**

Add to `backend/tests/integration/test_agent_search.py`:

```python
def test_summarize_endpoint_returns_summary_and_trimmed_history(self, client):
    """POST /agent/summarize returns summary and trimmed history."""
    mock_summary = "User explored Inception (2010) film."

    with patch(
        "backend.app.services.summarization_service.SummarizationService.summarize",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.return_value = mock_summary

        response = client.post(
            "/api/collections/test/search/agent/summarize",
            json={
                "conversation_history": [
                    {"role": "user", "content": "Tell me about Inception",
                     "rewritten_query": "Tell me about Inception", "method_used": "local"},
                    {"role": "assistant", "content": "Inception is a 2010 film..."},
                    {"role": "user", "content": "Who directed it?",
                     "rewritten_query": "Who directed Inception?", "method_used": "local"},
                    {"role": "assistant", "content": "Christopher Nolan."},
                ],
                "existing_summary": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"] == mock_summary
        assert "trimmed_history" in data
        assert isinstance(data["trimmed_history"], list)

def test_agent_search_passes_history_and_summary_to_router(self, client):
    """agent_search endpoint passes both conversation_history and conversation_summary to router."""
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
                    "conversation_summary": "User asked about Inception (2010).",
                    "conversation_history": [
                        {"role": "user", "content": "Who starred in it?",
                         "rewritten_query": "Who starred in Inception?", "method_used": "local"},
                        {"role": "assistant", "content": "Leonardo DiCaprio."},
                    ],
                },
            )

            assert response.status_code == 200
            args, kwargs = mock_router.call_args
            assert kwargs.get("conversation_summary") == "User asked about Inception (2010)."
            history = kwargs.get("conversation_history") or (args[2] if len(args) > 2 else None)
            assert history is not None
            assert len(history) == 2

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

            args, kwargs = mock_local.call_args
            query_used = kwargs.get("query") or args[1]
            assert query_used == "Who directed Inception?"
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/integration/test_agent_search.py -v
```

Expected: New 4 tests FAIL.

**Step 3: Update `backend/app/routers/search.py`**

Add import at top:
```python
from ..services import summarization_service
from ..models import SummarizeRequest, SummarizeResponse
```

Add `/agent/summarize` endpoint before `agent_search`:

```python
@router.post("/agent/summarize", response_model=SummarizeResponse)
async def summarize_conversation(collection_id: str, request: SummarizeRequest):
    """
    Compress conversation history into a summary.

    Call this when conversation_history exceeds your threshold (e.g. 6 turns).
    Returns a new summary and trimmed recent history to carry forward.
    """
    try:
        summary = await summarization_service.summarize(
            conversation_history=request.conversation_history,
            existing_summary=request.existing_summary,
        )
        trimmed = summarization_service.get_trimmed_history(request.conversation_history)
        return SummarizeResponse(summary=summary, trimmed_history=trimmed)
    except Exception as e:
        logger.exception("Error summarizing conversation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
```

Replace the `agent_search` function:

```python
@router.post("/agent", response_model=AgentSearchResponse)
async def agent_search(collection_id: str, request: AgentSearchRequest):
    """
    Perform an agent-routed search.

    Supports multi-turn conversations via conversation_history and conversation_summary.
    The router rewrites the query and selects the search method in a single LLM call.
    """
    try:
        collection_context = f"Collection: {collection_id}"

        route_decision = await router_agent.route(
            request.query,
            collection_context,
            conversation_history=request.conversation_history or None,
            conversation_summary=request.conversation_summary,
        )
        logger.info(
            f"Router decision: {route_decision.method} "
            f"(confidence: {route_decision.confidence}) "
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
                collection_id=collection_id, query=search_query
            )
        elif route_decision.method == "tog":
            result = await query_service.tog_search(
                collection_id=collection_id, query=search_query
            )
        elif route_decision.method == "drift":
            result = await query_service.drift_search(
                collection_id=collection_id, query=search_query
            )
        else:
            result = await query_service.local_search(
                collection_id=collection_id, query=search_query
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

Also update `agent_search_stream` — replace the `route` call:

```python
# Replace:
route_decision = await router_agent.route(request.query, collection_context)
# With:
route_decision = await router_agent.route(
    request.query,
    collection_context,
    conversation_history=request.conversation_history or None,
    conversation_summary=request.conversation_summary,
)
```

Replace `request.query` with `route_decision.rewritten_query or request.query` in the search method calls (not the SSE status messages). Add `rewritten_query` to the `done` event:

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

**Step 4: Export `SummarizeRequest` and `SummarizeResponse` from models `__init__.py`**

Open `backend/app/models/__init__.py` and add:

```python
from .schemas import SummarizeRequest, SummarizeResponse
```

**Step 5: Run all integration tests**

```bash
pytest backend/tests/integration/ -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add backend/app/routers/search.py \
        backend/app/models/__init__.py \
        backend/tests/integration/test_agent_search.py
git commit -m "feat: add /agent/summarize endpoint and thread history+summary through agent search"
```

---

## Task 6: Fix existing test mocks and final verification

**Files:**
- Modify: `backend/tests/integration/test_agent_search.py`
- Modify: `backend/tests/unit/test_router_agent.py`

**Step 1: Run full test suite to find breakage**

```bash
pytest backend/tests/ -v
```

Look for existing tests that mock `RouteDecision` without `rewritten_query`, or that call `router_agent.route()` without the new params.

**Step 2: Fix `test_full_agent_search_flow`**

In `backend/tests/integration/test_agent_search.py`, add `rewritten_query` to the existing mock:

```python
mock_route.rewritten_query = "What are the latest FDA regulations?"
```

**Step 3: Fix any existing router agent tests**

In `backend/tests/unit/test_router_agent.py`, update existing LLM mock responses to include `rewritten_query`:

```python
# Existing test_route_returns_route_decision — update mock content:
mock_response.choices[0].message.content = (
    '{"rewritten_query": "What is chamomile used for?", '
    '"method": "local", "confidence": 0.85, "reasoning": "Query asks about specific entity"}'
)

# Existing test_route_identifies_web_search_query — update mock content:
mock_response.choices[0].message.content = (
    '{"rewritten_query": "What are the latest FDA regulations?", '
    '"method": "web", "confidence": 0.92, "reasoning": "Query asks about current FDA regulations"}'
)
```

**Step 4: Run full test suite**

```bash
pytest backend/tests/ -v --tb=short
```

Expected: All tests PASS

**Step 5: Verify imports**

```bash
python -c "
from backend.app.models.schemas import ConversationTurn, SummarizeRequest, SummarizeResponse, AgentSearchRequest, AgentSearchResponse
from backend.app.services.router_agent import RouterAgent, RouteDecision
from backend.app.services.summarization_service import SummarizationService
print('All imports OK')
"
```

Expected: `All imports OK`

**Step 6: Commit**

```bash
git add backend/tests/
git commit -m "fix: update test mocks for RouteDecision.rewritten_query and new route() signature"
```

---

## Summary

| Task | Commits | Key Changes |
|---|---|---|
| 1 | 1 | `ConversationTurn`, `SummarizeRequest/Response`, updated `AgentSearch*` schemas |
| 2 | 1 | `RouterAgent._format_history_block()`, `conversation_summary` param, `RouteDecision.rewritten_query` |
| 3 | 1 | `SummarizationService`, `summarization_prompt.txt` |
| 4 | 1 | Updated `router_prompt.txt` |
| 5 | 1 | `/agent/summarize` endpoint, history+summary threaded through agent search |
| 6 | 1 | Fix existing test mocks |

**Token budget per request (always bounded):**
```
conversation_summary  ≈  100-200 tokens
recent_turns (3)      ≈  300-500 tokens
router prompt base    ≈  200 tokens
─────────────────────────────────────
total                 ≈  600-900 tokens  (fixed, regardless of conversation length)
```
