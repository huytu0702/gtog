# Router Agent Conversation History — Design

**Date:** 2026-02-27
**Status:** Approved
**Branch:** feature/router-agent-conversation-history

---

## Background

The `RouterAgent` currently makes routing decisions based only on the current query. In multi-turn conversations, follow-up questions like *"who directed it?"* or *"what about his earlier work?"* contain unresolved pronouns and implicit references. Without conversation history, the router misroutes these queries and the search methods receive queries with no context.

The ToG search layer already supports `ConversationHistory` (merged on `feature/tog-conversation-history`). This design extends conversation awareness up to the routing layer and defines the state model the frontend carries across turns.

---

## Goals

- Router agent resolves pronouns and implicit references in follow-up queries
- Single LLM call handles both query rewriting and routing decision
- Frontend owns session state (stateless backend)
- Full conversation context preserved indefinitely via hierarchical summarization
- `AgentSearchResponse` returns `rewritten_query` and updated `conversation_summary` so frontend can store them
- Consistent with the existing `ConversationHistory` pattern used by ToG/Local/Global search

---

## Non-Goals

- No backend session storage (no Redis, no DB)
- No changes to `ToGSearchConfig` or other search configs
- No conversation history threading to local/global/drift search (separate feature)
- No automatic entity extraction from responses
- No embedding-based retrieval of past turns

---

## Memory Strategy: Hierarchical Summarization

A sliding window alone drops early context permanently. Hierarchical summarization keeps full semantic context at bounded token cost regardless of conversation length.

```
Turns 1-5  →  LLM summarizes into 2-3 sentences  →  summary_v1
Turns 6-8  →  kept as recent_turns
Turn 9     →  router sees: summary_v1 + turns 6-8 + current query
             token budget stays fixed no matter how long the conversation runs

Turns 9-13 →  LLM re-summarizes: summary_v1 + turns 6-13  →  summary_v2
Turns 14-16 → kept as recent_turns
Turn 17    →  router sees: summary_v2 + turns 14-16 + current query
```

**Token budget (always bounded):**

```
conversation_summary  ≈  100-200 tokens  (compressed past)
recent_turns          ≈  300-500 tokens  (last 3 turns)
router prompt base    ≈  200 tokens
─────────────────────────────────────────
total per request     ≈  600-900 tokens  (fixed ceiling)
```

**Summarization trigger:** when `len(conversation_history) > SUMMARY_THRESHOLD` (default: 6 turns = 3 user + 3 assistant). The `/agent/summarize` endpoint performs the compression and returns the new summary + trimmed history. Frontend calls this endpoint before sending the next query when the threshold is exceeded.

---

## Data Flow

```
Frontend owns: conversation_summary + conversation_history (recent turns)
  │
  │  [When len(conversation_history) > SUMMARY_THRESHOLD]
  │  POST /agent/summarize {
  │    conversation_history: [all recent turns],
  │    existing_summary: "User was asking about Christopher Nolan..."
  │  }
  │  → returns: { new_summary, trimmed_history: [last 3 turns] }
  │  Frontend replaces its state with new_summary + trimmed_history
  │
  │  POST /agent/stream {
  │    query: "Who directed it?",
  │    conversation_summary: "User was asking about Inception (2010 film)...",
  │    conversation_history: [last 3 turns],
  │  }
  ▼
agent_search endpoint
  │  builds formatted context block: summary + recent turns
  │
  ▼
RouterAgent.route(query, collection_context, conversation_history, conversation_summary)
  │
  │  Single LLM call with combined prompt:
  │    - Available methods + descriptions
  │    - Collection context
  │    - Conversation summary (if present)
  │    - Last N recent turns
  │    - Current query
  │  Output: { rewritten_query, method, confidence, reasoning }
  │
  ▼
RouteDecision {
  method: "local",
  confidence: 0.91,
  reasoning: "Follow-up about Inception director — specific entity query",
  rewritten_query: "Who directed Inception?"
}
  │
  ├── if method == "tog":
  │     query_service.tog_search(collection_id,
  │       query=rewritten_query,
  │       conversation_history=ConversationHistory(...))
  │
  └── else:
        query_service.<method>_search(collection_id,
          query=rewritten_query)
          # history threading to local/global/drift: future work
  │
  ▼
AgentSearchResponse {
  method_used: "local",
  rewritten_query: "Who directed Inception?",
  router_reasoning: "Follow-up about Inception director — specific entity query",
  response: "Christopher Nolan directed Inception...",
  sources: []
}
  │
  │  Frontend appends turn to conversation_history
  │  Frontend checks if len(history) > SUMMARY_THRESHOLD
  │  If yes → calls /agent/summarize before next request
  ▼
```

---

## State Model

### `ConversationTurn` (new Pydantic model)

```python
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    rewritten_query: str | None = None   # user turns only
    method_used: str | None = None       # user turns only
```

### `AgentSearchRequest` (updated)

```python
class AgentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = True
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    conversation_summary: str | None = None   # compressed past context
```

### `AgentSearchResponse` (updated)

```python
class AgentSearchResponse(BaseModel):
    method_used: str
    router_reasoning: str
    rewritten_query: str | None = None
    response: str
    sources: list = Field(default_factory=list)
```

### `SummarizeRequest` / `SummarizeResponse` (new)

```python
class SummarizeRequest(BaseModel):
    conversation_history: list[ConversationTurn]
    existing_summary: str | None = None

class SummarizeResponse(BaseModel):
    summary: str                            # new compressed summary
    trimmed_history: list[ConversationTurn] # last RECENT_TURNS_KEPT turns to keep
```

### `RouteDecision` (updated)

```python
@dataclass
class RouteDecision:
    method: SearchMethodType
    confidence: float
    reasoning: str
    rewritten_query: str = field(default="")  # always populated
```

---

## Router Prompt Changes

The prompt gains three new sections:

1. **Conversation summary block** — compressed past context (if present)
2. **Recent turns block** — last N turns (default: 3)
3. **Rewriting instruction** — LLM must output `rewritten_query` as a standalone question

### Updated `router_prompt.txt`

```
You are a query routing assistant. Analyze the user's query and determine
which search method is most appropriate.

Available search methods:
- GLOBAL: Questions requiring understanding of the dataset as a whole.
- LOCAL: Questions about specific entities mentioned in documents.
- TOG: Questions about relationships between entities. Multi-hop reasoning.
- DRIFT: Local search + community context for broader variety.
- WEB: External information not in documents. Real-time/current events.

Collection context:
{collection_context}

{conversation_history_block}

Current query: {query}

YOU MUST respond with ONLY a valid JSON object:
{{
  "rewritten_query": "Standalone version of the query with all references resolved. If already standalone, repeat unchanged.",
  "method": "local|global|tog|drift|web",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

Rules:
- If history or summary is present, resolve pronouns and implicit references in rewritten_query.
- If query asks about current events or real-time info → WEB.
- If query asks about relationships between entities → TOG.
- If query asks for broad overviews or trends → GLOBAL.
- If query asks about specific entities → LOCAL.
- Use DRIFT only when LOCAL might miss important related context.
- Default to LOCAL if uncertain.
- Consider method_used in history — prefer consistency unless query type clearly differs.
```

### History block format (injected at `{conversation_history_block}`)

When both summary and recent turns are present:
```
Past conversation summary:
User has been exploring Christopher Nolan's filmography, starting with Inception (2010).
Discussion covered plot, themes, and cast. User showed interest in directorial style.

Recent conversation (most recent last):
[User] Who starred in it?  →  rewritten: "Who starred in Inception?"  →  method: local
[Assistant] Leonardo DiCaprio, Joseph Gordon-Levitt, and Elliot Page starred in Inception...
[User] What about the composer?
```

When only recent turns (no summary yet):
```
Conversation history (most recent last):
[User] Tell me about Inception  →  rewritten: "Tell me about Inception"  →  method: local
[Assistant] Inception is a 2010 science fiction film directed by Christopher Nolan...
[User] Who directed it?
```

---

## Summarization Service

A new `SummarizationService` handles history compression. It uses the LLM to produce a compact, routing-relevant summary of past turns.

**Summarization prompt focus:** The summary captures topics, entities, and intent — not verbatim answers. It is optimized for routing, not recall.

```python
class SummarizationService:
    async def summarize(
        self,
        turns: list[ConversationTurn],
        existing_summary: str | None = None,
    ) -> str:
        """Compress turns into a routing-relevant summary."""
```

**Trigger logic (frontend responsibility):**
```
SUMMARY_THRESHOLD = 6  # total turns (user + assistant)
RECENT_TURNS_KEPT = 6  # turns to keep after summarization (3 user + 3 assistant)

After each response:
  if len(conversation_history) > SUMMARY_THRESHOLD:
    POST /agent/summarize
    replace conversation_history with trimmed_history
    store new summary as conversation_summary
```

---

## Files Changed

| File | Change |
|---|---|
| `backend/app/models/schemas.py` | Add `ConversationTurn`, `SummarizeRequest`, `SummarizeResponse`; update `AgentSearchRequest` (add `conversation_summary`), `AgentSearchResponse` (add `rewritten_query`) |
| `backend/app/services/router_agent.py` | Add `conversation_history` + `conversation_summary` params to `route()`; update `RouteDecision`; add `_format_history_block()` with summary support |
| `backend/app/services/summarization_service.py` | New — `SummarizationService.summarize()` |
| `backend/prompts/router_prompt.txt` | Add `{conversation_history_block}` and `rewritten_query` output field |
| `backend/prompts/summarization_prompt.txt` | New — prompt for compressing conversation turns |
| `backend/app/routers/search.py` | Pass history + summary to `router_agent.route()`; add `/agent/summarize` endpoint; include `rewritten_query` in response |

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Memory strategy | Hierarchical summarization | Bounded tokens, infinite turns, no data loss |
| Single vs two LLM calls (routing) | Single combined call | Lower latency, lower cost |
| Summarization trigger | Frontend, threshold-based | Backend stays stateless; frontend controls timing |
| Who owns session state | Frontend | Scales horizontally, no shared state |
| Recent turns kept after summarization | 3 user + 3 assistant (6 total) | Enough for immediate context; summary covers the rest |
| Summarization model | Same `default_chat_model` | No extra config; task is simple compression |
| Rewrite always or only when history present | Always populate `rewritten_query` | Simplifies response parsing |
| Thread history to local/global/drift | Not in this feature | Separate concern; scope control |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| LLM returns no `rewritten_query` field | Fall back to original `query` |
| LLM returns invalid `method` | Default to `"local"` |
| `conversation_history` empty, no summary | Skip history block in prompt entirely |
| Summarization LLM call fails | Return original turns unchanged; log warning |
| History turn has malformed content | Skip that turn, log warning |

---

## Example Turn Sequence

**Turns 1-3 (no summarization yet)**
```
Turn 1: query="Tell me about Inception"
        → rewrite="Tell me about Inception", method=local

Turn 2: query="Who directed it?"
        history=[turn1_user, turn1_asst]
        → rewrite="Who directed Inception?", method=local

Turn 3: query="What is his relationship to Emma Thomas?"
        history=[turn1, turn2_user, turn2_asst]
        → rewrite="What is Christopher Nolan's relationship to Emma Thomas?"
        → method=tog  ← relationship query, router switches correctly
```

**Turn 7 (summarization triggered after turn 6)**
```
Frontend: len(history) > 6 → POST /agent/summarize
Summary:  "User explored Inception (2010): director Nolan, cast, composer Zimmer,
           and Nolan's relationship with producer Emma Thomas."
History trimmed to last 6 turns.

Turn 7: query="What other films did they produce together?"
        summary="User explored Inception..."
        history=[turn4, turn5, turn6_user, turn6_asst]
        → rewrite="What other films did Christopher Nolan and Emma Thomas produce together?"
        → method=tog
```

---

## Testing

- Unit test `_format_history_block()`: summary only, turns only, both combined
- Unit test `RouterAgent.route()` with mocked LLM: verify `rewritten_query` populated
- Unit test fallback: missing `rewritten_query` → original query used
- Unit test `SummarizationService.summarize()` with mocked LLM
- Integration test: multi-turn `/agent` endpoint, verify `rewritten_query` in response
- Integration test: `/agent/summarize` endpoint returns `summary` + `trimmed_history`
