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
- `AgentSearchResponse` returns `rewritten_query` so frontend can store it
- Consistent with the existing `ConversationHistory` pattern used by ToG/Local/Global search

---

## Non-Goals

- No backend session storage (no Redis, no DB)
- No changes to `ToGSearchConfig` or other search configs
- No conversation history threading to local/global/drift search (separate feature)
- No automatic entity extraction from responses

---

## Data Flow

```
Frontend owns conversation_history (list of ConversationTurn)
  │
  │  POST /agent/stream {
  │    query: "Who directed it?",
  │    conversation_history: [
  │      { role: "user",      content: "Tell me about Inception",
  │        rewritten_query: "Tell me about Inception", method_used: "local" },
  │      { role: "assistant", content: "Inception is a 2010 film by Christopher Nolan..." }
  │    ]
  │  }
  ▼
agent_search endpoint
  │  builds formatted history string from conversation_history
  │
  ▼
RouterAgent.route(query, collection_context, conversation_history)
  │
  │  Single LLM call with combined prompt:
  │    - Available methods + descriptions
  │    - Collection context
  │    - Last N turns of conversation
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
  │  Frontend appends turn to conversation_history, next request carries it
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
```

### `AgentSearchResponse` (updated)

```python
class AgentSearchResponse(BaseModel):
    method_used: str
    router_reasoning: str
    rewritten_query: str | None          # new — frontend stores this
    response: str
    sources: list = Field(default_factory=list)
```

### `RouteDecision` (updated)

```python
@dataclass
class RouteDecision:
    method: SearchMethodType
    confidence: float
    reasoning: str
    rewritten_query: str                 # new — always populated
```

---

## Router Prompt Changes

The prompt gains two new sections:

1. **Conversation history block** — formatted last N turns (default: 5)
2. **Rewriting instruction** — LLM must output `rewritten_query` as a standalone question

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
  "rewritten_query": "Standalone version of the query with all references resolved. If the query is already standalone, repeat it unchanged.",
  "method": "local|global|tog|drift|web",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

Rules:
- If history is present, resolve pronouns and implicit references in rewritten_query.
- If query asks about current events or real-time info → WEB.
- If query asks about relationships between entities → TOG.
- If query asks for broad overviews or trends → GLOBAL.
- If query asks about specific entities → LOCAL.
- Use DRIFT only when LOCAL might miss important related context.
- Default to LOCAL if uncertain.
- Consider method_used in history — prefer consistency unless query type clearly differs.
```

### History block format (injected at `{conversation_history_block}`)

```
Conversation history (most recent last):
[User] Tell me about Inception  →  rewritten: "Tell me about Inception"  →  method: local
[Assistant] Inception is a 2010 science fiction film directed by Christopher Nolan...
[User] Who directed it?
```

---

## Files Changed

| File | Change |
|---|---|
| `backend/app/models/schemas.py` | Add `ConversationTurn`; update `AgentSearchRequest`, `AgentSearchResponse` |
| `backend/app/services/router_agent.py` | Add `conversation_history` param to `route()`; update `RouteDecision`; add history formatter; update prompt parsing to extract `rewritten_query` |
| `backend/prompts/router_prompt.txt` | Add `{conversation_history_block}` placeholder and `rewritten_query` output field |
| `backend/app/routers/search.py` | Pass `conversation_history` from request to `router_agent.route()`; pass `rewritten_query` to search methods; include `rewritten_query` in response |

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single vs two LLM calls | Single combined call | Lower latency, lower cost, simpler failure surface |
| Who owns session state | Frontend | Stateless backend scales horizontally |
| History depth limit | 5 turns (10 messages) | Matches LocalSearch default; avoids prompt bloat |
| Rewrite always or only when history present | Always populate `rewritten_query` | Simplifies response parsing; frontend always gets it |
| Thread history to local/global/drift | Not in this feature | Separate concern; ToG already done; scope control |
| History format in prompt | Compact single-line per turn | Minimizes tokens, preserves key routing signals |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| LLM returns no `rewritten_query` field | Fall back to original `query` |
| LLM returns invalid `method` | Default to `"local"` (existing behavior) |
| `conversation_history` is empty or None | Skip history block in prompt entirely |
| History turn has malformed content | Skip that turn, log warning |

---

## Example Turn Sequence

**Turn 1**
```
Request:  { query: "Tell me about Inception" }
Rewrite:  "Tell me about Inception"   ← unchanged (no history)
Method:   local
Response: "Inception is a 2010 film directed by Christopher Nolan..."
```

**Turn 2**
```
Request:  { query: "Who directed it?", conversation_history: [turn1_user, turn1_assistant] }
Rewrite:  "Who directed Inception?"   ← pronoun resolved
Method:   local
Response: "Christopher Nolan directed Inception..."
```

**Turn 3**
```
Request:  { query: "What is his relationship to Emma Thomas?",
            conversation_history: [turn1, turn2_user, turn2_assistant] }
Rewrite:  "What is Christopher Nolan's relationship to Emma Thomas?"
Method:   tog   ← relationship query → router switches method correctly
Response: "Christopher Nolan and Emma Thomas are married and long-time collaborators..."
```

---

## Testing

- Unit test `RouterAgent.route()` with mocked LLM: verify `rewritten_query` populated correctly
- Unit test history formatter: verify correct turn count, role labels, method_used included
- Unit test fallback: missing `rewritten_query` in LLM response → original query used
- Integration test: multi-turn agent search via `/agent` endpoint, verify `rewritten_query` in response
