# Multi-Agent Search System with Web Fallback

**Date:** 2026-02-05
**Status:** Implemented
**Branch:** feat/web_search

## Overview

Add a multi-agent search system that intelligently routes queries to the optimal search method, including a new web search fallback for queries that cannot be answered from indexed documents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-AGENT SEARCH SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query                                                                │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       ROUTER AGENT (LLM)                            │   │
│   │  • Analyzes query intent                                            │   │
│   │  • Considers available data sources                                 │   │
│   │  • Selects optimal search method                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│           ┌───────────┬───────────┼───────────┬───────────┐                 │
│           ▼           ▼           ▼           ▼           ▼                 │
│       ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐            │
│       │ LOCAL │   │GLOBAL │   │  TOG  │   │ DRIFT │   │  WEB  │            │
│       │Search │   │Search │   │Search │   │Search │   │Search │            │
│       └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘            │
│           │           │           │           │           │                 │
│           └───────────┴───────────┴─────┬─────┴───────────┘                 │
│                                         ▼                                   │
│                                Response (Streaming)                         │
│                                + Source Citations                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Router Agent (decide upfront) | Simple, predictable, lower LLM cost |
| Location | Backend only | Keep graphrag/ pure as library |
| Router Mechanism | LLM-based classification | Flexible, handles nuanced queries |
| Search Methods | local, global, tog, drift, web | Full capability set |
| Web Provider | Tavily | AI-optimized, clean structured results |
| Response Format | LLM synthesis + citations | Polished answers with source transparency |
| Streaming | Full streaming with status events | Good UX for long responses |
| Observability | Real-time UI + detailed dev logs | Production + debugging visibility |

## Search Methods

| Method | Use When |
|--------|----------|
| **GLOBAL** | Questions requiring understanding of dataset as a whole. Overview, trends, summaries across entire corpus. Example: "What are the most significant values across all herbs?" |
| **LOCAL** | Questions about specific entities mentioned in documents. Focused queries on particular topics/names/concepts. Example: "What are the healing properties of chamomile?" |
| **DRIFT** | Local search + community context for broader variety. Expands query into detailed follow-up questions. Example: "How do different herbs compare for sleep disorders?" |
| **TOG** | Questions about relationships between entities. Multi-hop reasoning through entity connections. Example: "What is the relationship between chamomile and lavender?" |
| **WEB** | External information not in documents. Real-time/current events, topics outside indexed data. Example: "What are the latest FDA regulations on herbal supplements?" |

## Router Agent

**Location:** `backend/app/services/router_agent.py`

```python
class RouterAgent:
    def __init__(self, llm_client, collection_metadata):
        self.llm = llm_client
        self.metadata = collection_metadata  # Topics, entities in indexed docs

    async def route(self, query: str) -> SearchMethod:
        # Returns: local | global | tog | drift | web
```

**Response Format:**
```json
{
  "method": "local",
  "confidence": 0.85,
  "reasoning": "Query asks about specific entity mentioned in documents"
}
```

## Web Search

**Location:** `backend/app/services/web_search.py`

```python
class WebSearchService:
    def __init__(self, tavily_api_key: str, llm_client):
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.llm = llm_client

    async def search(self, query: str) -> WebSearchResult:
        # 1. Call Tavily API
        # 2. Synthesize with LLM (include inline citations)
        # 3. Return response with sources

    async def search_streaming(self, query: str) -> AsyncIterator[str]:
        # Streaming version for SSE endpoints
```

**Response Format:**
```json
{
  "response": "The EU AI Act came into force in August 2024 [1]. The US issued Executive Order 14110 [2]...",
  "sources": [
    {"id": 1, "title": "EU AI Act Implementation", "url": "https://europa.eu/..."},
    {"id": 2, "title": "Biden's AI Executive Order", "url": "https://whitehouse.gov/..."}
  ]
}
```

## API Endpoints

**New Endpoints:**

```
POST /api/collections/{collection_id}/search/agent
     └─► Router Agent endpoint - auto-selects best method

POST /api/collections/{collection_id}/search/web
     └─► Direct web search endpoint (bypass router)
```

**Request Schema:**
```python
class AgentSearchRequest(BaseModel):
    query: str
    stream: bool = True
```

**Response Schema:**
```python
class AgentSearchResponse(BaseModel):
    method_used: str          # "local" | "global" | "tog" | "drift" | "web"
    router_reasoning: str     # Why this method was chosen
    response: str             # The answer with citations
    sources: list[Source]     # Citation sources

class Source(BaseModel):
    id: int
    title: str
    url: str | None           # URL for web sources
    text_unit_id: str | None  # Text unit ID for GraphRAG sources
```

## Streaming Status Events

**SSE Event Stream:**
```
event: status
data: {"step": "routing", "message": "Analyzing query..."}

event: status
data: {"step": "routed", "method": "local", "message": "Using LOCAL search"}

event: status
data: {"step": "searching", "message": "Searching knowledge graph..."}

event: content
data: {"delta": "Chamomile has been used..."}

event: done
data: {"sources": [...], "method_used": "local"}
```

**Frontend Display:**
```
⚡ Analyzing query...          ← step: routing
🎯 Using LOCAL search          ← step: routed
🔍 Searching knowledge graph...← step: searching
✍️ Generating response...      ← step: generating

Chamomile has been used for centuries... [1]

Sources:
[1] herbs-chapter-3.txt
```

## Development Logs

```python
logger.debug({
    "event": "router_decision",
    "query": "What are healing properties of chamomile?",
    "method": "local",
    "confidence": 0.92,
    "reasoning": "Query asks about specific entity 'chamomile' properties",
    "alternatives": [
        {"method": "global", "score": 0.31},
        {"method": "web", "score": 0.15}
    ],
    "latency_ms": 245
})
```

## Error Handling

| Scenario | Action |
|----------|--------|
| Router fails to decide | Default to LOCAL search |
| Tavily API error | Return error message, suggest retry |
| Web search returns no results | LLM responds: "No relevant results found..." |
| Collection not indexed | Auto-fallback to WEB search |
| LLM synthesis fails | Return raw Tavily results |

## File Structure

**New Files:**
```
backend/
├── app/
│   ├── services/
│   │   ├── router_agent.py      # Router Agent logic
│   │   └── web_search.py        # Tavily integration
│   └── models/
│       └── events.py            # SSE event schemas
├── prompts/
│   ├── router_prompt.txt        # Router decision prompt
│   └── web_synthesis_prompt.txt # Web result synthesis prompt

frontend/
└── app/
    └── components/
        └── AgentStatus.tsx      # Status display component
```

**Modified Files:**
- `backend/app/routers/search.py` - Add /agent and /web endpoints
- `backend/app/models/search.py` - Add new schemas

## Dependencies

```toml
# backend/pyproject.toml
tavily-python = "^0.3.0"
```

## Environment Variables

```bash
# backend/.env
TAVILY_API_KEY=tvly-xxxxx
```
