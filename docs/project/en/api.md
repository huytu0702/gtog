# REST API Reference

This document is the complete reference for the **GraphRAG.ToG backend REST API**, served by FastAPI from `backend/app/main.py`.

- **Base URL** (local): `http://127.0.0.1:8000`
- **Base URL** (prod): `https://api.gtog.id.vn`
- **API prefix**: all endpoints below are relative to `/api`, except health endpoints which are at root.
- **Content-Type**: `application/json` unless noted (`multipart/form-data` for uploads).
- **OpenAPI**: `GET /docs` (Swagger UI) and `GET /openapi.json` (raw spec).

## 1. Conventions

### Authentication

| Header | Required when | Source |
|---|---|---|
| `x-edge-secret` | `REQUIRE_EDGE_AUTH=true` | Set by Cloudflare/edge layer |
| `X-MS-CLIENT-PRINCIPAL` | `REQUIRE_PLATFORM_AUTH=true` | Azure Container Apps Easy Auth |
| `cf-ray` | Trusted-tunnel detection | Cloudflare Tunnel |

The frontend does not send credentials directly; auth is terminated at the edge. CORS allows credentials from configured origins (default: `http://localhost:3000`).

### Rate limiting

Per-process in-memory limiter applies to all `/api/*` routes. Limits are configurable via `Settings`. On exceedance: `429 Too Many Requests`.

### Errors

All errors use FastAPI's standard envelope:

```json
{ "detail": "human-readable error message" }
```

| Status | Meaning |
|---|---|
| 400 | Validation error or invalid state (e.g., session collection mismatch) |
| 401 | Missing/invalid edge or platform auth |
| 403 | Authenticated but lacks permission |
| 404 | Resource not found (collection, document, session) |
| 409 | Conflict (e.g., collection name already exists) |
| 413 | Payload too large (document upload) |
| 422 | Pydantic validation error (FastAPI default) |
| 429 | Rate limit exceeded |
| 500 | Internal error |
| 503 | Dependency unavailable (Cosmos, Blob, queue) |

### Common identifiers

- `collection_id`: kebab-case, alphanumeric + `_-`, length 1–100. Pattern: `^[a-zA-Z0-9_-]+$`.
- `document_name`: original filename including extension (`.txt`, `.md`).
- `job_id`: UUID v4 string.
- `session_id`: opaque string, 1–128 chars.

## 2. Health

### `GET /health`

Liveness probe. Always returns 200 if the process is up.

**Response 200:**
```json
{ "status": "healthy", "service": "graphrag-api" }
```

### `GET /health/readiness`

Readiness probe. Checks Cosmos, Blob, Queue, AI Search, Key Vault.

**Response 200 (or 503):**
```json
{
  "status": "ready",
  "checks": {
    "cosmos":    { "ok": true,  "detail": "..." },
    "blob":      { "ok": true,  "detail": "..." },
    "queue":     { "ok": true,  "detail": "..." },
    "search":    { "ok": false, "detail": "not configured" },
    "key_vault": { "ok": true,  "detail": "..." }
  }
}
```

`status=not_ready` and HTTP 503 when any required dependency check fails.

### `GET /`

Root info endpoint.

**Response 200:**
```json
{ "name": "graphrag-api", "version": "1.0.0", "docs": "/docs" }
```

## 3. Collections

Manage logical knowledge bases. Each collection owns a Blob container and Cosmos partition.

### `POST /api/collections`

Create a new collection.

**Request:**
```json
{
  "name": "research-papers",
  "description": "Internal research documents"
}
```

| Field | Type | Constraints |
|---|---|---|
| `name` | string | required, `^[a-zA-Z0-9_-]+$`, 1–100 chars |
| `description` | string \| null | optional, max 500 chars |

**Response 201 — `CollectionResponse`:**
```json
{
  "id": "research-papers",
  "name": "research-papers",
  "description": "Internal research documents",
  "created_at": "2026-05-18T10:00:00Z",
  "document_count": 0,
  "indexed": false
}
```

**Errors:** `409` if a collection with the same name exists.

### `GET /api/collections`

List all collections.

**Response 200 — `CollectionList`:**
```json
{
  "collections": [
    { "id": "...", "name": "...", "description": null,
      "created_at": "...", "document_count": 3, "indexed": true }
  ],
  "total": 1
}
```

### `GET /api/collections/{collection_id}`

Get one collection.

**Response 200:** `CollectionResponse` (same shape as POST).

**Errors:** `404` if not found.

### `DELETE /api/collections/{collection_id}`

Delete a collection and **all** of its documents, indexing artifacts, and conversation sessions. Irreversible.

**Response:** `204 No Content`.

**Errors:** `404` if not found; `409` if an indexing job is currently running (cancel first).

## 4. Documents

Upload and manage source documents within a collection.

### `POST /api/collections/{collection_id}/documents`

Upload a document. `multipart/form-data`, single `file` field.

**Constraints:**
- Allowed extensions: `.txt`, `.md`.
- Max size: configured by `MAX_UPLOAD_SIZE_BYTES` (default 10 MB).

**Request example (curl):**
```bash
curl -X POST \
  -F "file=@notes.md" \
  https://api.gtog.id.vn/api/collections/research-papers/documents
```

**Response 201 — `DocumentResponse`:**
```json
{
  "name": "notes.md",
  "size": 4321,
  "uploaded_at": "2026-05-18T10:05:00Z"
}
```

**Errors:** `404` if collection not found; `409` if document name exists; `413` if too large; `415` for unsupported extension.

### `GET /api/collections/{collection_id}/documents`

List documents.

**Response 200 — `DocumentList`:**
```json
{
  "documents": [
    { "name": "notes.md", "size": 4321, "uploaded_at": "..." }
  ],
  "total": 1
}
```

### `DELETE /api/collections/{collection_id}/documents/{document_name}`

Delete a single document. Does **not** automatically re-index.

**Response:** `204 No Content`.

## 5. Indexing

### `POST /api/collections/{collection_id}/index`

Start indexing the collection. Returns immediately with a job reference; the actual work runs in the background worker.

**Response 202 — `IndexStatusResponse`:**
```json
{
  "collection_id": "research-papers",
  "job_id": "9c3d8e3a-...",
  "status": "pending",
  "progress": 0.0,
  "message": "Job queued",
  "attempt": 0,
  "max_attempts": 3,
  "started_at": null,
  "completed_at": null,
  "retry_at": null,
  "lease_owner_id": null,
  "heartbeat_at": null,
  "error": null
}
```

**Errors:** `409` if a job is already pending/running for this collection.

### `GET /api/collections/{collection_id}/index`

Get the current/latest indexing status for the collection. Polled by the frontend every 2s while running.

**Response 200:** `IndexStatusResponse` (statuses: `pending` | `running` | `retrying` | `completed` | `failed` | `cancelled`).

```json
{
  "collection_id": "research-papers",
  "job_id": "9c3d8e3a-...",
  "status": "running",
  "progress": 42.5,
  "message": "completed: extract_graph",
  "attempt": 1,
  "max_attempts": 3,
  "started_at": "2026-05-18T10:10:00Z",
  "completed_at": null,
  "retry_at": null,
  "lease_owner_id": "worker-1",
  "heartbeat_at": "2026-05-18T10:11:42Z",
  "error": null
}
```

### `GET /api/index-jobs/{job_id}`

Canonical job document by ID (any collection).

**Response 200 — `IndexJobResponse`:** same fields as `IndexStatusResponse`.

**Errors:** `404` if job not found.

## 6. Search

### Direct search methods

User-selected method, no routing or rewrite. Route-level guardrails still apply to the request and returned answer. Common shape:

| Endpoint | Method | Request body |
|---|---|---|
| `/api/collections/{id}/search/global` | POST | `GlobalSearchRequest` |
| `/api/collections/{id}/search/local`  | POST | `LocalSearchRequest` |
| `/api/collections/{id}/search/drift`  | POST | `DriftSearchRequest` |
| `/api/collections/{id}/search/tog`    | POST | `ToGSearchRequest` |

All return `SearchResponse`:

```json
{
  "query": "...",
  "response": "string | structured object",
  "context_data": { "...": "..." },
  "method": "global"
}
```

If input or output guardrails block a direct/manual route, the API still returns `SearchResponse` with the original `method` and a safe canned `response`.

#### `POST /search/global`

```json
{
  "query": "What are the major themes?",
  "streaming": false,
  "community_level": 1,
  "dynamic_community_selection": false,
  "response_type": "Multiple Paragraphs"
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | 1–1000 chars |
| `streaming` | bool | false | (currently non-streaming on direct endpoints) |
| `community_level` | int \| null | from config | 0–10 |
| `dynamic_community_selection` | bool | false | Score-based community selection |
| `response_type` | string | required | "Single Paragraph", "Multiple Paragraphs", "List of 3-7 Points", "Multiple Page Report", etc. |

#### `POST /search/local`

```json
{
  "query": "Tell me about Acme Corp's leadership",
  "streaming": false,
  "community_level": 2,
  "response_type": "Multiple Paragraphs"
}
```

#### `POST /search/drift`

Same shape as `/search/local`.

#### `POST /search/tog`

```json
{
  "query": "How are X and Y connected through Z?",
  "streaming": false,
  "max_depth": 3,
  "beam_width": 3,
  "show_exploration_paths": true
}
```

| Field | Type | Notes |
|---|---|---|
| `max_depth` | int \| null | Overrides `ToGSearchConfig.depth` |
| `beam_width` | int \| null | Overrides `ToGSearchConfig.width` |
| `show_exploration_paths` | bool | Reserved for UI/debug intent; GraphRAG ToG returns exploration paths in its native `context_data` contract |

ToG returns GraphRAG-native context data rather than the lookup tables used by Global, Local, and DRIFT search:

```json
{
  "context_data": {
    "exploration_paths": [
      "Entity A --[relationship]--> Entity B"
    ]
  }
}
```

The backend passes `entities`, `relationships`, and `text_units` into GraphRAG ToG for traversal and reasoning. The response preserves GraphRAG's native `exploration_paths`, adds a frontend-compatible `Relationships` lookup for explicit edge segments in those paths, and may add a `Sources` lookup for text units linked to explored entities so citation hover cards can show source chunks. It does not synthesize separate `Entities` or `RawContext` tables.

#### `GET /search/tog/debug`

Optional debug endpoint (enabled in non-prod). Returns a preview of entities used by ToG for the collection.

**Response 200:**
```json
{
  "collection_id": "...",
  "entity_count": 1234,
  "sample": [{ "id": "...", "title": "...", "type": "..." }]
}
```

### Agent-routed search

LLM picks the method, applies guardrails, may trigger web fallback, persists conversation.

#### `POST /api/collections/{collection_id}/search/agent`

Non-streaming variant.

**Request — `AgentSearchRequest`:**
```json
{
  "query": "What about its founder?",
  "stream": false,
  "session_id": "sess_abc123",
  "conversation_history": [
    { "role": "user", "content": "Tell me about Microsoft" },
    { "role": "assistant", "content": "Microsoft is..." }
  ],
  "conversation_summary": "Earlier: discussion of Microsoft's history."
}
```

| Field | Type | Notes |
|---|---|---|
| `query` | string | required, 1–1000 chars |
| `stream` | bool | false on this endpoint |
| `session_id` | string \| null | Server-side session for persistence |
| `conversation_history` | array \| null | Client-managed history (used if no session_id) |
| `conversation_summary` | string \| null | Compressed older context |

**Response 200 — `AgentSearchResponse`:**
```json
{
  "method_used": "tog",
  "router_reasoning": "Multi-hop entity question — ToG fits.",
  "rewritten_query": "Who founded Microsoft?",
  "response": "Bill Gates and Paul Allen founded Microsoft in 1975 [Data: Entities (12, 47)].",
  "sources": [],
  "context_data": { "...": "..." },
  "session_id": "sess_abc123",
  "web_response": null,
  "web_sources": [],
  "web_search_triggered": false
}
```

When the input, rewrite, or primary GraphRAG output is blocked by guardrails, `method_used="blocked"` and `response` contains a safe canned response.

When web fallback runs, `response` remains the GraphRAG answer. The synthesized web supplement is returned separately in `web_response` and `web_sources` only if it also passes output guardrails.

#### `GET|POST /api/collections/{collection_id}/search/agent/stream`

Streaming variant. Returns `text/event-stream` (SSE).

**GET query params** (used by EventSource): `query`, `session_id?`.

**POST body:** same as `/search/agent` (with `stream=true`).

**Event types:**

```
event: status
data: {"step":"routing","message":"Analyzing query..."}

event: status
data: {"step":"routed","method":"tog","rewritten_query":"Who founded Microsoft?","message":"Using TOG search"}

event: status
data: {"step":"searching","message":"Searching..."}

event: content
data: {"delta":"Bill Gates"}

event: content
data: {"delta":" and Paul Allen..."}

event: status
data: {"step":"judging_sufficiency","message":"Checking if indexed data is sufficient..."}

event: done
data: {
  "method_used":"tog",
  "router_reasoning":"...",
  "rewritten_query":"...",
  "response":"...full text...",
  "sources":[],
  "context_data":{...},
  "session_id":"sess_abc123",
  "web_response":null,
  "web_sources":[],
  "web_search_triggered":false
}
```

If web fallback runs but its synthesized answer is blocked by output guardrails, the `done` payload may still show `web_search_triggered:true` while `web_response:null` and `web_sources:[]`.

On error:
```
event: error
data: {"message":"Internal error while processing stream."}
```

#### `POST /api/collections/{collection_id}/search/agent/summarize`

Compress a conversation history into a running summary (used proactively by the frontend before history gets too long).

**Request — `SummarizeRequest`:**
```json
{
  "conversation_history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "existing_summary": "Earlier: ..."
}
```

**Response 200 — `SummarizeResponse`:**
```json
{
  "summary": "Updated rolling summary covering all turns.",
  "trimmed_history": [
    { "role": "user", "content": "most recent user turn" },
    { "role": "assistant", "content": "most recent assistant turn" }
  ]
}
```

#### `POST /api/collections/{collection_id}/search/web`

Direct web search (no GraphRAG). Useful for testing the Tavily integration. This route runs `check_web_query(query)` before Tavily search and `check_output(response)` after LLM synthesis.

**Request — `WebSearchRequest`:**
```json
{ "query": "latest GraphRAG paper" }
```

**Response 200:**
```json
{
  "query": "latest GraphRAG paper",
  "response": "Synthesized answer...",
  "sources": [
    { "id": 1, "title": "...", "url": "https://..." }
  ],
  "method": "web"
}
```

If either the web query or synthesized web answer is blocked by guardrails, the route still returns HTTP 200 with the same response shape, but `response` becomes the safe canned answer and `sources` is empty.

## 7. Conversation Sessions

Server-side session storage for multi-turn agent search.

### `POST /api/collections/{collection_id}/sessions`

Create a new session. The frontend may also let the server auto-create one on first agent call by passing a fresh `session_id`.

**Response 201 — `SessionCreateResponse`:**
```json
{
  "session_id": "sess_01HXYZ...",
  "collection_id": "research-papers",
  "created_at": "2026-05-18T10:20:00Z"
}
```

### `GET /api/collections/{collection_id}/sessions/{session_id}`

Get session metadata + the most recent turns (capped by `conversation_recent_user_turns`).

**Response 200 — `SessionDetailResponse`:**
```json
{
  "session_id": "sess_01HXYZ...",
  "collection_id": "research-papers",
  "summary": "Earlier: discussed Microsoft and its founders.",
  "turn_count": 12,
  "user_turn_count": 6,
  "created_at": "2026-05-18T10:20:00Z",
  "updated_at": "2026-05-18T10:35:00Z",
  "recent_turns": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "timestamp": "..." }
  ]
}
```

**Errors:** `404` if session not found, `400` if `session_id` belongs to a different collection.

## 8. Schema Reference (Pydantic)

Defined in `backend/app/models/schemas.py` and `enums.py`.

### Enums

```python
class SearchMethod(str, Enum):
    GLOBAL = "global"
    LOCAL  = "local"
    TOG    = "tog"
    DRIFT  = "drift"
    WEB    = "web"

class IndexStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    RETRYING   = "retrying"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
```

### Conversation models

```python
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None
    method_used: str | None = None
    rewritten_query: str | None = None
```

### Indexing models

```python
class IndexStatusResponse(BaseModel):
    collection_id: str
    job_id: str
    status: IndexStatus
    progress: float       # 0.0–100.0
    message: str | None
    attempt: int
    max_attempts: int
    started_at: datetime | None
    completed_at: datetime | None
    retry_at: datetime | None
    lease_owner_id: str | None
    heartbeat_at: datetime | None
    error: str | None
```

## 9. End-to-End Examples

### Example 1 — Full collection lifecycle

```bash
BASE=https://api.gtog.id.vn

# 1. Create collection
curl -X POST $BASE/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","description":"Quick demo"}'

# 2. Upload a document
curl -X POST $BASE/api/collections/demo/documents \
  -F "file=@./README.md"

# 3. Start indexing
curl -X POST $BASE/api/collections/demo/index

# 4. Poll until completed
while :; do
  curl -s $BASE/api/collections/demo/index | jq '.status'
  sleep 5
done

# 5. Run an agent search
curl -X POST $BASE/api/collections/demo/search/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this project about?","stream":false}'
```

### Example 2 — Streaming agent search (browser fetch)

```ts
const res = await fetch(`${BASE}/api/collections/demo/search/agent/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Summarize the key entities", stream: true }),
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  // Parse SSE frames: "event: <type>\ndata: <json>\n\n"
  console.log(decoder.decode(value));
}
```

## 10. Endpoint Summary

| Method | Path | Purpose |
|---|---|---|
| GET    | `/health` | Liveness |
| GET    | `/health/readiness` | Readiness (deps) |
| POST   | `/api/collections` | Create collection |
| GET    | `/api/collections` | List collections |
| GET    | `/api/collections/{id}` | Get collection |
| DELETE | `/api/collections/{id}` | Delete collection |
| POST   | `/api/collections/{id}/documents` | Upload doc |
| GET    | `/api/collections/{id}/documents` | List docs |
| DELETE | `/api/collections/{id}/documents/{name}` | Delete doc |
| POST   | `/api/collections/{id}/index` | Start indexing |
| GET    | `/api/collections/{id}/index` | Status |
| GET    | `/api/index-jobs/{job_id}` | Job by ID |
| POST   | `/api/collections/{id}/search/global` | Global search |
| POST   | `/api/collections/{id}/search/local` | Local search |
| POST   | `/api/collections/{id}/search/drift` | DRIFT search |
| POST   | `/api/collections/{id}/search/tog` | ToG search |
| GET    | `/api/collections/{id}/search/tog/debug` | ToG debug |
| POST   | `/api/collections/{id}/search/agent` | Agent search |
| GET    | `/api/collections/{id}/search/agent/stream` | Agent stream (SSE) |
| POST   | `/api/collections/{id}/search/agent/stream` | Agent stream (POST SSE) |
| POST   | `/api/collections/{id}/search/agent/summarize` | Summarize history |
| POST   | `/api/collections/{id}/search/web` | Web search |
| POST   | `/api/collections/{id}/sessions` | Create session |
| GET    | `/api/collections/{id}/sessions/{sid}` | Get session |

## Related Docs

- [architecture.md](architecture.md) — System overview
- [query_flow.md](query_flow.md) — Behavior behind `/search/*` endpoints
- [index_flow.md](index_flow.md) — Behavior behind `/index` endpoints
- [database_schema.md](database_schema.md) — Persistence model behind these endpoints
