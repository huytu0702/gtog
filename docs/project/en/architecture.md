# Architecture Overview

This document describes the high-level architecture of the **GraphRAG with ToG Enhancement** system: how the frontend, backend, GraphRAG core library, and Azure infrastructure interact end-to-end.

## 1. System Context

The system is a knowledge-graph powered RAG application that:

1. Ingests user-uploaded documents into per-collection blob containers.
2. Builds a knowledge graph (entities, relationships, communities) via the GraphRAG indexing pipeline.
3. Serves five different search strategies (Global, Local, DRIFT, Basic, ToG) through a FastAPI backend.
4. Exposes a Next.js neo-brutalist frontend for collection management and conversational chat.

```mermaid
C4Context
    title System Context — GraphRAG + ToG

    Person(user, "End User", "Uploads docs, asks questions")

    System_Boundary(gtog, "GraphRAG.ToG Platform") {
        System(frontend, "Next.js Frontend", "UI for collections, indexing, chat")
        System(backend, "FastAPI Backend", "REST API, agent routing, guardrails")
        System(core, "GraphRAG Core Library", "Indexing pipeline + 5 query engines")
    }

    System_Ext(azure_blob, "Azure Blob Storage", "Raw uploaded documents")
    System_Ext(cosmos, "Azure Cosmos DB", "Control plane + pipeline output + conversation")
    System_Ext(queue, "Azure Storage Queue", "Indexing job dispatch")
    System_Ext(kv, "Azure Key Vault", "Secrets")
    System_Ext(llm, "OpenAI / Google", "Chat + embedding models")
    System_Ext(tavily, "Tavily Web Search", "Web fallback")

    Rel(user, frontend, "HTTPS")
    Rel(frontend, backend, "REST + SSE")
    Rel(backend, core, "Python imports")
    Rel(backend, azure_blob, "Read raw docs")
    Rel(backend, cosmos, "Metadata + pipeline output + conversations")
    Rel(backend, queue, "Enqueue jobs")
    Rel(backend, kv, "Fetch secrets")
    Rel(backend, llm, "Chat + embeddings")
    Rel(backend, tavily, "Web fallback search")
```

## 2. Logical Layers

```mermaid
graph TB
    subgraph FE["Frontend — Next.js 16 + React 19"]
        FE_Pages["Pages: /, /collections/:id"]
        FE_Comp["Components: NB UI, CollectionChat, CollectionDocuments"]
        FE_API["lib/api.ts — Axios client + types"]
    end

    subgraph BE["Backend — FastAPI"]
        BE_Routers["routers/<br/>collections, documents,<br/>indexing, search, conversation"]
        BE_Services["services/<br/>storage, indexing, query, conversation,<br/>router_agent, guardrails, web_search"]
        BE_Repos["repositories/<br/>control_plane, conversation, pipeline_output"]
        BE_Worker["worker.py — queue consumer"]
    end

    subgraph CORE["GraphRAG Core (graphrag/)"]
        CORE_Index["index/ — workflows + operations"]
        CORE_Query["query/ — global, local, drift, basic, tog"]
        CORE_Config["config/ — GraphRagConfig"]
        CORE_Storage["storage/ — file, blob, cosmos, memory"]
        CORE_Model["data_model/ — Entity, Relationship, Community..."]
    end

    subgraph INFRA["Azure Infrastructure"]
        INFRA_Blob["Blob Storage<br/>(per-collection containers)"]
        INFRA_Cosmos["Cosmos DB<br/>(control plane + pipeline output<br/>+ conversations)"]
        INFRA_Queue["Storage Queue<br/>(indexing-jobs)"]
        INFRA_KV["Key Vault"]
        INFRA_Search["AI Search (optional)"]
    end

    FE_Pages --> FE_Comp
    FE_Comp --> FE_API
    FE_API -->|REST + SSE| BE_Routers

    BE_Routers --> BE_Services
    BE_Services --> BE_Repos
    BE_Services --> CORE_Query
    BE_Services --> CORE_Index
    BE_Worker --> CORE_Index

    BE_Repos --> INFRA_Cosmos
    BE_Services --> INFRA_Blob
    BE_Services --> INFRA_Queue
    BE_Services --> INFRA_KV
    CORE_Storage --> INFRA_Blob
    CORE_Storage --> INFRA_Cosmos
```

## 3. Frontend Architecture

**Stack:** Next.js 16 (App Router), React 19, TypeScript, TanStack Query 5, Axios, Tailwind CSS v4.

**Routes:**

| Route | File | Purpose |
|---|---|---|
| `/` | `app/page.tsx` | Collections dashboard (list, create, delete) |
| `/collections/[id]` | `app/collections/[id]/page.tsx` | Collection detail with tabs (Documents, Chat) |
| `/api/health` | `app/api/health/route.ts` | Liveness probe for Docker |

**State management:**
- **Server state**: TanStack Query (`QueryClient`) — caches collections, documents, indexing status; auto-polls indexing status every 2s while running.
- **Local state**: per-component `useState` — chat messages, conversation history, search method selection, streaming controllers.

**API integration (`lib/api.ts`):**
- Axios instance with base URL `${NEXT_PUBLIC_API_BASE_URL}/api`.
- `validateStatus: () => true` + response interceptor that throws `ApiStatusError` for >= 400.
- SSE handled directly via `fetch()` with `ReadableStream` for `/search/agent/stream`.

**Design system — neo-brutalism:**
- 3px black borders, hard offset shadows (`4px 4px 0 0 #000`).
- Lime-green primary, pink secondary, cream background.
- Components: `NBButton`, `NBCard`, `NBInput`, `NBLayout`.

## 4. Backend Architecture

**Stack:** FastAPI, Pydantic v2, LiteLLM, Azure SDKs (cosmos, storage, identity, keyvault), NeMo Guardrails (optional), Tavily.

**Layered structure:**

```mermaid
graph LR
    Router["Routers<br/>(HTTP boundary)"] --> Service["Services<br/>(business logic)"]
    Service --> Repo["Repositories<br/>(data access)"]
    Repo --> Storage[("Cosmos / Blob / Queue")]
    Service --> Core["GraphRAG Core<br/>(graphrag.api.*)"]
    Service --> External["External LLMs<br/>(LiteLLM, Tavily)"]
```

**Routers** (`backend/app/routers/`):
- `collections.py` — CRUD for collections
- `documents.py` — Upload/list/delete documents
- `indexing.py` — Start/poll indexing jobs
- `search.py` — Direct + agent-routed search, web search, summarize
- `conversation.py` — Server-side session management

**Key services:**

| Service | Responsibility |
|---|---|
| `StorageService` | Collection + document CRUD across blob and Cosmos |
| `IndexingService` | Enqueue and track indexing jobs |
| `QueryService` | Orchestrate the 5 search methods, manage dataset cache |
| `RouterAgent` | LLM-based routing decision (which search method) |
| `InsufficiencyJudge` | Decides if web fallback is needed |
| `NemoGuardrailsService` | Input/output safety (deterministic + NeMo) |
| `ConversationService` | Server-side session persistence + recent turns |
| `SummarizationService` | Compress long conversations |
| `WebSearchService` | Tavily-backed web search with LLM synthesis |
| `QueueService` | Azure Storage Queue wrapper for job dispatch |

**Middleware (`main.py`):**
1. CORS (configurable origins).
2. Edge auth (`x-edge-secret`) when `REQUIRE_EDGE_AUTH=true`.
3. Trusted tunnel proxy detection (Cloudflare).
4. Rate limiting (in-memory, per-process).
5. Structured JSON logging with request IDs.

## 5. GraphRAG Core Library

The `graphrag/` package is consumed as a library by the backend (no separate process).

**Key sub-packages:**

| Package | Role |
|---|---|
| `graphrag.api` | Top-level entry points: `build_index`, `global_search`, `local_search`, `drift_search`, `basic_search`, `tog_search` |
| `graphrag.index` | Indexing pipeline (workflows + operations) |
| `graphrag.query` | 5 search engines (global, local, drift, basic, tog) |
| `graphrag.config` | `GraphRagConfig` Pydantic model + YAML loader |
| `graphrag.storage` | `PipelineStorage` abstraction (file, blob, cosmos, memory) |
| `graphrag.data_model` | `Entity`, `Relationship`, `Community`, `CommunityReport`, `TextUnit`, `Document`, `Covariate` |
| `graphrag.query.llm.tog` | ToG-specific exploration, pruning, reasoning |

See [index_flow.md](index_flow.md) and [query_flow.md](query_flow.md) for detailed flows.

## 6. Indexing Job Lifecycle (High-Level)

```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend
    participant API as FastAPI
    participant Cosmos as Cosmos (jobs)
    participant Queue as Storage Queue
    participant Worker as worker.py
    participant Core as graphrag.api.build_index
    participant Blob as Blob Storage

    FE->>API: POST /collections/{id}/index
    API->>Cosmos: Create job (status=pending)
    API->>Queue: Enqueue job message
    API-->>FE: 202 + job_id
    Worker->>Queue: Dequeue
    Worker->>Cosmos: Lease job (status=running)
    Worker->>Blob: Read input docs (pipeline-input container)
    Worker->>Core: build_index(config, output.type=cosmosdb)
    Core->>Cosmos: Write pipeline datasets to<br/>pipeline-{collection}-{version} container<br/>(parquet bytes — no files on disk)
    Worker->>Cosmos: Verify row counts + upsert artifactManifest
    Worker->>Cosmos: set_active_version(collection, version)
    Worker->>Cosmos: Mark completed (status=completed)
    FE->>API: GET /collections/{id}/index (polling)
    API->>Cosmos: Read job status
    API-->>FE: status=completed
```

## 7. Query / Agent Search Lifecycle (High-Level)

```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend
    participant API as FastAPI
    participant GR as Guardrails
    participant Conv as ConversationService
    participant Router as RouterAgent
    participant Q as QueryService
    participant Judge as InsufficiencyJudge
    participant Web as WebSearchService

    FE->>API: POST /search/agent (query, session_id?)
    API->>GR: check_input(query)
    GR-->>API: allow / block
    API->>Conv: get_prompt_context(session_id)
    Conv-->>API: summary + recent turns
    API->>Router: route(query, history)
    Router-->>API: method (local/global/tog/drift) + rewritten_query
    API->>GR: check_rewrite(query, rewritten)
    API->>Q: dispatch to chosen method
    Q-->>API: response + context_data
    API->>GR: check_output(graphrag_response)
    API->>Judge: judge(graphrag_response)
    Judge-->>API: sufficient? needs_web?
    alt needs web fallback
        API->>GR: check_web_query(rewritten_query)
        API->>Web: search(rewritten_query)
        Web-->>API: web_response + web_sources
        API->>GR: check_output(web_response)
    end
    API->>Conv: append_exchange(...)
    API-->>FE: AgentSearchResponse (or SSE stream)
```

## 8. Authentication & Security

- **Edge auth**: optional `x-edge-secret` header validated by middleware when `REQUIRE_EDGE_AUTH=true`.
- **Trusted tunnel**: Cloudflare Tunnel traffic from private IP ranges is recognized via `cf-ray` and `x-forwarded-for`.
- **Platform auth**: `REQUIRE_PLATFORM_AUTH=true` enables `X-MS-CLIENT-PRINCIPAL` validation (Azure Container Apps Easy Auth).
- **Secrets**: Pulled from Key Vault at startup via `azure_runtime.py`; never hardcoded.
- **Guardrails**: Deterministic regex checks + optional NeMo Rails for jailbreak/secret leakage detection.
- **Rate limiting**: Per-process in-memory limiter (not distributed — fronted by Cloudflare for production).

## 9. Deployment Topology (Production)

- **Frontend**: `ca-gtog-frontend-prod` (Azure Container Apps) — internal origin reachable only via Cloudflare Tunnel.
- **Backend**: `ca-gtog-api-prod` (Azure Container Apps) — Easy Auth `AllowAnonymous`, backend enforces auth via `REQUIRE_PLATFORM_AUTH=true`.
- **Public hostnames**: `app.gtog.id.vn` (frontend), `api.gtog.id.vn` (API).
- **Worker**: Same image as API, runs `python -m app.worker` to consume the indexing queue.
| **Storage** | Azure Blob (raw documents in `pipeline-input` container), Cosmos DB (control plane + pipeline output + conversations + on-demand vector containers `vec-{collection}-{version}-{embedding}`), Storage Queue (jobs). |

## 10. Cross-Cutting Concerns

| Concern | Implementation |
|---|---|
| **Configuration** | `backend/app/config.py` (Pydantic Settings) + `graphrag/settings.yaml` |
| **Secrets** | Azure Key Vault, fetched at startup |
| **Observability** | Structured JSON logs with `request_id`, `cf_ray`, latency, status |
| **Caching** | LRU dataset cache in `serving_context_cache.py` (configurable max entries) |
| **Async I/O** | All Azure SDK calls + LLM calls are async; SSE for streaming |
| **Versioning** | `semversioner` for changelog; image tags include date + commit |

## Related Docs

- [api.md](api.md) — REST API reference
- [database_schema.md](database_schema.md) — Cosmos containers + pipeline dataset schemas
- [index_flow.md](index_flow.md) — Detailed indexing pipeline
- [query_flow.md](query_flow.md) — Detailed query flows for all 5 methods
