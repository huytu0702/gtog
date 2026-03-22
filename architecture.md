# Architecture: GraphRAG with ToG Enhancement

> A comprehensive architectural reference for the GToG (Graph-RAG + Think-on-Graph) system —
> a Microsoft Research GraphRAG project enhanced with iterative, beam-search-driven deep reasoning
> over knowledge graphs.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Frontend Layer](#3-frontend-layer)
4. [Backend Layer](#4-backend-layer)
5. [GraphRAG Core Library](#5-graphrag-core-library)
6. [ToG (Think-on-Graph) Engine](#6-tog-think-on-graph-engine)
7. [Indexing Pipeline](#7-indexing-pipeline)
8. [Query Engine](#8-query-engine)
9. [Data Flow](#9-data-flow)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Technology Stack](#11-technology-stack)
12. [Configuration Model](#12-configuration-model)
13. [Security Model](#13-security-model)

---

## 1. System Overview

GToG is a **knowledge-graph-based RAG system** that transforms unstructured documents into a structured knowledge graph, then enables five complementary search strategies over that graph — including **ToG (Think-on-Graph)**, an ICLR 2024 research algorithm for multi-hop, chain-of-thought reasoning.

### Why GraphRAG over Baseline RAG?

Standard vector-similarity RAG fails at two classes of questions:

| Problem | Baseline RAG | GraphRAG |
|---------|-------------|----------|
| Multi-hop reasoning across disparate facts | Fails — no traversal | Supported via graph paths |
| Holistic summarization of a large corpus | Fails — no hierarchy | Supported via community reports |

GraphRAG builds a **hierarchical knowledge graph** (entities → relationships → communities) and lets each query method pick the most appropriate slice of that graph.

### The ToG Advantage

ToG adds a sixth dimension: **iterative beam-search exploration** with LLM-guided pruning, producing explicit, auditable reasoning paths rather than opaque context windows.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            User Browser                                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTPS
                    ┌───────────▼──────────┐
                    │   Cloudflare Edge    │
                    │  DNS · WAF · Rate    │
                    │  Limit · Cache Bypass│
                    └───────────┬──────────┘
                                │ Cloudflare Tunnel
           ┌────────────────────▼────────────────────┐
           │          Azure Container Apps            │
           │         (Private Environment)            │
           │                                          │
           │  ┌────────────┐    ┌────────────────┐   │
           │  │  Frontend  │    │   API Backend  │   │
           │  │  Next.js   │    │   FastAPI      │   │
           │  │  :3000     │    │   :8000        │   │
           │  └────────────┘    └───────┬────────┘   │
           │                           │              │
           │  ┌────────────────────────▼───────────┐  │
           │  │       Indexing Worker (no ingress) │  │
           │  └────────────────────────────────────┘  │
           │                                          │
           │  Azure Cosmos DB  ·  Azure Blob Storage  │
           │  Azure Storage Queue  ·  Azure Key Vault │
           │  Log Analytics                          │
           └──────────────────────────────────────────┘
```

### Component Summary

| Component | Role | Runtime |
|-----------|------|---------|
| **Frontend** | Neo-brutalist SPA for collections, documents, and chat | Next.js 16 on Node 20 |
| **API Backend** | REST + SSE gateway; routes queries, manages jobs | FastAPI (Python 3.11) |
| **Indexing Worker** | Async consumer of indexing jobs from queue | Same image, `APP_ROLE=worker` |
| **GraphRAG Core** | Knowledge graph build + query library | Python package |
| **Cosmos DB** | **Unified Storage**: Control-plane metadata + Vector Store | Azure PaaS |
| **Blob Storage** | Raw documents + indexed Parquet artifacts | Azure PaaS |
| **Storage Queue** | Durable indexing job dispatch | Azure PaaS |
| **Key Vault** | Runtime secret management | Azure PaaS |

---

## 3. Frontend Layer

### Technology

- **Framework**: Next.js 16 with React 19 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 4 — neo-brutalist design system (heavy borders, shadow-hard utilities)
- **State / Data Fetching**: TanStack Query v5 (React Query) + Axios
- **Icons**: Lucide React

### Application Structure

```
frontend/
├── app/
│   ├── layout.tsx                # Root layout with providers (QueryClient, etc.)
│   ├── page.tsx                  # Dashboard — collection list + create/delete
│   ├── providers.tsx             # TanStack Query provider
│   ├── globals.css               # Global Tailwind styles
│   ├── collections/
│   │   └── [id]/
│   │       ├── page.tsx          # Collection detail shell
│   │       ├── collection-chat.tsx    # Chat UI — search method selector + conversation
│   │       └── collection-documents.tsx  # Document upload + indexing trigger
│   └── api/
│       └── health/route.ts       # Next.js health probe
├── components/
│   └── ui/
│       ├── NBButton.tsx          # Neo-brutalist button
│       ├── NBCard.tsx            # Neo-brutalist card
│       └── NBInput.tsx           # Neo-brutalist input
└── lib/
    └── api.ts                    # Typed Axios client — all API calls
```

### Key UI Flows

1. **Dashboard** — CRUD for collections; each card shows document count and index status.
2. **Collection Detail** — two tabs:
   - *Documents*: upload files, trigger indexing, monitor job status via polling.
   - *Chat*: select search method (Global / Local / DRIFT / ToG / Agent), send queries, receive SSE-streamed responses.
3. **Agent Search** — calls `/agent/stream` SSE endpoint; frontend renders routing decision and progressive content chunks.

### API Communication

```
lib/api.ts  →  /api/collections/*   (CRUD, upload, indexing status)
           →  /api/collections/{id}/search/global|local|drift|tog|agent
           →  /api/collections/{id}/search/agent/stream  (EventSource SSE)
```

---

## 4. Backend Layer

### Technology

- **Framework**: FastAPI 0.x (ASGI via Uvicorn)
- **Language**: Python 3.11
- **Validation**: Pydantic v2 + pydantic-settings
- **Container mode**: `APP_ROLE=api` (Uvicorn) or `APP_ROLE=worker` (async queue loop)

### Module Structure

```
backend/app/
├── main.py                # FastAPI app factory, middleware, health endpoints
├── config.py              # Pydantic Settings — all env-var driven
├── azure_runtime.py       # Azure SDK bootstrap (Key Vault, MSI, Cosmos)
├── worker.py              # Standalone indexing worker entry point
├── errors.py              # Domain exception classes
├── models/                # Pydantic request/response schemas
├── repositories/          # Data access — Cosmos DB control-plane repo
├── routers/
│   ├── collections.py     # GET/POST/DELETE /api/collections
│   ├── documents.py       # File upload, document metadata
│   ├── indexing.py        # Submit indexing job
│   ├── indexing_jobs.py   # Job status polling
│   ├── search.py          # All search endpoints (global/local/drift/tog/agent)
│   └── conversation.py    # Session create, history append
└── services/
    ├── query_service.py              # Dispatcher: routes to per-method services
    ├── query_service_global.py
    ├── query_service_local.py
    ├── query_service_drift.py
    ├── query_service_tog.py          # ToG-specific adapter
    ├── query_service_base.py         # Shared helpers (config load, citation normalization)
    ├── indexing_service.py           # Job lifecycle management
    ├── queue_service.py              # Azure Storage Queue client
    ├── conversation_service.py       # Session + turn persistence to Cosmos
    ├── serving_context_cache.py      # LRU cache — loaded graph frames per collection
    ├── serving_materialization_service.py  # Parquet/Cosmos frame hydration
    ├── router_agent.py               # LLM-based query routing + rewriting
    ├── summarization_service.py      # Conversation history compression
    ├── web_search.py                 # Tavily-backed web search
    └── storage_service.py            # Blob storage abstraction
```

### Middleware Stack (in order)

```
Request → CORS preflight check
        → Security & logging middleware:
            ├── Edge auth check (X-Edge-Secret header or Cloudflare Tunnel IP)
            ├── In-memory rate limiter (per-IP, sliding window)
            └── Structured JSON request log (method, path, status, latency_ms, cf-ray)
        → Router handlers
        → CORS header injection (response)
        → X-Request-Id header injection
```

### Router Inventory

| Prefix | Purpose |
|--------|---------|
| `GET/POST/DELETE /api/collections` | Collection management |
| `POST /api/collections/{id}/documents` | Document upload |
| `GET /api/collections/{id}/documents` | List documents |
| `POST /api/collections/{id}/index` | Submit indexing job |
| `GET /api/collections/{id}/index/jobs` | List jobs |
| `GET /api/collections/{id}/index/jobs/{job_id}` | Job status |
| `POST /api/collections/{id}/search/{method}` | Direct search (global/local/drift/tog) |
| `POST /api/collections/{id}/search/agent` | Agent-routed search |
| `GET/POST /api/collections/{id}/search/agent/stream` | SSE streaming agent search |
| `POST /api/collections/{id}/search/agent/summarize` | Conversation compression |
| `POST /api/collections/{id}/search/web` | Direct Tavily web search |
| `GET /health` | Liveness probe |
| `GET /health/readiness` | Readiness probe (Cosmos, Blob, Queue, KV) |

### Indexing Worker

The same Docker image runs as the worker when `APP_ROLE=worker`:

```
worker.py
  └── _run_worker_loop()
        ├── requeue_recoverable_jobs()       # recover stale/crashed jobs
        ├── queue_service.receive_messages() # poll Azure Storage Queue
        ├── control_plane.acquire_indexing_job_lease()
        ├── indexing_service.run_indexing()  # calls graphrag.api.index()
        └── serving_materialization_service  # warm serving cache on completion
```

The worker uses a **lease pattern** — it acquires a time-limited lease before processing to prevent duplicate execution across replicas.

### Agent Router

`router_agent.py` performs a **single LLM call** that simultaneously:
1. Rewrites the user query for clarity/context.
2. Selects the optimal search method (`global`, `local`, `drift`, `tog`, `web`).
3. Returns a confidence score and reasoning explanation.

This decision is surfaced to the frontend as part of the SSE `status` events.

---

## 5. GraphRAG Core Library

```
graphrag/
├── api/                   # Public Python API surface (index, query functions)
├── cache/                 # Cache adapters (filesystem, blob, noop)
├── callbacks/             # Workflow + query callback protocols
├── cli/                   # CLI entry points (main.py, index.py, query.py, eval.py)
├── config/                # Config models, YAML loader, env reader
├── data_model/            # Canonical data classes: Entity, Relationship, Community, etc.
├── eval/                  # Evaluation harness
├── factory/               # Generic factory utilities
├── index/                 # Indexing pipeline (workflows, steps, run/)
├── language_model/        # LLM/embedding protocol + provider adapters (fnllm, litellm)
├── logger/                # Structured logger factory
├── prompt_tune/           # Automatic prompt tuning module
├── prompts/               # Jinja2 prompt templates
│   └── query/             # ToG-specific prompts (relation scoring, entity scoring, reasoning)
├── query/                 # Query engine
│   ├── context_builder/   # Context assembly + conversation history
│   ├── factory.py         # Engine factory functions (get_tog_search_engine, etc.)
│   ├── indexer_adapters.py
│   └── structured_search/
│       ├── base.py            # SearchResult dataclass
│       ├── basic_search/
│       ├── drift_search/
│       ├── global_search/
│       ├── local_search/
│       └── tog_search/        # ToG implementation (see §6)
├── storage/               # Storage adapters (blob, file, memory)
├── tokenizer/             # Token counting
├── utils/                 # Shared utilities
└── vector_stores/         # Vector store adapters (cosmosdb, azure ai search, lancedb, etc.)
```

### Data Model

```
TextUnit       ← raw document chunk (source of truth for citations)
Entity         ← extracted named entity (id, title, type, description, embedding)
Relationship   ← directed edge between entities (source, target, description, weight)
Covariate      ← optional claims extracted per entity
Community      ← Leiden-detected entity cluster
CommunityReport← LLM-generated summary of a community
```

---

## 6. ToG (Think-on-Graph) Engine

ToG is an ICLR 2024 algorithm adapted from the original academic paper. It replaces static context retrieval with **iterative beam-search graph traversal** guided by LLM scoring.

### Module Map

```
graphrag/query/structured_search/tog_search/
├── search.py       # ToGSearch — orchestrator; beam-search loop
├── state.py        # ToGSearchState + ExplorationNode — immutable traversal tree
├── exploration.py  # GraphExplorer — adjacency index, entity linking (semantic + keyword)
├── pruning.py      # PruningStrategy ABC; LLMPruning, SemanticPruning, BM25Pruning
└── reasoning.py    # ToGReasoning — chain-of-thought final answer generation
```

### Algorithm Walkthrough

```
INPUT: query, entities[], relationships[], config(width, depth, prune_strategy)

1. ENTITY LINKING
   ├── Embed query (EmbeddingModel.aembed)
   ├── Dot-product similarity against pre-computed entity embeddings
   └── Top-k entities → initial frontier (depth=0)

2. EARLY TERMINATION CHECK
   └── Can depth-0 entities alone answer the query? (ToGReasoning.check_early_termination)
       → YES: stream answer + reasoning paths, return
       → NO:  continue

3. BEAM-SEARCH LOOP  (repeat depth=1..max_depth)
   │
   ├── For each node in current frontier:
   │   ├── GraphExplorer.get_relations(entity_id)   ← bidirectional adjacency lookup
   │   ├── PruningStrategy.score_relations()         ← score + filter to top-width
   │   └── PruningStrategy.score_entities()          ← second-stage entity prune
   │
   ├── Create ExplorationNode children (combined score = parent_score × hop_score)
   ├── ToGSearchState.prune_current_frontier()       ← keep top beam_width nodes
   └── Early termination check at each depth

4. FINAL REASONING
   └── ToGReasoning.generate_answer(query, all_explored_nodes)
       ├── Format paths: "A --[rel]--> B --[rel]--> C"
       ├── Build reasoning prompt from TOG_REASONING_PROMPT template
       └── LLM call (temperature=0.0 for determinism) → final answer + path citations
```

### Pruning Strategies

| Strategy | Mechanism | Speed | Quality |
|----------|-----------|-------|---------|
| `LLMPruning` | LLM scores each relation/entity 0–10 | Slower | Highest |
| `SemanticPruning` | Embedding cosine similarity | Fast | High |
| `BM25Pruning` | BM25 keyword relevance | Fastest | Good for lexical queries |

### Key Data Structures

```python
@dataclass
class ExplorationNode:
    entity_id: str
    entity_name: str
    entity_description: str
    depth: int
    score: float            # cumulative beam score (decays with depth)
    parent: ExplorationNode | None
    relation_from_parent: str | None
    relation_full_description: str | None
    entity_full_description: str | None

@dataclass
class ToGSearchState:
    query: str
    current_depth: int
    nodes_by_depth: dict[int, list[ExplorationNode]]  # depth → frontier
    finished_paths: list[ExplorationNode]
    max_depth: int
    beam_width: int
```

### Configuration Reference

```yaml
tog_search:
  chat_model_id: default_chat_model       # model for pruning + reasoning
  embedding_model_id: default_embedding_model

  # Beam search
  width: 3                  # paths to keep per depth level
  depth: 3                  # max graph hops

  # Pruning
  prune_strategy: llm       # llm | semantic | bm25
  num_retain_entity: 5      # entity candidates sampled per hop

  # Temperature
  temperature_exploration: 0.4   # stochastic exploration
  temperature_reasoning: 0.0     # deterministic final answer

  # Resource limits
  max_context_tokens: 8000
  max_exploration_paths: 10
```

### LLM Call Budget

ToG makes multiple LLM calls per query. Approximate call count:

```
relation_scoring:  width × depth × |frontier|  calls
entity_scoring:    width × depth × |frontier|  calls
early_termination: depth + 1                   calls
final_reasoning:   1–2                         calls
```

Use `semantic` pruning and smaller `width`/`depth` to reduce cost in production.

### Comparison with Other Search Methods

| Dimension | ToG | Local | Global | DRIFT | Basic |
|-----------|-----|-------|--------|-------|-------|
| Best for | Multi-hop reasoning, path-finding | Specific entities | Thematic summaries | Balanced local+community | Simple keyword |
| Reasoning depth | Multi-hop | Single-hop | Aggregated | Mixed | None |
| Explainability | **High** — explicit paths | Medium | Low | Medium | Low |
| LLM calls | Many | Few | Many (map-reduce) | Medium | Minimal |
| Speed | Slowest | Fast | Medium | Medium | Fastest |

---

## 7. Indexing Pipeline

### Overview

The indexing pipeline converts raw text documents into a structured knowledge graph stored as Parquet files and vector embeddings.

### Pipeline Stages

```
Raw Documents (txt, pdf, json…)
        │
        ▼
┌─────────────────────────────┐
│ 1. Document Loading         │  Read from local filesystem or Azure Blob
│    & Chunking (TextUnits)   │  Split into overlapping text chunks
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. Entity Extraction        │  LLM extracts: organizations, people,
│                             │  locations, events, concepts
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. Relationship Building    │  LLM identifies semantic relationships
│                             │  between extracted entities
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. Community Detection      │  Leiden algorithm clusters entities
│                             │  into hierarchical communities
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. Embedding Generation     │  Entity description embeddings
│                             │  Text chunk embeddings → vector store
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 6. Community Report Gen     │  LLM bottom-up summaries per community
│                             │  for global search context
└──────────────┬──────────────┘
               │
               ▼
    Parquet Artifacts (output/)
    create_final_entities.parquet
    create_final_relationships.parquet
    create_final_communities.parquet
    create_final_community_reports.parquet
    create_final_text_units.parquet
    create_final_covariates.parquet
    + Vector store embeddings
```

### Incremental Indexing

`graphrag update` performs **delta indexing** — only newly added documents are processed and merged with the existing index. The pipeline tracks state via `context.json` in output storage.

### Job Lifecycle (Production)

```
POST /api/collections/{id}/index
        │
        ▼
IndexingService.submit_job()
  → Cosmos DB: job record (status=queued)
  → Azure Storage Queue: {job_id} message
        │
        ▼
Worker.receive_messages()
  → Acquire lease (prevent duplicate processing)
  → graphrag.api.index(config, callbacks)
        │
  ┌─────▼──────┐
  │  Running   │ Heartbeat every 30s to Cosmos
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │ Completed  │ Artifacts written to Blob
  └─────┬──────┘
        │
  serving_materialization_service.warm_cache()
        │
  Search immediately available
```

---

## 8. Query Engine

### Factory Pattern

All search engines are created via `graphrag/query/factory.py`:

```python
get_local_search_engine(config, reports, text_units, entities, relationships, ...)
get_global_search_engine(config, reports, entities, relationships, ...)
get_drift_search_engine(config, reports, entities, relationships, ...)
get_tog_search_engine(config, entities, relationships, ...)
get_basic_search_engine(config, text_units, ...)
```

Each factory function:
1. Resolves the LLM and embedding model from `GraphRagConfig`.
2. Initializes the appropriate context builder.
3. Returns a configured search engine instance.

### Serving Context Cache

The backend maintains an **LRU cache** of loaded graph frames (entities, relationships, community reports, text units) per collection. This avoids re-reading Parquet files on every query.

```python
class ServingContextCache:
    # Max entries: settings.serving_dataset_cache_max_entries (default: 96)
    # TTL: settings.cache_ttl_seconds (default: 1800s)
    # Keyed by: (collection_id, method, version)
```

On indexing completion, the worker triggers cache warm-up via `serving_materialization_service`.

### Query Context Modes

The backend supports two context-loading modes (controlled by `QUERY_CONTEXT_MODE`):

| Mode | Source | Use Case |
|------|--------|---------|
| `cosmos_only` | Cosmos DB containers (entities, relationships, etc.) | Production — all artifacts in managed storage |
| `blob_parquet` | Parquet files from Azure Blob / local filesystem | Dev / migration |

---

## 9. Data Flow

### Indexing Flow

```
User uploads files via Frontend
  → POST /api/collections/{id}/documents  (Backend)
  → Files stored in Azure Blob Storage
  → POST /api/collections/{id}/index      (Backend)
  → Job record created in Cosmos DB
  → Job ID enqueued in Azure Storage Queue
  → Worker dequeues, acquires lease
  → graphrag.api.index() runs the full pipeline
  → Parquet artifacts written to Blob
  → Entity/relationship data written to Cosmos DB containers
  → Vector embeddings written to Cosmos DB (vector index)
  → Worker marks job "completed" in Cosmos DB
  → Serving cache warmed
  → Frontend polls GET .../index/jobs/{id} until completed
```

### Query Flow (ToG Example)

```
User types query in Frontend → selects "ToG" method
  → POST /api/collections/{id}/search/tog
  → Backend: query_service.tog_search(collection_id, query)
  → Serving cache: load entities + relationships (Cosmos or Parquet)
  → graphrag.api.tog_search(config, entities, relationships, query)
  → ToGSearch.search(query)
      ├── GraphExplorer.find_starting_entities_semantic(query)  [embedding lookup]
      ├── Loop: score_relations → score_entities → expand → prune
      └── ToGReasoning.generate_answer(query, all_paths)
  → SearchResult(response, context_data={exploration_paths}, metrics)
  → SearchResponse returned to Frontend
  → Frontend renders answer + path visualization
```

### Agent-Routed Streaming Flow

```
User types query → selects "Agent" mode
  → GET /api/collections/{id}/search/agent/stream  (EventSource)
  → SSE event: status=routing
  → router_agent.route(query)  [single LLM call]
  → SSE event: status=routed, method=<chosen>, rewritten_query=<...>
  → SSE event: status=searching
  → Appropriate search method executed
  → SSE events: content chunks (50-char deltas)
  → SSE event: done, method_used, router_reasoning, session_id
  → Heartbeat every 25s (SSE ping, keepalive)
```

---

## 10. Deployment Architecture

### Production Topology (v4)

```
                    ┌──────────────────────────────┐
  User Browser ───► │       Cloudflare Edge         │
                    │  Proxied DNS · WAF Rules       │
                    │  Rate Limit on api.gtog.id.vn │
                    │  Cache-bypass: /api/* + SSE    │
                    └──────────────┬───────────────┘
                                   │ Cloudflare Tunnel
                    ┌──────────────▼───────────────┐
                    │  cloudflared connectors (×2)  │  RFC 6598 / RFC 1918 IPs
                    │  (ACA internal networking)    │
                    └─────┬─────────────┬──────────┘
                          │             │
               ┌──────────▼──┐   ┌──────▼──────────┐
               │  Frontend   │   │   API Backend    │
               │  Next.js    │   │   FastAPI        │
               │  :3000      │   │   :8000          │
               │  internal   │   │   internal       │
               │  ingress    │   │   ingress        │
               └─────────────┘   └────────┬─────────┘
                                          │
                              ┌───────────▼──────────┐
                              │   Indexing Worker    │
                              │   (no ingress)       │
                              └───────────┬──────────┘
                                          │
               ┌──────────────────────────▼──────────────────────┐
               │              Azure Managed Services              │
               │                                                  │
               │  Cosmos DB  ──────────────── control-plane meta  │
               │  Blob Storage ─────────────── documents, Parquet │
               │  Storage Queue ────────────── indexing dispatch  │
               │  Azure AI Search ──────────── vector index       │
               │  Key Vault ────────────────── secrets at runtime │
               │  Log Analytics / Monitor ───── observability      │
               └──────────────────────────────────────────────────┘
```

### Public Hostnames

| Hostname | Service | Routing |
|----------|---------|---------|
| `app.gtog.id.vn` | Frontend (Next.js) | Cloudflare → Tunnel → ACA |
| `api.gtog.id.vn` | API Backend (FastAPI) | Cloudflare → Tunnel → ACA |

### ACA Ingress Rules

| Container App | Ingress | Rationale |
|---------------|---------|-----------|
| Frontend | Internal only | No direct public origin |
| API Backend | Internal only | No direct public origin |
| Indexing Worker | None | Background processor |

### Cloudflare Edge Policies

- **WAF**: Managed rules for OWASP Top-10.
- **Rate Limit**: Applied to `api.gtog.id.vn` (complements in-process backend limiter).
- **Cache Bypass**: Forced for `/api/*` and all SSE routes to prevent stale responses.
- **Tunnel**: Two `cloudflared` connector replicas for redundancy.

### Container Images

| Image | Registry | Role env var |
|-------|----------|-------------|
| `acrgtogshared.azurecr.io/gtog/backend:*` | ACR | `APP_ROLE=api` or `APP_ROLE=worker` |
| `acrgtogshared.azurecr.io/gtog/frontend:*` | ACR | N/A |

### Cosmos DB Schema

| Container | Content |
|-----------|---------|
| `collections` | Collection metadata |
| `documents` | Document records per collection |
| `indexingJobs` | Job lifecycle records |
| `jobEvents` | Job event log |
| `artifactManifest` | Index artifact version tracking |
| `entities` | Extracted entities (query-time context source) |
| `relationships` | Extracted relationships |
| `textUnits` | Document chunks |
| `communities` | Leiden community records |
| `communityReports` | LLM community summaries |
| `covariates` | Entity claims |
| `conversationSessions` | Multi-turn session metadata |
| `conversationTurns` | Individual Q&A turns |

### Observability

- Structured JSON logs emitted by all three containers to **Log Analytics**.
- `X-Request-Id` header propagated through all requests.
- `cf-ray` header preserved in request logs for Cloudflare correlation.
- Backend readiness probe (`/health/readiness`) checks all four Azure dependencies (Cosmos, Blob, Queue, KV).

---

## 11. Technology Stack

### Core Language & Runtime

| Layer | Language | Runtime |
|-------|----------|---------|
| Frontend | TypeScript 5 | Node.js 20 |
| Backend / Core | Python 3.11 | CPython |

### Frontend Stack

| Concern | Library |
|---------|---------|
| Framework | Next.js 16 (App Router) |
| UI library | React 19 |
| Styling | Tailwind CSS 4 |
| Data fetching | TanStack Query v5 + Axios |
| Icons | Lucide React |
| Build | npm / Next.js build pipeline |

### Backend Stack

| Concern | Library |
|---------|---------|
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Config | pydantic-settings v2 |
| Async queue | Azure SDK (azure-storage-queue) |
| Database client | Azure SDK (azure-cosmos) |
| Blob client | Azure SDK (azure-storage-blob) |
| Secret management | Azure SDK (azure-keyvault-secrets) |
| Vector search | Azure SDK (azure-cosmos) |
| SSE streaming | sse-starlette |
| HTTP client | httpx |

### GraphRAG Core Stack

| Concern | Library |
|---------|---------|
| LLM orchestration | fnllm (Microsoft), litellm |
| Embeddings | Supports OpenAI, Azure OpenAI, Gemini, HuggingFace |
| Default models | `gemini/gemini-2.5-flash-lite` (chat), `gemini/gemini-embedding-001` (embed) |
| Data processing | pandas, pyarrow |
| Graph algorithms | graspologic (Leiden community detection) |
| Vector math | numpy |
| Tokenization | tiktoken |
| Serialization | Parquet (via pyarrow) |

### Vector Store Adapters

| Adapter | Use Case |
|---------|---------|
| Azure Cosmos DB (NoSQL Vector) | Production vector index |
| Azure Cosmos DB (DiskANN) | Alternative cloud vector store |
| LanceDB | Local development |

### Infrastructure

| Concern | Technology |
|---------|-----------|
| Cloud | Microsoft Azure |
| Container runtime | Azure Container Apps |
| Ingress / CDN | Cloudflare (Tunnel, WAF, DNS) |
| Container registry | Azure Container Registry |
| Package manager (Python) | uv |
| Task runner | poethepoet (poe) |
| Linting / formatting | ruff |
| Type checking | pyright |
| Testing | pytest |

---

## 12. Configuration Model

The system is configured at two levels:

### 1. GraphRAG `settings.yaml` (per project)

Controls the knowledge graph pipeline:
- LLM model references and parameters
- Embedding model references
- Storage backends (input, output, cache)
- Indexing workflow options (chunk size, entity types, etc.)
- Per-method search parameters (community level, beam width, pruning strategy)

### 2. Backend `.env` / Environment Variables (per deployment)

Loaded by `pydantic-settings` from environment variables or `.env` file:

| Variable | Purpose |
|----------|---------|
| `AZURE_COSMOS_ENDPOINT` | Cosmos DB endpoint |
| `AZURE_STORAGE_ACCOUNT_NAME` | Blob storage |
| `AZURE_STORAGE_QUEUE_NAME` | Job queue name |
| `AZURE_KEY_VAULT_URL` | Runtime secrets |
| `AZURE_USE_MANAGED_IDENTITY` | MSI vs. key auth |
| `CORS_ORIGINS` | Allowed frontend origins |
| `REQUIRE_EDGE_AUTH` | Enforce X-Edge-Secret header |
| `EDGE_ORIGIN_SECRET` | HMAC secret for edge auth |
| `RATE_LIMIT_ENABLED` | Toggle rate limiting |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Default: 120 |
| `QUERY_CONTEXT_MODE` | `cosmos_only` or `blob_parquet` |
| `DEFAULT_CHAT_MODEL` | Default LLM for backend |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model |
| `ENABLE_TOG_DEBUG_ENDPOINT` | Expose `GET /search/tog/debug` |
| `CONVERSATION_TURN_TTL_DAYS` | Conversation retention (default: 30) |
| `APP_ROLE` | `api` or `worker` |

---

## 13. Security Model

### Ingress Protection

```
Internet → Cloudflare WAF (OWASP rules)
         → Cloudflare Rate Limiter
         → Cloudflare Tunnel (encrypted, no public ACA origin exposed)
         → ACA internal networking
         → FastAPI edge auth middleware (X-Edge-Secret HMAC or trusted tunnel IP)
         → FastAPI in-process rate limiter (per-IP, sliding window)
```

### Authentication Flow

- **`REQUIRE_EDGE_AUTH=true`** (production): Every API request must carry `X-Edge-Secret` matching the configured HMAC secret, **or** originate from a trusted Cloudflare Tunnel IP range (RFC 6598 / RFC 1918).
- **`REQUIRE_EDGE_AUTH=false`**: Restricted to localhost CORS origins only (development).

### Secret Management

Secrets are never stored in source code or container images. At startup, `bootstrap_runtime_secrets()` reads from Azure Key Vault (via MSI or client credentials) and populates:
- `AZURE_COSMOS_KEY`
- `OPENAI_API_KEY` (if used)
- `AZURE_STORAGE_CONNECTION_STRING`
- `GEMINI_API_KEY` (if used)

### CORS Policy

- `CORS_ORIGINS` must explicitly list allowed origins (default: localhost dev origins).
- In production: `["https://app.gtog.id.vn"]`.
- `Vary: Origin` header is correctly set to prevent cache poisoning.

### Rate Limiting

| Layer | Mechanism | Notes |
|-------|-----------|-------|
| Cloudflare | Edge rate limit on `api.gtog.id.vn` | Global enforcement |
| Backend | `InMemoryRateLimiter` (process-local) | Defence-in-depth; not distributed across replicas |
| Recommended | Set `RATE_LIMITER_BACKEND=redis` | For multi-replica distributed enforcement |

### Data Isolation

- Each collection's graph data is stored in separate Cosmos DB documents partitioned by `collection_id`.
- Blob artifacts are stored under `collections/{collection_id}/` prefix.
- The indexing worker processes one job at a time per replica; lease mechanism prevents duplicate runs.

---

*This document reflects the codebase as of 2026-03-22.*
