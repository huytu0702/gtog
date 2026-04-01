# GraphRAG + ToG (Think-on-Graph)

> A Microsoft Research GraphRAG implementation enhanced with **Think-on-Graph (ToG)** deep reasoning — deployed on Azure Container Apps with Cloudflare edge protection.

👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br/>
👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)<br/>
👉 [ToG Arxiv — ICLR 2024](https://arxiv.org/abs/2307.07697)

<div align="left">
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/graphrag">
  </a>
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/graphrag">
  </a>
</div>

---

## Overview

**GraphRAG + ToG** transforms unstructured documents into structured knowledge graphs and enables deep reasoning over those graphs using LLMs. This fork extends Microsoft GraphRAG with:

- **ToG (Think-on-Graph)** — iterative beam-search exploration + LLM-guided pruning over the knowledge graph (ICLR 2024)
- **Full-stack web application** — Next.js frontend + FastAPI backend for managing document collections and running queries
- **Production-grade cloud deployment** — private Azure Container Apps environment behind Cloudflare edge

---

## Cloud Architecture

![Cloud Architecture](docs\img\cloud_architecture.png) 

| Component | Technology |
|-----------|-----------|
| **Edge** | Cloudflare (Proxied DNS, WAF, Rate Limiting, Tunnel) |
| **Compute** | Azure Container Apps (private environment, internal ingress) |
| **Ingress agent** | `cloudflared` connector (no public inbound ports) |
| **Frontend** | Next.js 16 / React 19 / Tailwind CSS 4 |
| **Backend API** | Python / FastAPI / Uvicorn |
| **Worker** | Python async worker (long-running index / eval jobs) |
| **Queue** | Azure Storage Queue (job dispatch) |
| **Database** | Azure Cosmos DB |
| **Object storage** | Azure Blob Storage |
| **Secrets** | Azure Key Vault |
| **Observability** | Log Analytics + Azure Monitor |

### Public Hostnames

| Service | URL |
|---------|-----|
| Frontend | `app.gtog.id.vn` |
| Backend API | `api.gtog.id.vn` |

---

## Query Methods

| Method | Description |
|--------|-------------|
| **Global Search** | Map-reduce over community reports — best for broad thematic questions |
| **Local Search** | Entity-centric retrieval with direct evidence — best for specific entities |
| **DRIFT Search** | Multi-hop reasoning with dynamic context expansion |
| **Basic Search** | Simple vector similarity search |
| **ToG Search** | Iterative graph beam-search + LLM pruning + chain-of-thought reasoning |

---

## Quick Start

### Prerequisites

- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- An LLM API key (OpenAI, Azure OpenAI, or compatible)

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Initialize a GraphRAG project

```bash
graphrag init --root ./my-project
# Edit my-project/settings.yaml with your LLM keys
```

### 3. Index your documents

```bash
# Place .txt files in ./my-project/input/
graphrag index --root ./my-project
```

### 4. Query the knowledge graph

```bash
# Global overview question
graphrag query --root ./my-project --method global "What are the main themes?"

# Specific entity question
graphrag query --root ./my-project --method local "Tell me about character X"

# Deep reasoning with ToG
graphrag query --root ./my-project --method tog "How are X and Y connected?"
```

---

## Local Development

### Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Visit http://localhost:3000
```

### Docker Compose (both services)

```bash
docker compose -f docker-compose.dev.yml up
```

---

## Project Structure

```
graphrag/                  # Core GraphRAG + ToG library
├── cli/                   # CLI commands (index, query, eval, prompt-tune)
├── index/                 # Indexing pipeline (extract → embed → graph → reports)
├── query/                 # Search strategies
│   └── llm/tog/           # ToG implementation (beam search, pruning, reasoning)
├── prompts/query/         # LLM prompt templates
├── config/                # YAML-based configuration system
├── vector_stores/         # LanceDB, Azure AI Search adapters
└── model/                 # Knowledge graph data models

backend/                   # FastAPI REST API
├── app/
│   ├── main.py            # Application entry point
│   ├── routers/           # API route handlers
│   ├── services/          # Business logic & GraphRAG orchestration
│   └── worker.py          # Background job worker

frontend/                  # Next.js web application
├── app/
│   ├── page.tsx           # Main page
│   └── components/        # UI components (chat, document manager)
└── lib/                   # API clients, shared utilities

scripts/                   # Azure provisioning & hardening scripts
docs/                      # Full documentation (MkDocs)
eval/                      # Evaluation datasets and configs
```

---

## Evaluation

Run batch evaluation over a QA dataset:

```bash
# Full evaluation with LLM scoring
graphrag eval --root ./my-project

# Skip LLM evaluation — save raw search results only
graphrag eval --root ./my-project --skip-evaluation
# or
graphrag eval --root ./my-project -s
```

---

## Code Quality

```bash
ruff format .          # Format
ruff check . --fix     # Lint + auto-fix
pyright                # Type checking
uv run poe check       # All static checks
pytest ./tests/unit    # Unit tests
uv run poe test        # Full test suite with coverage
```

---

## Documentation

| Topic | Link |
|-------|------|
| Getting started | [docs/get_started.md](docs/get_started.md) |
| CLI reference | [docs/cli.md](docs/cli.md) |
| Configuration (YAML) | [docs/config/yaml.md](docs/config/yaml.md) |
| Indexing pipeline | [docs/index/overview.md](docs/index/overview.md) |
| Query engine | [docs/query/overview.md](docs/query/overview.md) |
| ToG search | [docs/query/tog_search.md](docs/query/tog_search.md) |
| ToG usage guide | [docs/tunh/tog_search_guide.md](docs/tunh/tog_search_guide.md) |
| Prompt tuning | [docs/config/models.md](docs/config/models.md) |

---

## Important Notes

- ⚠️ **GraphRAG indexing can be expensive** — start with small datasets and read the docs before scaling.
- Always run `graphrag init --root [path] --force` between minor version bumps.
- ToG search gives the most transparent reasoning but uses significantly more LLM calls than other methods.
- Prompt tuning (`graphrag prompt-tune`) is strongly recommended before production use.

---

## Responsible AI

See [RAI_TRANSPARENCY.md](./RAI_TRANSPARENCY.md) — What is GraphRAG, intended uses, limitations, and operational guidance.

---

## Trademarks

This project contains code derived from Microsoft GraphRAG. Use of Microsoft trademarks or logos in modified versions must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general) and must not imply Microsoft sponsorship.
