# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Type

This is a **GraphRAG with ToG Enhancement** repository - a Microsoft Research project that combines knowledge graph-based retrieval-augmented generation with Think-on-Graph deep reasoning capabilities.

## Project Overview

This is GraphRAG (Graph-based Retrieval-Augmented Generation) from Microsoft Research, enhanced with **ToG (Think-on-Graph)** - a deep reasoning search algorithm. The project transforms unstructured text into structured knowledge graphs and enables sophisticated querying over those graphs.

## Code Principles (Tier 1: Always Apply)

**SRP** - One class = one responsibility. Split when a class does too much.

**DRY** - Extract duplicated code into functions/utils. Three identical blocks = refactor.

**YAGNI** - Only implement what's specified. No "just in case" code.

**KISS** - Simple beats clever. If it's hard to read, simplify it.

Use default model to implementation task.
Use haiku model to exploration task.

Don't include Co-author in commit message.

Use .venv_new for virtual environment.

## Common Development Tasks

### Core Commands

```bash
# Initialize GraphRAG project
graphrag init --root ./my-graphrag-project

# Index documents (build knowledge graph)
graphrag index --root ./my-graphrag-project --config settings.yaml

# Update existing index
graphrag update --root ./my-graphrag-project

# Query the knowledge graph
graphrag query --root ./my-graphrag-project --method global "your query"
graphrag query --root ./my-graphrag-project --method local "your query"
graphrag query --root ./my-graphrag-project --method tog "your query"

# Generate optimized prompts
graphrag prompt-tune --root ./my-graphrag-project --config settings.yaml

# Run tests using pytest directly
pytest ./tests/unit                    # Unit tests
pytest ./tests/integration             # Integration tests
pytest ./tests/smoke                   # Smoke tests
pytest ./tests/notebook                # Notebook tests

# Or use uv with poethepoet scripts
uv run poe test_unit                   # Unit tests
uv run poe test_integration            # Integration tests
uv run poe test_smoke                  # Smoke tests
uv run poe test                        # All tests with coverage

# Code formatting and linting
ruff format .                         # Format code
ruff check . --fix                    # Fix linting issues
pyright                              # Type checking
uv run poe format                     # Format using poe
uv run poe fix                        # Auto-fix issues
uv run poe check                      # Run all static checks

# Chạy với đánh giá đầy đủ (mặc định)
graphrag eval --root ./my-project
# Chạy không đánh giá - chỉ lưu kết quả thô
graphrag eval --root ./my-project --skip-evaluation
# hoặc
graphrag eval --root ./my-project -s
```

### Development Setup

```bash
# Install dependencies using uv
uv sync

# Run individual commands using uv
uv run poe index <...args>
uv run poe query <...args>
uv run poe prompt_tune <...args>
```

### Backend Development

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
```

## Architecture Overview

The project follows a multi-layered architecture:

### 1. Core GraphRAG Library (`graphrag/`)
- **CLI Module**: Command-line interface with init, index, update, prompt-tune, query commands
- **API Module**: Programmatic access to GraphRAG functionality
- **Indexing Pipeline**: Processes documents to build knowledge graphs
- **Query Engine**: Five search strategies including ToG
- **Configuration System**: YAML-based settings management

### 2. ToG (Think-on-Graph) Implementation
The key enhancement - implements academic research (ICLR 2024) for deep reasoning:

- **Exploration Phase**: Iteratively expands through knowledge graph using beam search
- **Pruning Strategies**: LLM-based or semantic similarity-based filtering
- **Reasoning Phase**: Chain-of-thought reasoning over explored paths
- **Configurable Parameters**: Width, depth, pruning strategy, temperature settings

### 3. FastAPI Backend (`backend/`)
- REST API wrapper around GraphRAG
- Multiple document collections support
- Storage management and health checks

### 4. Next.js Frontend (`frontend/`)
- Neo-brutalist UI design
- React components for interaction
- Search interface with method selection

## Query Methods

The system supports five different search strategies:

1. **Global Search**: Map-reduce over community reports for broad overviews
2. **Local Search**: Entity-centric retrieval with direct evidence
3. **DRIFT Search**: Multi-hop reasoning with dynamic context
4. **Basic Search**: Simple vector similarity search
5. **ToG Search**: Iterative graph exploration with beam search and deep reasoning

## Key Components

### Indexing Pipeline
1. **Document Loading & Chunking**: Splits documents into manageable pieces
2. **Entity Extraction**: Identifies organizations, people, locations, events
3. **Relationship Building**: Creates semantic relationships between entities
4. **Community Detection**: Groups entities using Leiden algorithm
5. **Embedding Generation**: Creates vector representations for semantic search
6. **Report Generation**: Summarizes each community

## Entry Points

- **CLI**: `graphrag` command (defined in `pyproject.toml`)
- **Backend API**: `backend/app/main.py`
- **Frontend**: `frontend/app/`

## Testing

The project uses pytest for testing with separate test suites:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Smoke tests for basic functionality
- Notebook tests for documentation examples

## Important Notes

- GraphRAG indexing can be expensive - start with small datasets
- Always run `graphrag init --force` between minor version updates
- Prompt tuning is strongly recommended for best results with your data
- ToG search provides the most transparent reasoning but uses more LLM calls
- Use `uv sync` to install dependencies after pulling changes
- The project uses semversioner for versioning - run `uv run semversioner add-change` when submitting PRs

## Testing Requirements

- Tests use `.env` file for configuration (see `pytest.ini`)
- Integration and smoke tests may require API keys in environment variables

## File Structure Highlights

```
graphrag/
├── api/           # Library API definitions
├── cache/         # Cache module with factory pattern
├── callbacks/     # Common callback functions
├── cli/           # Command-line interface
├── config/        # Configuration management
├── index/         # Indexing engine
│   └── run/       # Main entrypoint to build an index
├── logger/        # Logger module with factory pattern
├── model/         # Data model definitions for knowledge graph
├── prompt_tune/   # Prompt tuning module
├── prompts/       # All system prompts used by GraphRAG
│   └── query/     # Query-specific prompts (including ToG prompts)
├── query/         # Query engine
│   └── llm/tog/   # ToG (Think-on-Graph) implementation
├── storage/       # Storage module with factory pattern
├── utils/         # Helper functions
└── vector_stores/ # Vector store module with factory pattern

backend/
└── app/
    ├── api/       # API routes
    └── storage/   # Document collections storage

frontend/
└── app/
    ├── components/ # React components
    └── pages/     # UI pages
```