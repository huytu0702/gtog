# GraphRAG Evaluation Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive evaluation framework to compare ToG, Local, and Basic search methods using LLM-as-Judge metrics.

**Architecture:** A modular evaluation framework consisting of 4 phases: (1) Modify ToGSearch to return SearchResult with metrics, (2) Create graphrag/eval module with metrics and runner, (3) Add CLI `eval` command, (4) Write tests. Each phase builds on the previous.

**Tech Stack:** Python 3.11+, pandas, pydantic, pytest, asyncio

---

## Overview

This plan is split into 4 separate phase files for focused implementation:

| Phase | File | Summary |
|-------|------|---------|
| 1 | `phase1-tog-searchresult.md` | Modify ToGSearch to return SearchResult with LLM call/token metrics |
| 2 | `phase2-eval-module.md` | Create `graphrag/eval/` package with metrics, runner, and results |
| 3 | `phase3-cli-integration.md` | Add `graphrag eval` CLI command with config support |
| 4 | `phase4-testing.md` | Unit and integration tests for the evaluation framework |

## Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3
                │
                └──► Phase 4
```

Phase 2 depends on Phase 1 (needs SearchResult return type).
Phases 3 and 4 both depend on Phase 2.
Phases 3 and 4 can be executed in parallel.

## Key Design Decisions

### 1. SearchResult Consistency
All search methods (Local, Global, DRIFT, Basic) already return `SearchResult` with metrics. ToG currently returns `str`. We'll modify ToG to match.

### 2. Metric Tracking
Track LLM calls and tokens at the source:
- `LLMPruning.score_relations()` - each call is one LLM invocation
- `ToGReasoning.generate_answer()` - final answer generation
- `ToGReasoning.check_early_termination()` - early termination check

### 3. LLM-as-Judge Implementation
Use the same LLM configured in settings.yaml with temperature=0.0 for consistent scoring.

### 4. Output Format
Match the design document's JSON structure exactly.

## Files to Create

```
graphrag/
├── eval/
│   ├── __init__.py
│   ├── metrics.py          # LLM-as-Judge implementations
│   ├── runner.py           # Evaluation orchestrator
│   └── results.py          # Result aggregation and output
├── cli/
│   └── eval.py             # CLI entry point (new file)
└── query/structured_search/tog_search/
    ├── search.py           # Modify to return SearchResult
    ├── pruning.py          # Add metric tracking
    └── reasoning.py        # Add metric tracking

tests/
├── unit/
│   └── eval/
│       ├── __init__.py
│       ├── test_metrics.py
│       ├── test_runner.py
│       └── test_results.py
└── integration/
    └── eval/
        ├── __init__.py
        └── test_eval_integration.py
```

## Files to Modify

- `graphrag/cli/main.py` - Add `eval` command registration
- `graphrag/api/__init__.py` - Export tog_search with SearchResult
- `graphrag/config/enums.py` - May need new enum for eval config

## Quick Reference

### Current ToG Return Type
```python
# graphrag/query/structured_search/tog_search/search.py:57
async def search(self, query: str) -> str:
```

### Target SearchResult Type
```python
# graphrag/query/structured_search/base.py:27-43
@dataclass
class SearchResult:
    response: str | dict[str, Any] | list[dict[str, Any]]
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    context_text: str | list[str] | dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    llm_calls_categories: dict[str, int] | None = None
    prompt_tokens_categories: dict[str, int] | None = None
    output_tokens_categories: dict[str, int] | None = None
```

### Existing CLI Pattern
```python
# graphrag/cli/main.py - commands use typer decorators
@app.command("query")
def _query_cli(...):
    from graphrag.cli.query import run_tog_search
    ...
```

## Implementation Order

Execute phases in this order:
1. **Phase 1**: ToG SearchResult (blocking - needed by Phase 2)
2. **Phase 2**: Eval module (blocking - needed by Phases 3 & 4)
3. **Phase 3 & 4**: CLI and Tests (can run in parallel)
