# ToG Conversation History Support — Design

**Date:** 2026-02-27
**Status:** Approved

## Background

GraphRAG natively supports `ConversationHistory` for local, global, basic, and drift searches. `ToGSearch` currently has no conversation history support — `search()` and `stream_search()` only accept `query: str`.

This design adds conversation history to ToG search following the same pattern as `LocalSearch`.

## Approach

Minimal surface change: add `conversation_history` parameter to `ToGSearch` methods and use it in two places — entity linking and final reasoning. No new classes, no changes to config, no restructuring of the class hierarchy.

## Data Flow

```
User calls tog_search(query, conversation_history)
  │
  ├─ 1. Entity linking
  │       effective_query = query + "\n" + past_user_questions
  │       → find_starting_entities_semantic(effective_query)
  │
  ├─ 2. Graph exploration + pruning  (unchanged)
  │       original query used for relation/entity scoring prompts
  │
  └─ 3. Reasoning / final answer
          history_context = conversation_history.build_context(...)
          prompt = history_context + exploration_paths
          → generate_answer(query, paths, history_context)
          → check_early_termination(query, nodes, history_context)
```

## Files Changed

| File | Change |
|---|---|
| `graphrag/query/structured_search/tog_search/search.py` | Add `conversation_history` param to `search()`, `stream_search()`, `_stream_search_with_metrics()` |
| `graphrag/query/structured_search/tog_search/reasoning.py` | Add `conversation_history_context: str` param to `generate_answer()` and `check_early_termination()` |
| `graphrag/api/query.py` | Add `conversation_history` param to `tog_search()` and `tog_search_streaming()` |

## API Signatures

```python
# tog_search/search.py
async def search(
    self,
    query: str,
    conversation_history: ConversationHistory | None = None,
) -> SearchResult: ...

async def stream_search(
    self,
    query: str,
    conversation_history: ConversationHistory | None = None,
) -> AsyncGenerator[str, None]: ...

# api/query.py
async def tog_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    query: str,
    conversation_history: ConversationHistory | None = None,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[...]: ...
```

## Key Decisions

- **Entity linking uses enriched query** — `query + "\n" + past_user_questions` (same as LocalSearch) so ambiguous follow-up questions ("who directed it?") resolve correctly
- **Pruning prompts use original query only** — appending past questions to relation/entity scoring prompts could confuse the LLM scorer
- **Reasoning prompt gets history prepended** — `conversation_history.build_context()` output is prepended to the exploration paths text passed to `generate_answer()`
- **`check_early_termination` also receives history context** — so early termination decisions are conversation-aware
- **`conversation_history_max_turns` = 5** — same default as LocalSearch; not exposed as config (per-call param)
- **No config changes** — `ToGSearchConfig` unchanged; history is a runtime parameter, not a configuration option
