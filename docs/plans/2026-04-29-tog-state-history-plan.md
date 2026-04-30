# Implementation Plan: ToG State History Preservation

## Overview

Improve the local ToG GraphRAG implementation so each exploration branch preserves relation direction and previous relation history more faithfully, without fully porting upstream ToG’s `topic_entity` / `pre_relations` / `pre_heads` state model.

The implementation should extend the existing `ExplorationNode` tree state to carry direction-aware edge metadata and expose branch history to relation scoring, especially the LLM relation scoring prompt.

## Requirements Restatement

- Preserve the existing local architecture under `graphrag/query/structured_search/tog_search`.
- Do not rewrite the search loop around upstream-style `topic_entity`, `pre_relations`, or `pre_heads`.
- Extend `ExplorationNode` so each branch can represent:
  - relation text from parent,
  - relation direction from the current node’s expansion perspective,
  - relationship source/target IDs when available,
  - direction-aware previous relation history derived from the parent chain.
- Pass previous relation/path history into relation scoring.
- Update the LLM relation scoring prompt to use relation history when scoring next-hop relations.
- Keep non-LLM pruning strategies backward-compatible and minimally changed.
- Preserve current query behavior except where the new history context improves relation pruning.
- Add focused tests for state/history behavior, prompt construction, search metadata preservation, and direction-aware formatting.

## Current Known Gap

The local implementation already models exploration as a tree using `ExplorationNode`, but it loses state that upstream ToG uses when scoring subsequent relations.

Current `ExplorationNode` stores:

- `parent`
- `relation_from_parent`
- `relation_full_description`
- entity information and score

It does not store:

- relation direction from parent,
- relationship source/head ID,
- relationship target/tail ID,
- direction-aware relation history.

`LLMPruning.score_relations(...)` currently receives only:

- `query`
- `entity_name`
- `relations`
- optional `query_embedding`

So relation scoring has no direct awareness of prior selected relations or direction/head semantics along the branch.

Relevant files:

- `graphrag/query/structured_search/tog_search/state.py`
- `graphrag/query/structured_search/tog_search/exploration.py`
- `graphrag/query/structured_search/tog_search/search.py`
- `graphrag/query/structured_search/tog_search/pruning.py`
- `graphrag/query/structured_search/tog_search/reasoning.py`
- `graphrag/prompts/query/tog_relation_scoring_prompt.py`

## Design Direction

Instead of porting upstream ToG state literally, use the existing `ExplorationNode` parent chain as the branch state source of truth.

| Upstream ToG Concept | Local Equivalent |
|---|---|
| `topic_entity` | Initial/root `ExplorationNode` frontier |
| `pre_relations` | Derived from `ExplorationNode` parent chain |
| `pre_heads` | Derived from relation direction/source/target metadata on each node |
| Current topic/entity being expanded | Current `ExplorationNode` |
| Branch-specific state | Parent-linked `ExplorationNode` chain |

Why this approach:

- avoids rewriting the search algorithm,
- keeps compatibility with current beam search and reasoning code,
- makes state branch-local by construction,
- allows relation scoring to become history-aware while preserving current APIs as much as possible.

## Phase 1: State Model Extension

### 1. Extend `ExplorationNode`

File: `graphrag/query/structured_search/tog_search/state.py`

Add optional metadata fields:

```python
relation_direction_from_parent: str | None = None
relation_source_id: str | None = None
relation_target_id: str | None = None
```

Keep all new fields defaulted to `None` so existing tests and direct construction remain compatible.

### 2. Add relation history helpers

File: `graphrag/query/structured_search/tog_search/state.py`

Add helper methods:

- `get_relation_history()` — returns ordered relation records from root to current node.
- `get_relation_history_text()` — returns compact prompt-ready text.

Root nodes should return an empty history or a clear `"None"` string for prompt use.

Suggested relation history tuple shape:

```python
tuple[str, str, str, str]
```

Representing:

- parent entity name,
- relation text,
- child entity name,
- direction.

## Phase 2: Search Expansion Metadata Preservation

### 3. Populate edge metadata when creating child nodes

File: `graphrag/query/structured_search/tog_search/search.py`

In `_process_node(...)`, preserve `direction` from candidate relation data:

```python
(rel_desc, target_id, direction, weight, rel_score, ...)
```

When creating a child `ExplorationNode`, pass:

```python
relation_direction_from_parent=direction
relation_source_id=node.entity_id
relation_target_id=target_id
```

The `target_id` remains the next entity to visit, regardless of whether the original edge is incoming or outgoing.

### 4. Pass branch history into relation scoring

File: `graphrag/query/structured_search/tog_search/search.py`

Before calling `score_relations(...)`, build:

```python
current_path = self._node_to_path_string(node)
relation_history = node.get_relation_history_text()
```

Pass both into the pruning strategy call.

## Phase 3: Pruning Strategy Interface Update

### 5. Update base pruning interface

File: `graphrag/query/structured_search/tog_search/pruning.py`

Update `PruningStrategy.score_relations(...)` to accept optional context:

```python
relation_history: str | None = None
current_path: str | None = None
```

### 6. Update `LLMPruning.score_relations(...)`

File: `graphrag/query/structured_search/tog_search/pruning.py`

Accept the new optional parameters and normalize missing values:

```python
relation_history = relation_history or "None"
current_path = current_path or entity_name
```

Include both values in prompt formatting while preserving current score parsing and metrics behavior.

### 7. Update non-LLM pruning strategies

File: `graphrag/query/structured_search/tog_search/pruning.py`

Add the same optional parameters to:

- `SemanticPruning.score_relations(...)`
- `BM25Pruning.score_relations(...)`
- any other concrete relation scorer.

They can ignore the new parameters initially to avoid changing non-LLM behavior.

## Phase 4: Prompt Update

### 8. Add history context to relation scoring prompt

File: `graphrag/prompts/query/tog_relation_scoring_prompt.py`

Add sections similar to:

```text
Current reasoning path:
{current_path}

Previous relations followed:
{relation_history}
```

Update instructions so the model scores candidate relations as next-hop options given this path, and asks it to avoid redundant backtracking unless useful.

Preserve the output contract:

```text
Output ONLY a list of numbers in brackets, e.g., [8, 3, 6, 4]
```

### 9. Preserve custom prompt compatibility

File: `graphrag/query/structured_search/tog_search/pruning.py`

Supplying extra keyword arguments to `.format(...)` is safe if the template does not use them, so older custom prompts that only reference `{query}`, `{entity_name}`, and `{relations}` should continue to work.

## Phase 5: Direction-Aware Path Formatting

### 10. Update search path formatter

File: `graphrag/query/structured_search/tog_search/search.py`

Update `_node_to_path_string(...)` to render incoming edges differently from outgoing edges.

Recommended formatting:

- outgoing: `parent --[relation]--> current`
- incoming: `parent <--[relation]-- current`

Keep fallback behavior when direction metadata is missing.

### 11. Update reasoning path formatting

File: `graphrag/query/structured_search/tog_search/reasoning.py`

Update path rendering in:

- `_path_to_string(...)`
- `_extract_triplets(...)` if downstream reasoning needs direction-correct triplets,
- `_format_paths(...)` relationship display if it currently assumes parent-to-child direction.

Keep legacy fallback for nodes without direction metadata.

## Phase 6: Tests

### 12. Add state/history unit tests

Suggested file: `tests/unit/query/structured_search/tog_search/test_tog_state_history.py`

Cover:

- root node returns empty history,
- outgoing child history includes parent, relation, child, and direction,
- incoming child history renders direction correctly,
- multi-hop history preserves root-to-leaf order.

### 13. Add relation scoring prompt tests

Suggested file: `tests/unit/query/structured_search/tog_search/test_tog_pruning_history.py`

Use a fake `ChatModel` to capture the prompt. Assert the prompt includes:

- query,
- entity name,
- current path,
- previous relation history,
- candidate relations.

Also assert parsed scores still map to relation tuples.

### 14. Add search expansion metadata tests

Suggested file: `tests/unit/query/structured_search/tog_search/test_tog_search_history.py`

Extend existing tests if appropriate. Verify `_process_node(...)` creates child nodes with:

- `relation_direction_from_parent`,
- `relation_source_id`,
- `relation_target_id`,
- correct parent reference.

Also verify `score_relations(...)` receives `relation_history` on non-root expansions.

### 15. Add reasoning/path formatting tests

Suggested file: `tests/unit/query/structured_search/tog_search/test_tog_reasoning_history.py`

Cover:

- outgoing path formatting,
- incoming path formatting,
- fallback formatting when direction metadata is missing.

## Validation Commands

Targeted ToG tests:

```bash
pytest F:/KL/gtog/tests/unit/query/structured_search/tog_search
```

API-level ToG history tests:

```bash
pytest F:/KL/gtog/tests/unit/api/test_tog_api_history.py
```

Broader unit tests:

```bash
pytest F:/KL/gtog/tests/unit
```

Static checks for changed areas:

```bash
ruff format F:/KL/gtog/graphrag/query/structured_search/tog_search F:/KL/gtog/graphrag/prompts/query F:/KL/gtog/tests/unit/query/structured_search/tog_search
ruff check F:/KL/gtog/graphrag/query/structured_search/tog_search F:/KL/gtog/graphrag/prompts/query F:/KL/gtog/tests/unit/query/structured_search/tog_search --fix
pyright
```

## Edge Cases

- Root node has no relation history.
- Custom relation scoring prompt omits new placeholders.
- Relation direction is missing because a node was created by older test setup.
- Relationship lookup fails in `get_full_relation_info(...)`; child node should still preserve relation direction from `get_relations(...)`.
- Incoming edge traversal should not be displayed as if the original graph edge were outgoing from the parent.
- Relation history grows with depth; keep it compact.
- `SemanticPruning` and `BM25Pruning` should not fail due to unused new parameters.
- Existing tests that instantiate `ExplorationNode` directly should continue working due to default `None` fields.

## Risks and Mitigations

### Risk: prompt formatting breaks runtime

Mitigation:

- Supply all expected placeholders: `query`, `entity_name`, `relations`, `current_path`, `relation_history`.
- Add unit tests around prompt construction.

### Risk: exact-string tests fail due to direction-aware formatting

Mitigation:

- Preserve fallback formatting when direction is missing.
- Update assertions only where the previous output was semantically wrong.

### Risk: non-LLM pruning strategies break due to signature mismatch

Mitigation:

- Update every concrete `score_relations(...)` implementation in `pruning.py`.
- Run targeted ToG tests and `pyright`.

### Risk: history context increases prompt tokens

Mitigation:

- Keep history concise.
- Use relation/entity names only, not full descriptions.
- Rely on bounded ToG depth.

### Risk: direction semantics are ambiguous

Mitigation:

- Treat `relation_direction_from_parent` as traversal direction from the current node’s expansion perspective.
- Preserve source/target IDs separately.
- Avoid claiming full upstream `pre_heads` parity.

### Risk: overengineering into an upstream rewrite

Mitigation:

- Keep `ExplorationNode` parent chain as source of truth.
- Do not introduce `topic_entity`, `pre_relations`, or `pre_heads` as primary control structures.
- Add only minimal metadata and helper methods.

## Acceptance Criteria

- [ ] `ExplorationNode` stores relation direction/source/target metadata with backward-compatible defaults.
- [ ] `ExplorationNode` can derive previous relation history from root to current node.
- [ ] Search expansion populates relation direction metadata for child nodes.
- [ ] Relation scoring receives current path and previous relation history.
- [ ] LLM relation scoring prompt includes relation history and current path.
- [ ] Semantic and BM25 pruning strategies remain compatible with the updated interface.
- [ ] Direction-aware paths render incoming edges differently from outgoing edges.
- [ ] Root-node relation scoring works with empty history.
- [ ] Existing direct `ExplorationNode` construction does not break.
- [ ] Targeted ToG unit tests pass.
- [ ] API-level ToG history tests pass.
- [ ] Static checks pass for changed Python files.

## Implementation Order Summary

1. Update `graphrag/query/structured_search/tog_search/state.py`.
2. Add state/history unit tests.
3. Update pruning strategy signatures in `graphrag/query/structured_search/tog_search/pruning.py`.
4. Update relation scoring prompt in `graphrag/prompts/query/tog_relation_scoring_prompt.py`.
5. Add LLM prompt construction tests.
6. Update search expansion and relation scoring calls in `graphrag/query/structured_search/tog_search/search.py`.
7. Add search metadata tests.
8. Update direction-aware formatting in `search.py` and `reasoning.py`.
9. Update reasoning/path tests.
10. Run targeted tests, broader unit tests, and static checks.
