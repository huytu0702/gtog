# Context

Local ToG currently applies `num_retain_entity` while expanding entity candidates from all scored relation groups, and only applies `width` later when pruning the next frontier. That differs from the upstream ToG intent, where relation fanout is limited first, then entity retention happens inside those selected relations, and only then the next frontier is beam-pruned again. The goal of this change is to align local traversal behavior with upstream while keeping the existing config surface and minimizing code churn.

# Recommended approach

## Behavioral target

For each frontier node in `ToGSearch._process_node`:
1. score relations
2. sort relations by relevance descending
3. keep only the top `width` distinct relation groups
4. expand entity candidates only from those selected groups
5. filter missing entities
6. apply per-group `num_retain_entity` sampling as today
7. score entity candidates
8. create next-depth nodes
9. keep the existing global frontier prune by `width`

This preserves the current local meaning of:
- `width` as the next-frontier beam size
- `num_retain_entity` as a per-selected-relation-group retention cap

while adding the missing upstream-style relation fanout gate before entity expansion.

## Files to modify

- `F:/KL/gtog/graphrag/query/structured_search/tog_search/search.py`
- `F:/KL/gtog/tests/unit/tog/test_search_result.py`

## Existing code to reuse

- Relation scoring hook: `F:/KL/gtog/graphrag/query/structured_search/tog_search/search.py` (`_score_relations`)
- Current relation grouping key `(rel_desc, direction)`: `F:/KL/gtog/graphrag/query/structured_search/tog_search/search.py`
- Current per-group sampling logic using `ENTITY_CANDIDATE_SAMPLE_THRESHOLD` and `random.sample`: `F:/KL/gtog/graphrag/query/structured_search/tog_search/search.py`
- Existing global beam prune: `F:/KL/gtog/graphrag/query/structured_search/tog_search/state.py` (`prune_current_frontier`)

## Implementation steps

1. In `search.py`, change `_process_node` so that relation-group selection happens immediately after relation scoring and sorting.
2. Select the first `self.width` distinct relation groups in sorted order, using the existing group identity `(rel_desc, direction)`.
3. Discard lower-ranked relation groups before building `candidate_groups` and before any entity scoring.
4. Keep the rest of `_process_node` unchanged as much as possible:
   - entity lookup
   - filtering missing entities
   - per-group `num_retain_entity` sampling
   - entity scoring
   - combined score formula
   - node creation
5. Leave `ToGSearchState.prune_current_frontier()` unchanged so global beam pruning still happens after all selected-group candidates are scored.

## Test changes

Update `F:/KL/gtog/tests/unit/tog/test_search_result.py` to reflect width-first relation gating.

1. Rewrite the existing branch-selection test so it verifies that with `width=1`, only the top-ranked relation group is expanded, even if another lower-ranked group would have produced a higher entity score.
2. Update the `num_retain_entity` tests that currently rely on multiple relation groups being expanded despite `width=1`:
   - set `width` large enough to include all intended relation groups for those tests
   - keep the current assertions about per-group sampling behavior
3. Add one explicit regression test that proves candidates from relation groups beyond top-`width` never reach `score_entities`.
4. Keep the single-group missing-entity sampling test mostly unchanged.

## Edge cases to preserve

- Multiple edges belonging to the same `(rel_desc, direction)` group count as one selected relation group for width gating.
- Missing entity info is still filtered before sampling.
- If a selected relation group ends up with zero valid entities, the search should continue with fewer candidates rather than add fallback behavior.
- Stable order among tied relation scores can continue to follow current input order; tests should avoid depending on ties.

# Verification

1. Run targeted unit tests:
   - `pytest F:/KL/gtog/tests/unit/tog/test_search_result.py -q`
2. Verify specifically that:
   - `score_entities(...)` only receives candidates from top-`width` relation groups
   - `random.sample(...)` is only called for already-selected groups
   - the updated branch-selection/regression tests pass
3. If targeted tests pass, optionally run broader ToG unit coverage:
   - `pytest F:/KL/gtog/tests/unit/tog -q`

# Out of scope

- Do not change config names or defaults.
- Do not change pruning strategy interfaces.
- Do not copy upstream bugs/inconsistencies from the Wiki path; only align the intended traversal order.
