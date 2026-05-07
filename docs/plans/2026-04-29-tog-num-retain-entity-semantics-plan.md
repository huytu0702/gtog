# Plan: Align ToG `num_retain_entity` Semantics with Upstream

## Context

Local ToG currently applies `num_retain_entity` after relation scoring by sampling from the full `candidate_data` list aggregated for a node:

```python
if len(candidate_data) >= 10 and len(candidate_data) > self.num_retain_entity:
    candidate_data = random.sample(candidate_data, self.num_retain_entity)
```

This differs from upstream `GasolSun36/ToG` Freebase behavior, where `num_retain_entity` is applied to the candidate entity list for each selected relation before entity scoring:

```python
if len(entity_candidates_id) >= 20:
    entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)
```

Beam pruning remains a separate score-based step using `width`.

## Requirements

- Make local `num_retain_entity` match upstream semantics more closely.
- Apply candidate sampling per relation group, not across the entire node candidate pool.
- Keep sampling before entity scoring.
- Keep sampling random, not top-N by score.
- Keep beam pruning by `width` unchanged.
- Preserve existing metadata/history behavior in exploration nodes.
- Add tests that lock in the new semantics.

## Current Behavior

For one current node, local code:

1. Scores relations.
2. Builds one aggregated `candidate_data` list across all scored relations.
3. If aggregate candidate count is at least 10 and greater than `num_retain_entity`, randomly samples the aggregate list down to `num_retain_entity`.
4. Sends the sampled candidates to entity scoring.
5. Prunes the next frontier by `beam_width`.

Problem: if many relations each produce a small number of candidates, local code can still randomly drop most candidates before scoring, even though upstream would not sample any individual relation's candidate list.

## Desired Behavior

For one current node:

1. Score relations.
2. Build candidates grouped by relation.
3. For each relation group independently:
   - if that relation group has at least the sampling threshold and more than `num_retain_entity`, randomly sample that group down to `num_retain_entity`.
   - otherwise keep the group intact.
4. Flatten sampled groups back into one candidate list, preserving relation-score order as much as possible.
5. Send flattened candidates to entity scoring.
6. Prune the next frontier by `beam_width` exactly as today.

## Files to Change

### Required

- `graphrag/query/structured_search/tog_search/search.py`
  - Change `_process_node` candidate sampling from aggregate-node-level to per-relation-group.

### Optional

- `graphrag/config/models/tog_search_config.py`
  - Add a configurable sampling threshold if we do not want to hardcode upstream Freebase threshold `20`.

- `graphrag/query/factory.py`
  - Wire threshold config into `ToGSearch` if a new config field is added.

### Tests

- `tests/unit/query/structured_search/tog_search/test_tog_search_history.py`
  - Add or update `_process_node` tests for sampling semantics.

- Search existing ToG tests for old aggregate-sampling expectations and update them if present.

## Design Decisions

### Relation Group Key

Use a relation-group key scoped to the current node. Recommended key:

```python
(rel_desc, direction)
```

Rationale:

- `_process_node` already runs for a single current node, so source node identity is implicit.
- `rel_desc` identifies the relation label/description.
- `direction` preserves incoming vs outgoing semantics.

If duplicate relation descriptions point to different underlying relationship IDs and need separation, consider extending the group key with relation metadata from `get_full_relation_info`.

### Sampling Threshold

Recommended first implementation: use an internal constant matching upstream Freebase:

```python
ENTITY_CANDIDATE_SAMPLE_THRESHOLD = 20
```

Reason:

- Upstream Freebase uses `>=20`.
- Current local `>=10` is more aggressive and can reduce recall.
- Avoid adding config until there is a demonstrated need.

Alternative: expose a config field such as `entity_candidate_sample_threshold` with default `20`.

## Implementation Steps

### Phase 1: Update Candidate Grouping

In `_process_node`:

1. Keep relation scoring as-is.
2. Replace direct append to a single `candidate_data` list with a grouped structure:

```python
candidate_groups: dict[tuple[str, str], list[CandidateData]] = {}
relation_order: list[tuple[str, str]] = []
```

3. While iterating `scored_relations`, append each candidate to its relation group.
4. Preserve group order based on sorted `scored_relations` order.

### Phase 2: Apply Per-Relation Sampling

For each relation group:

1. If `len(group) >= ENTITY_CANDIDATE_SAMPLE_THRESHOLD` and `len(group) > self.num_retain_entity`, sample that group:

```python
sampled_group = random.sample(group, self.num_retain_entity)
```

2. Otherwise keep the full group.
3. Log sampling per relation group, e.g.:

```text
[ToG][node=...][depth=...] sample_relation_candidates <relation> 25->5
```

4. Flatten sampled groups into `candidate_data`.

### Phase 3: Preserve Existing Scoring and Node Creation

Keep these steps unchanged:

- Build `entity_candidates` from `candidate_data`.
- Call `self.pruning_strategy.score_entities(...)`.
- Create `ExplorationNode` objects with relation metadata.
- Let `ToGSearchState.prune_current_frontier()` prune by `beam_width`.

### Phase 4: Add Tests

Add tests covering:

#### Case A: Aggregate Pool Large, Each Relation Below Threshold

Setup:

- 3 relation groups.
- Each group has 8 candidates.
- Total candidates = 24.
- `num_retain_entity = 5`.
- Threshold = 20.

Expected:

- No sampling occurs.
- `score_entities` receives all 24 candidates.

This is the key regression test against current local behavior.

#### Case B: One Relation Group Exceeds Threshold

Setup:

- Relation A has 25 candidates.
- Relation B has 4 candidates.
- `num_retain_entity = 5`.

Expected:

- Relation A is sampled to 5.
- Relation B remains 4.
- `score_entities` receives 9 candidates.

#### Case C: Multiple Relation Groups Exceed Threshold

Setup:

- Relation A has 25 candidates.
- Relation B has 30 candidates.
- Relation C has 3 candidates.
- `num_retain_entity = 5`.

Expected:

- Relation A sampled to 5.
- Relation B sampled to 5.
- Relation C remains 3.
- `score_entities` receives 13 candidates.

#### Case D: Beam Width Still Controls Frontier

Setup:

- More scored child nodes than `width`.

Expected:

- Candidate sampling controls pre-score candidate volume only.
- `state.prune_current_frontier()` still keeps only `beam_width` nodes after scoring.

### Phase 5: Validate with Real Flow

Run the forced-depth ToG query against `medical_graphrag_project` and inspect logs.

Expected log differences:

- No more aggregate line like:

```text
sample_candidates 23->3
```

- New per-relation logs should only appear when a relation group itself exceeds threshold.

If most GraphRAG relationships produce one candidate per relation, then `num_retain_entity` may not trigger in many real runs, which is consistent with upstream semantics.

## Validation Commands

Run targeted tests:

```bash
./.venv/Scripts/python -m pytest tests/unit/query/structured_search/tog_search -q
```

Run broader ToG unit tests:

```bash
./.venv/Scripts/python -m pytest tests/unit -k tog -q
```

Run formatting/checks:

```bash
./.venv/Scripts/python -m ruff format graphrag/query/structured_search/tog_search/search.py tests/unit/query/structured_search/tog_search/test_tog_search_history.py
./.venv/Scripts/python -m ruff check graphrag/query/structured_search/tog_search/search.py tests/unit/query/structured_search/tog_search/test_tog_search_history.py
```

Optional real-flow validation:

```bash
./.venv/Scripts/graphrag query --root ./medical_graphrag_project --method tog --query "How are UV exposure, immune suppression, diagnosis, recurrence, and treatment connected in basal cell skin cancer care?"
```

If using the debug force-depth path, run through API or config with `debug_force_max_depth=True` and inspect:

```text
medical_graphrag_project/logs/query.log
```

## Risks

### Relation Grouping May Not Perfectly Match Upstream

Upstream data model returns multiple candidate entity IDs for a selected relation. Local GraphRAG relationships may often be one edge to one target entity. Grouping by `(rel_desc, direction)` approximates upstream relation-level semantics but may not be identical.

Mitigation: keep grouping logic small, well-tested, and documented.

### Recall/Latency Tradeoff Changes

Moving from aggregate sampling to per-relation sampling can increase the number of candidates sent to entity scoring.

Mitigation:

- Use upstream threshold `20`.
- Keep beam pruning unchanged.
- Compare real query latency before/after on `medical_graphrag_project`.

### Randomness in Tests

Random sampling can make tests flaky.

Mitigation:

- Patch `random.sample` in tests to return deterministic slices.
- Assert counts and group membership, not exact random order unless patched.

## Success Criteria

- `_process_node` no longer samples the aggregate node-level candidate list.
- `num_retain_entity` applies independently per relation group.
- Sampling happens before entity scoring.
- Beam pruning by `width` remains unchanged.
- Unit tests prove aggregate-large/group-small candidates are not incorrectly sampled.
- Real-flow logs show exploration/pruning still works.
