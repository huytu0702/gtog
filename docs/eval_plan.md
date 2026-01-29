# QA Evaluator for GraphRAG (basic/local/tog)

## Goal
Build a standalone evaluator script under `graphrag/` that compares Basic vs Local vs ToG search across film Q/A sets, includes evidence/citation scoring, and produces a dashboard of metrics. Support separate index roots per `imdb_key`. Add a minimal ToG SearchResult path so efficiency metrics (LLM calls/tokens/latency) are available.

## Decisions (from user)
- Indexes are separate per film (`imdb_key`).
- ToG should be patched to return `SearchResult` with efficiency metrics.
- Metrics stack includes ROUGE + embedding similarity + LLM judge.
- LLM judge should use `gpt-5.1` (model id must exist in config; evaluator should error clearly if missing).

## Implementation Plan

### 1) Add evaluator script
Create `graphrag/eval/qa_eval.py` (standalone script; no backend/frontend changes).

Key behavior:
- Load `eval/qa_eval.json` (JSON array) and support JSONL as a fallback.
- Group questions by `imdb_key`.
- Map each `imdb_key` to a GraphRAG project root via CLI:
  - `--index-root tt0097576=PATH --index-root tt0102798=PATH ...` (repeatable), or
  - `--index-map path/to/map.json` (JSON: `{ "tt0097576": "...", ... }`).
- For each index root:
  - Load config (`graphrag/config/load_config.py`).
  - Load parquet tables using `create_storage_from_config` + `load_table_from_storage`:
    - `text_units`, `entities`, `relationships`, `communities`, `community_reports`, optional `covariates` (handle missing file).
  - Convert DataFrames to model objects via `graphrag/query/indexer_adapters.py`:
    - `read_indexer_text_units`, `read_indexer_entities`, `read_indexer_relationships`, `read_indexer_reports`, `read_indexer_covariates`.
  - Create vector stores via `graphrag/utils/api.get_embedding_store`:
    - Basic: `text_unit_text_embedding` (from `graphrag/config/embeddings.py`).
    - Local/ToG: `entity_description_embedding`.
  - Build search engines via `graphrag/query/factory.py`:
    - `get_basic_search_engine`, `get_local_search_engine`, `get_tog_search_engine(track_stats=True)`.
  - Run `search_engine.search(query=...)` (basic/local) and `search_engine.search_result(query=...)` (ToG) to obtain `SearchResult`.
- Store per-question results and write:
  - `eval/results/eval_results.csv` (per-question rows)
  - `eval/results/eval_report.json` (summary dashboard)
  - `eval/results/details/*.json` (raw responses/citations)

CLI options (proposed):
- `--index-root` (repeatable) or `--index-map` (json path)
- `--qa-eval` (default `eval/qa_eval.json`)
- `--methods` (`basic local tog`)
- `--output-dir`
- `--community-level` (local search)
- `--response-type`
- `--limit` or `--per-imdb-limit`
- `--judge-model-id gpt-5.1`
- `--embedding-model-id` (default to config default embedding model)

### 2) Patch ToG to return SearchResult + stats
Goal: keep existing behavior intact, but add a path that returns `SearchResult` with totals.

Changes:
- Add a lightweight LLM stats tracker and model wrapper:
  - New helper in `graphrag/query/structured_search/tog_search/llm_stats.py` with:
    - `LLMStats` dataclass: `llm_calls`, `prompt_tokens`, `output_tokens`.
    - `CountingChatModel` wrapper implementing `ChatModel` that delegates to the underlying model and counts tokens using `Tokenizer.encode()`:
      - Increment `llm_calls` on each `achat/achat_stream` call.
      - Add prompt tokens from `prompt` (and `history` if present).
      - Add output tokens from generated chunks/content.
- Update `get_tog_search_engine` in `graphrag/query/factory.py`:
  - Add optional `track_stats: bool = False`.
  - When `track_stats=True`, wrap `chat_model` with `CountingChatModel` before passing it to `LLMPruning` and `ToGReasoning`.
  - Attach the `LLMStats` instance to the returned `ToGSearch` (e.g., `search_engine.stats`).
- Update `graphrag/query/structured_search/tog_search/search.py`:
  - Add `async def search_result(...) -> SearchResult`:
    - Use existing `stream_search` to build the answer string.
    - Pull token/call counts from `self.stats` (if set), otherwise default to zeros.
    - Set `completion_time` from wall-clock.
    - Capture exploration-paths text by calling `self.reasoning_module._format_paths(...)` on the same `all_paths` used for final reasoning and store it on `self` (e.g., `self._last_paths_text`) for `search_result` to read.
    - Set `context_text` to `self._last_paths_text` (or empty if unavailable).
    - Set `context_data` to `{}` (ToG evidence will be validated using the relationships table in the evaluator).
  - Keep existing `search()` and `stream_search()` signatures intact to avoid breaking CLI usage.

### 3) Metrics implementation (in evaluator)

Answer quality:
- LLM judge (gpt-5.1): score 0–10 with brief rationale; judge compares generated answer to gold.
- ROUGE-L: implement a small LCS-based ROUGE-L (avoid external deps).
- Embedding similarity: use GraphRAG embedding model (`EmbeddingModel`) to embed gold + response and compute cosine similarity.

Groundedness:
- Basic/Local:
  - Parse inline citations: `[Data: Sources (1,2), Reports (3)]`.
  - Validate IDs against `context_data` DataFrames (check columns: `id`, `source_id`, `human_readable_id`).
  - Compute: citation_validity, citation_density (#citations / #sentences), support_score.
- ToG:
  - Parse `**Evidence**` / `**Reasoning**` sections per `TOG_REASONING_PROMPT`.
  - Extract relationship lines from Evidence/Reasoning:
    - `• SOURCE --[relation]--> TARGET` (from `_format_paths`), plus any triplet strings if present.
  - Verify against relationships table using `(source, target, description)` match (normalize case/whitespace).
  - Compute: triplet_validity, evidence_coverage.

Efficiency:
- Use SearchResult fields for `llm_calls`, `prompt_tokens`, `output_tokens`, `completion_time`.

Robustness:
- Empty/failed responses, no citations, too-short answers.

### 4) Output dashboard
Aggregate by:
- Method (basic/local/tog)
- Film (`imdb_key`)

Report fields:
- Answer quality means/stds
- Groundedness metrics
- Efficiency metrics
- Failure rates

### 5) Verification
- Dry-run on a small subset (e.g., `--limit 5`) for each film.
- Validate citation parsing on at least one Basic and one Local response.
- Validate ToG triplet extraction against the relationships table.
- Confirm outputs are written to `eval/results/`.

## Files to Add/Change
- Add: `graphrag/eval/qa_eval.py`
- Add: `graphrag/query/structured_search/tog_search/llm_stats.py` (or similar helper)
- Change: `graphrag/query/factory.py` (optional `track_stats` + wrapper)
- Change: `graphrag/query/structured_search/tog_search/search.py` (add `search_result` returning `SearchResult`)

## Notes / Constraints
- No backend/frontend changes.
- `gpt-5.1` must exist as a model ID in each index’s `settings.yaml` or via config override.
- Evaluator should not read `qa_eval.json` incrementally from CLI, but it should support both JSON array and JSONL formats.
