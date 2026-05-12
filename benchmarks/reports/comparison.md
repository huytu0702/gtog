# LLM vs NLP GraphRAG smoke benchmark

## Scope

- Input: short synthetic medical/veterinary story in `benchmarks/llm_graph/input/sample_medical_story.txt` and `benchmarks/nlp_graph/input/sample_medical_story.txt`.
- Questions: 2 QA pairs in `benchmarks/questions/benchmark_questions.json`.
- LLM graph root: `benchmarks/llm_graph`, indexed with `--method standard`.
- NLP graph root: `benchmarks/nlp_graph`, indexed with `--method fast`.
- Query methods: `local`, `basic`, `tog` with `--skip-evaluation`.
- Settings were copied from `medical_graphrag_project/settings.yaml`; smoke roots use `prune_graph.min_node_freq: 1` and `remove_ego_nodes: false` so the tiny NLP graph is not pruned to zero nodes.

## Index latency

| Graph mode | Index method | Total runtime (s) | Graph extraction workflow | Graph extraction (s) | Community report workflow | Community reports (s) | Embeddings (s) |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| LLM graph | standard | 99.35 | `extract_graph` | 60.12 | `create_community_reports` | 32.95 | 4.64 |
| NLP graph | fast | 24.35 | `extract_graph_nlp` | 5.03 | `create_community_reports_text` | 14.32 | 3.84 |

Smoke result: the NLP graph index completed about 4.08x faster than the LLM graph index on this input.

## Index token usage

Source: `benchmarks/*_graph/output/llm_usage.jsonl`.

| Graph mode | Workflow | Request type | Model | Calls | Prompt tokens | Output tokens | Total tokens | Cost |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| LLM graph | `extract_graph` | chat | `openai/gpt-5.2` | 2 | 6,344 | 3,196 | 9,540 | not configured |
| LLM graph | `create_community_reports` | chat | `openai/gpt-5.2` | 4 | 11,666 | 4,675 | 16,341 | not configured |
| LLM graph | `generate_text_embeddings` | embedding | `gemini/gemini-embedding-001` | 4 | 4,850 | 0 | 4,850 | not configured |
| NLP graph | `create_community_reports_text` | chat | `openai/gpt-5.2` | 1 | 2,216 | 707 | 2,923 | not configured |
| NLP graph | `generate_text_embeddings` | embedding | `gemini/gemini-embedding-001` | 3 | 691 | 0 | 691 | not configured |

Totals:

| Graph mode | Chat calls | Embedding calls | Prompt tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 6 | 4 | 22,860 | 7,871 | 30,731 |
| NLP graph | 1 | 3 | 2,907 | 707 | 3,614 |

Smoke result: the NLP graph path avoided LLM calls during graph extraction and used about 88.2% fewer indexed tokens overall. Dollar cost is not populated in this run because `GRAPHRAG_LLM_PRICING_JSON` was not configured with prices for these model keys.

## Query benchmark summary

Source: `benchmarks/reports/llm_graph_query/eval_results_simple.json` and `benchmarks/reports/nlp_graph_query/eval_results_simple.json`.

| Graph mode | Method | Avg latency (s) | LLM calls | Input tokens | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| LLM graph | basic | 16.39 | 2 | 1,476 | 71 |
| LLM graph | local | 5.94 | 2 | 8,660 | 207 |
| LLM graph | tog | 5.11 | 2 | 773 | 62 |
| NLP graph | basic | 4.71 | 2 | 1,476 | 122 |
| NLP graph | local | 9.41 | 2 | 8,108 | 91 |
| NLP graph | tog | 8.86 | 2 | 584 | 72 |

Per-question query metrics:

| Graph mode | Question | Method | Latency (s) | LLM calls | Input tokens | Output tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LLM graph | Who treats Luna at Maple Clinic and what treatment is used? | local | 4.08 | 1 | 4,364 | 98 |
| LLM graph | Who treats Luna at Maple Clinic and what treatment is used? | basic | 2.43 | 1 | 738 | 51 |
| LLM graph | Who treats Luna at Maple Clinic and what treatment is used? | tog | 5.07 | 1 | 400 | 34 |
| LLM graph | Which pharmacy supplies cetirizine to Maple Clinic? | local | 7.80 | 1 | 4,296 | 109 |
| LLM graph | Which pharmacy supplies cetirizine to Maple Clinic? | basic | 30.36 | 1 | 738 | 20 |
| LLM graph | Which pharmacy supplies cetirizine to Maple Clinic? | tog | 5.15 | 1 | 373 | 28 |
| NLP graph | Who treats Luna at Maple Clinic and what treatment is used? | local | 12.54 | 1 | 4,054 | 71 |
| NLP graph | Who treats Luna at Maple Clinic and what treatment is used? | basic | 4.70 | 1 | 738 | 102 |
| NLP graph | Who treats Luna at Maple Clinic and what treatment is used? | tog | 6.57 | 1 | 288 | 40 |
| NLP graph | Which pharmacy supplies cetirizine to Maple Clinic? | local | 6.29 | 1 | 4,054 | 20 |
| NLP graph | Which pharmacy supplies cetirizine to Maple Clinic? | basic | 4.72 | 1 | 738 | 20 |
| NLP graph | Which pharmacy supplies cetirizine to Maple Clinic? | tog | 11.14 | 1 | 296 | 32 |

## Query telemetry caveat

Provider telemetry files were also written:

- `benchmarks/reports/llm_graph_query_usage.jsonl`
- `benchmarks/reports/nlp_graph_query_usage.jsonl`

Each file contains 12 provider records: 6 chat calls and 6 embedding calls. Embedding token usage is present, but chat token fields are zero because the query chat response path did not expose provider `usage` fields. For query token accounting, use `eval_results_simple.json` efficiency metrics instead.

## Notes

- Index latency source of truth: `output/stats.json`.
- Index token/cost source of truth: `output/llm_usage.jsonl`.
- Query latency/token source of truth: `eval_results_simple.json` for this smoke run.
- Both index runs completed and wrote artifacts. The Windows console printed async SSL transport cleanup warnings after pipeline completion; these did not prevent output generation.
