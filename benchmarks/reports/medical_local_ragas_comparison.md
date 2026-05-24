# Medical GraphRAG LLM vs NLP baseline comparison: local search + RAGAS

## Scope

- Corpus: `medical_graphrag_project/input/medical.txt`
- Runtime env: `medical_graphrag_project/.env` sourced at command runtime only; it was not copied into benchmark roots.
- Source settings: `medical_graphrag_project/settings.yaml`
- LLM graph root: `benchmarks/medical_llm_graph`, indexed with `--method standard`
- NLP graph root: `benchmarks/medical_nlp_graph`, indexed with `--method fast`
- Query datasets: `Datasets/Questions/medical_questions_eval_subset_200_chunk_01.json` and `Datasets/Questions/medical_questions_eval_subset_200_chunk_03.json`
- Query method: `local` only
- Query collection: `graphrag eval --skip-evaluation`
- Quality scoring: `graphrag.eval.ragas_runner`

## Index latency

Source: `output/stats.json`.

| Baseline | Index method | Total runtime | Total runtime (s) | Graph extraction workflow | Graph extraction | Community reports | Embeddings |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| LLM graph | `standard` | 4h 39m 06s | 16,745.60 | `extract_graph` | 9,298.74s | 6,283.12s | 1,161.90s |
| NLP graph | `fast` | 2h 02m 01s | 7,321.14 | `extract_graph_nlp` | 25.65s | 6,285.52s | 998.61s |

The NLP baseline indexed about **2.29x faster** end-to-end. The graph extraction phase itself was the largest difference: `extract_graph_nlp` took 25.65s versus 9,298.74s for LLM `extract_graph`.

## Index token usage

Source: `output/llm_usage.jsonl`. Estimated costs use: GPT-5.4 mini `$0.75 / 1M input tokens` and `$4.50 / 1M output tokens`; Gemini embedding `$0.15 / 1M input tokens`.

| Baseline | Workflow | Request type | Model | Calls | Prompt tokens | Output tokens | Total tokens | estimated_cost_usd |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLM graph | `extract_graph` | chat | `openai/gpt-5.4-mini` | 4,437 | 4,365,971 | 4,185,818 | 8,551,789 | $22.1107 |
| LLM graph | `create_community_reports` | chat | `openai/gpt-5.4-mini` | 1,326 | 4,636,554 | 2,892,469 | 7,529,023 | $16.4935 |
| LLM graph | `generate_text_embeddings` | embedding | `gemini/gemini-embedding-001` | 581 | 1,926,657 | 0 | 1,926,657 | $0.2890 |
| NLP graph | `create_community_reports_text` | chat | `openai/gpt-5.4-mini` | 607 | 4,447,865 | 2,189,964 | 6,637,829 | $13.1907 |
| NLP graph | `generate_text_embeddings` | embedding | `gemini/gemini-embedding-001` | 322 | 1,157,242 | 0 | 1,157,242 | $0.1736 |

Totals:

| Baseline | Total calls | Chat calls | Embedding calls | Prompt tokens | Output tokens | Total tokens | estimated_cost_usd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 6,344 | 5,763 | 581 | 10,929,182 | 7,078,287 | 18,007,469 | $38.8932 |
| NLP graph | 929 | 607 | 322 | 5,605,107 | 2,189,964 | 7,795,071 | $13.3643 |

The NLP baseline used about **56.7% fewer index tokens** overall, avoided LLM chat calls for graph extraction, and reduced estimated index cost by about **65.6%**.

## Local query efficiency

Source: `eval_results/local_chunk_01/eval_results_simple.json`.

| Baseline | Rows | Method | Errors | Avg latency (s) | LLM calls | Input tokens | Output tokens | estimated_cost_usd |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 50 | `local` | 0 | 9.09 | 50 | 509,289 | 7,121 | $0.4140 |
| NLP graph | 50 | `local` | 0 | 10.52 | 50 | 546,057 | 5,372 | $0.4337 |

The LLM graph baseline was about **13.5% faster** at local query time on this chunk, while the NLP graph baseline used slightly more input tokens, fewer output tokens, and about **4.8%** higher estimated query cost.

## RAGAS quality summary

Source: LLM uses `eval_results/local_chunk_01/eval_results_ragas_summary.json`

| Baseline | Count | Success | Fail | Context precision | Context recall | Faithfulness | Answer relevancy | Custom answer correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 50 | 50 | 0 | 0.940000 | 0.900000 | 0.831207 | 0.866779 | 0.900000 |
| NLP graph | 50 | 50 | 0 | 0.880000 | 0.880000 | 0.812512 | 0.841283 | 0.880000 |

On this 50-question chunk, the LLM graph baseline scored higher on context precision, context recall, faithfulness, answer relevancy, and custom answer correctness.

## Chunk 03 local query efficiency

Source: `eval_results/local_chunk_03/eval_results_simple.json`.

| Baseline | Rows | Method | Errors | Avg latency (s) | LLM calls | Input tokens | Output tokens | estimated_cost_usd |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 50 | `local` | 0 | 13.29 | 50 | 499,139 | 14,988 | $0.4418 |
| NLP graph | 50 | `local` | 0 | 18.53 | 50 | 547,074 | 12,465 | $0.4664 |

The LLM graph baseline was about **28.3% faster** at local query time on chunk 03 and about **5.3%** cheaper on estimated query cost.

## Chunk 03 RAGAS quality summary

Source: `eval_results/local_chunk_03/eval_results_ragas_summary.json`.

| Baseline | Count | Success | Fail | Context precision | Context recall | Faithfulness | Answer relevancy | Custom answer correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM graph | 50 | 50 | 0 | 0.960000 | 0.680000 | 0.823200 | 0.857265 | 0.680000 |
| NLP graph | 50 | 50 | 0 | 0.900000 | 0.400000 | 0.688012 | 0.870990 | 0.660000 |

On chunk 03, the LLM graph baseline scored higher on context precision, context recall, faithfulness, and custom answer correctness. The NLP graph baseline scored slightly higher on answer relevancy.

## Artifact locations

### LLM graph

- Index stats: `benchmarks/medical_llm_graph/output/stats.json`
- Index usage: `benchmarks/medical_llm_graph/output/llm_usage.jsonl`
- Simple local eval: `benchmarks/medical_llm_graph/eval_results/local_chunk_01/eval_results_simple.json`
- RAGAS detailed: `benchmarks/medical_llm_graph/eval_results/local_chunk_01/eval_results_ragas_detailed.json`
- RAGAS summary: `benchmarks/medical_llm_graph/eval_results/local_chunk_01/eval_results_ragas_summary.json`
- Chunk 03 simple local eval: `benchmarks/medical_llm_graph/eval_results/local_chunk_03/eval_results_simple.json`
- Chunk 03 RAGAS detailed: `benchmarks/medical_llm_graph/eval_results/local_chunk_03/eval_results_ragas_detailed.json`
- Chunk 03 RAGAS summary: `benchmarks/medical_llm_graph/eval_results/local_chunk_03/eval_results_ragas_summary.json`

### NLP graph

- Index stats: `benchmarks/medical_nlp_graph/output/stats.json`
- Index usage: `benchmarks/medical_nlp_graph/output/llm_usage.jsonl`
- Simple local eval: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/eval_results_simple.json`
- RAGAS detailed: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/eval_results_ragas_detailed.json`
- RAGAS summary: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/eval_results_ragas_summary.json`
- RAGAS failed-row retry input: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/ragas_retry_failed/eval_results_simple_failed_only.json`
- RAGAS failed-row retry detailed: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/ragas_retry_failed/eval_results_ragas_detailed.json`
- RAGAS failed-row retry summary: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/ragas_retry_failed/eval_results_ragas_summary.json`
- RAGAS consolidated detailed: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/eval_results_ragas_detailed_consolidated.json`
- RAGAS consolidated summary: `benchmarks/medical_nlp_graph/eval_results/local_chunk_01/eval_results_ragas_summary_consolidated.json`
- Chunk 03 simple local eval: `benchmarks/medical_nlp_graph/eval_results/local_chunk_03/eval_results_simple.json`
- Chunk 03 RAGAS detailed: `benchmarks/medical_nlp_graph/eval_results/local_chunk_03/eval_results_ragas_detailed.json`
- Chunk 03 RAGAS summary: `benchmarks/medical_nlp_graph/eval_results/local_chunk_03/eval_results_ragas_summary.json`

## Notes and caveats

- Query evals emitted repeated `Reached token limit - reverting to previous context state` warnings. Results were still produced for all 50 rows with no query errors.
- Some local search contexts emitted `No community records added when building community context` warnings.
- The provider returned 429 rate-limit errors during embedding calls, but retry logic recovered and both local eval outputs completed.
- A provider traceback printed a request URL containing an API key during an earlier verbose eval run. Treat that key as exposed: rotate/revoke it, audit usage, and redact/purge logs containing that traceback before sharing artifacts.
- RAGAS output was redirected to `ragas_run.log` for later runs to reduce console exposure, but those logs may still contain provider traces if retries occurred.