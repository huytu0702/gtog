# GraphRAG Evaluation Framework Design

**Date**: 2026-01-29
**Status**: Approved
**Goal**: Compare ToG, Local, and Basic search methods using movie QA dataset

## Overview

Build an evaluation framework to compare three GraphRAG search methods (ToG, Local, Basic) against a movie QA dataset with 359 question-answer pairs across 3 movies.

## Dataset

- **Source**: `eval/qa_eval.json`
- **Movies**:
  - tt0097576: 102 QA pairs
  - tt0102798: 51 QA pairs
  - tt2278388: 206 QA pairs
- **Transcripts**: `eval/tt0097576.txt`, `eval/tt0102798.txt`, `eval/tt2278388.txt`

## Folder Structure

Each movie has its own GraphRAG project folder:

```
F:/KL/gtog/
├── eval/
│   ├── qa_eval.json
│   ├── tt0097576.txt
│   ├── tt0102798.txt
│   └── tt2278388.txt
├── tt0097576/                    # graphrag init --root ./tt0097576
│   ├── settings.yaml
│   ├── input/
│   │   └── tt0097576.txt
│   └── output/
│       ├── entities.parquet
│       ├── relationships.parquet
│       └── ...
├── tt0102798/                    # graphrag init --root ./tt0102798
│   └── ...
└── tt2278388/                    # graphrag init --root ./tt2278388
    └── ...
```

## Metrics

### Accuracy Metrics
| Metric | Description | Method | Scale |
|--------|-------------|--------|-------|
| `correctness` | Does predicted answer match ground truth? | LLM-as-Judge | 0-1 |

### Reasoning Quality Metrics
| Metric | Description | Scale |
|--------|-------------|-------|
| `faithfulness` | Is the answer supported by retrieved context? | 0-1 |
| `relevance` | Is the retrieved context relevant to the question? | 0-1 |
| `completeness` | Does the answer fully address the question? | 0-1 |

### Efficiency Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `latency` | End-to-end response time | seconds |
| `llm_calls` | Number of LLM API calls | count |
| `prompt_tokens` | Total input tokens consumed | count |
| `output_tokens` | Total output tokens generated | count |

## LLM-as-Judge Prompts

### Correctness
```
Given the question, ground truth answer, and predicted answer,
judge if the prediction is correct.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Score 1 if the predicted answer conveys the same meaning as ground truth
(exact wording not required). Score 0 if incorrect or missing key information.

Return JSON: {"score": 0 or 1, "reason": "brief explanation"}
```

### Faithfulness
```
Given the context retrieved and the answer generated,
judge if the answer is supported by the context.

Context: {context}
Answer: {answer}

Score 1 if all claims in the answer can be traced to the context.
Score 0 if the answer contains unsupported claims (hallucination).

Return JSON: {"score": 0 or 1, "reason": "brief explanation"}
```

### Relevance
```
Given the question and the context retrieved,
judge if the context is relevant to answering the question.

Question: {question}
Context: {context}

Score 1 if the context contains information useful for answering the question.
Score 0 if the context is unrelated or unhelpful.

Return JSON: {"score": 0 or 1, "reason": "brief explanation"}
```

### Completeness
```
Given the question and the answer generated,
judge if the answer fully addresses the question.

Question: {question}
Answer: {answer}

Score 1 if the answer addresses all aspects of the question.
Score 0 if the answer is partial or misses key aspects.

Return JSON: {"score": 0 or 1, "reason": "brief explanation"}
```

## Module Structure

```
graphrag/
└── eval/
    ├── __init__.py
    ├── runner.py          # Main evaluation orchestrator
    ├── metrics.py         # LLM-as-Judge + reasoning metrics
    ├── results.py         # Result aggregation and output
    └── cli.py             # CLI entry point
```

## Runner Flow

```python
async def run_evaluation(
    qa_dataset_path: str,
    index_configs: dict[str, str],  # {"tt0097576": "tt0097576", ...}
    methods: list[str] = ["tog", "local", "basic"],
) -> EvaluationResults:

    dataset = load_qa_dataset(qa_dataset_path)
    results = []

    for qa_item in dataset:
        imdb_key = qa_item["imdb_key"]
        question = qa_item["question"]
        ground_truth = qa_item["answer"]

        config = load_config(index_configs[imdb_key])

        for method in methods:
            # 1. Run search and capture metrics
            response, metrics = await run_search(
                method=method,
                query=question,
                config=config,
            )

            # 2. Evaluate with LLM-as-Judge
            scores = await evaluate_response(
                question=question,
                ground_truth=ground_truth,
                predicted=response,
                context=metrics.context_text,
            )

            # 3. Store result
            results.append(QueryResult(
                imdb_key=imdb_key,
                question=question,
                method=method,
                response=response,
                scores=scores,
                efficiency=metrics,
            ))

    return aggregate_results(results)
```

## Output Format

### Detailed Results (`eval_results_detailed.json`)
```json
{
  "metadata": {
    "timestamp": "2026-01-29T10:30:00",
    "total_questions": 359,
    "methods": ["tog", "local", "basic"]
  },
  "results": [
    {
      "imdb_key": "tt0097576",
      "question": "Who does the golden crucifix belong to?",
      "ground_truth": "To Coronado",
      "methods": {
        "tog": {
          "response": "The golden crucifix belongs to Coronado...",
          "scores": {
            "correctness": 1,
            "faithfulness": 1,
            "relevance": 1,
            "completeness": 1
          },
          "efficiency": {
            "latency": 3.42,
            "llm_calls": 8,
            "prompt_tokens": 4200,
            "output_tokens": 350
          }
        },
        "local": { "..." : "..." },
        "basic": { "..." : "..." }
      }
    }
  ]
}
```

### Summary Report (`eval_results_summary.json`)
```json
{
  "by_method": {
    "tog":   { "correctness": 0.82, "faithfulness": 0.91, "relevance": 0.87, "completeness": 0.79 },
    "local": { "correctness": 0.75, "faithfulness": 0.88, "relevance": 0.82, "completeness": 0.71 },
    "basic": { "correctness": 0.68, "faithfulness": 0.72, "relevance": 0.75, "completeness": 0.65 }
  },
  "by_movie": {
    "tt0097576": { "tog": {}, "local": {}, "basic": {} },
    "tt0102798": { "tog": {}, "local": {}, "basic": {} },
    "tt2278388": { "tog": {}, "local": {}, "basic": {} }
  },
  "efficiency": {
    "tog":   { "avg_latency": 4.2, "avg_llm_calls": 8, "avg_prompt_tokens": 4500, "avg_output_tokens": 400 },
    "local": { "avg_latency": 2.1, "avg_llm_calls": 2, "avg_prompt_tokens": 3200, "avg_output_tokens": 300 },
    "basic": { "avg_latency": 1.5, "avg_llm_calls": 1, "avg_prompt_tokens": 2800, "avg_output_tokens": 250 }
  }
}
```

## CLI Interface

```bash
# Run full evaluation
graphrag eval --root ./gtog --config eval_config.yaml

# Run specific methods only
graphrag eval --root ./gtog --methods tog,local

# Run specific movie only
graphrag eval --root ./gtog --imdb-key tt0097576

# Resume interrupted evaluation
graphrag eval --root ./gtog --resume
```

### Evaluation Config (`eval_config.yaml`)
```yaml
dataset:
  path: "eval/qa_eval.json"

indexes:
  tt0097576: "tt0097576"
  tt0102798: "tt0102798"
  tt2278388: "tt2278388"

methods:
  - tog
  - local
  - basic

output:
  dir: "eval/results"
  save_intermediate: true

judge:
  model: gpt-5.1  # uses default from settings.yaml
  temperature: 0.0
```

## ToG Modification

Modify `ToGSearch` to return `SearchResult` for metric consistency:

```python
from graphrag.query.structured_search.base import SearchResult

class ToGSearch:
    async def search(self, query: str) -> SearchResult:
        start_time = time.time()
        llm_calls = 0
        prompt_tokens = 0
        output_tokens = 0

        # ... existing exploration logic with metric tracking ...

        return SearchResult(
            response=response,
            context_data=self._format_exploration_paths(all_paths),
            context_text=self._paths_to_text(all_paths),
            completion_time=time.time() - start_time,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
```

## Implementation Phases

### Phase 1: ToG SearchResult Modification
- Modify `ToGSearch` to return `SearchResult`
- Add metric tracking to `LLMPruning` and `ToGReasoning`
- Update `api.tog_search()` to handle new return type
- **Files**: `tog_search/search.py`, `tog_search/pruning.py`, `tog_search/reasoning.py`

### Phase 2: Evaluation Core Module
- Create `graphrag/eval/` package structure
- Implement `metrics.py` with LLM-as-Judge prompts
- Implement `runner.py` with evaluation orchestration
- Implement `results.py` for aggregation and output
- **Files**: New `graphrag/eval/*.py` files

### Phase 3: CLI Integration
- Add `eval` command to CLI
- Create `eval_config.yaml` schema
- Add resume/checkpoint logic for long runs
- **Files**: `graphrag/cli/__init__.py`, `graphrag/cli/eval.py`

### Phase 4: Testing & Validation
- Unit tests for metrics module
- Integration test with small QA subset
- Validate output format correctness
- **Files**: `tests/unit/eval/`, `tests/integration/eval/`

### Dependencies
```
Phase 1 ──► Phase 2 ──► Phase 3
                │
                └──► Phase 4
```

## Estimated LLM Cost

- **Search calls**: 3 methods × 359 questions = 1,077 search calls
- **Judge calls**: 4 metrics × 3 methods × 359 questions = 4,308 judge calls
- **Total**: ~5,385 LLM calls for full evaluation
