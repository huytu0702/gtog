# Using the GraphRAG Evaluation Framework

This guide explains how to run evaluations comparing ToG, Local, and Basic search methods using your movie QA dataset.

## Prerequisites

Before running an evaluation, ensure:

1. **GraphRAG indexes are built** for all movies in your dataset
2. **GraphRAG CLI is installed** and accessible
3. **LLM API key** is configured in your settings.yaml
4. **Evaluation dataset** exists at `eval/qa_eval.json`

## Quick Start

Run a full evaluation with all methods:

```bash
graphrag eval --root ./gtog --config eval_config.yaml
```

## Configuration File

Create an `eval_config.yaml` file in your project root:

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
  model: gpt-4o-mini  # or your preferred model
  temperature: 0.0
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `dataset.path` | Path to QA dataset JSON file | Required |
| `indexes` | Map of IMDb keys to folder names | Required |
| `methods` | Search methods to evaluate | `["tog", "local", "basic"]` |
| `output.dir` | Directory for result files | `"eval/results"` |
| `output.save_intermediate` | Save checkpoints during run | `false` |
| `judge.model` | LLM model for evaluation | From settings.yaml |
| `judge.temperature` | Temperature for LLM judge | `0.0` |

## Command Options

### Full Evaluation

```bash
graphrag eval --root ./gtog --config eval_config.yaml
```

### Run Specific Methods Only

```bash
graphrag eval --root ./gtog --config eval_config.yaml --methods tog,local
```

### Run Specific Movie Only

```bash
graphrag eval --root ./gtog --config eval_config.yaml --imdb-key tt0097576
```

### Resume Interrupted Evaluation

```bash
graphrag eval --root ./gtog --config eval_config.yaml --resume
```

### Custom Output Directory

```bash
graphrag eval --root ./gtog --config eval_config.yaml --output-dir ./my_results
```

## Output Files

After running, two result files are generated:

### 1. `eval_results_detailed.json`

Contains per-question results with scores and metrics for each method:

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
        }
      }
    }
  ]
}
```

### 2. `eval_results_summary.json`

Contains aggregated statistics by method and movie:

```json
{
  "by_method": {
    "tog": {
      "correctness": 0.82,
      "faithfulness": 0.91,
      "relevance": 0.87,
      "completeness": 0.79
    },
    "local": {
      "correctness": 0.75,
      "faithfulness": 0.88,
      "relevance": 0.82,
      "completeness": 0.71
    },
    "basic": {
      "correctness": 0.68,
      "faithfulness": 0.72,
      "relevance": 0.75,
      "completeness": 0.65
    }
  },
  "by_movie": {
    "tt0097576": { "tog": {}, "local": {}, "basic": {} }
  },
  "efficiency": {
    "tog": {
      "avg_latency": 4.2,
      "avg_llm_calls": 8,
      "avg_prompt_tokens": 4500,
      "avg_output_tokens": 400
    }
  }
}
```

## Metrics Explained

### Accuracy Metrics

| Metric | Description | Scale |
|--------|-------------|-------|
| `correctness` | Does predicted answer match ground truth? | 0-1 |

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

## Dataset Format

Your `qa_eval.json` should follow this format:

```json
[
  {
    "imdb_key": "tt0097576",
    "question": "Who does the golden crucifix belong to?",
    "answer": "To Coronado"
  },
  {
    "imdb_key": "tt0102798",
    "question": "What happens in the final scene?",
    "answer": "The protagonist escapes"
  }
]
```

## Complete Example

### Step 1: Create eval_config.yaml

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

judge:
  model: gpt-4o-mini
  temperature: 0.0
```

### Step 2: Run Evaluation

```bash
graphrag eval --root ./gtog --config eval_config.yaml
```

### Step 3: View Results

```bash
# View summary
cat eval/results/eval_results_summary.json

# View detailed results (format for readability)
cat eval/results/eval_results_detailed.json | jq
```

## Tips

1. **Start small**: Test with a subset of questions using `--imdb-key` before full evaluation
2. **Save checkpoints**: Enable `save_intermediate: true` for long-running evaluations
3. **Monitor costs**: Each question uses ~4 judge LLM calls × number of methods
4. **Resume capability**: Use `--resume` to continue interrupted evaluations

## Troubleshooting

### "Index not found" error

Ensure each movie folder in `indexes` has been indexed:

```bash
graphrag index --root ./tt0097576
graphrag index --root ./tt0102798
graphrag index --root ./tt2278388
```

### "LLM API key not configured" error

Check your settings.yaml has the LLM configuration:

```yaml
llm:
  type: openai_chat
  api_key: ${GRAPHRAG_OPENAI_API_KEY}
  model: gpt-4o-mini
```

### "Dataset file not found" error

Verify the path in `dataset.path` is correct relative to the `--root` directory.

## Cost Estimation

For a full evaluation on 359 questions with 3 methods:

- **Search calls**: 1,077 (3 methods × 359 questions)
- **Judge calls**: 4,308 (4 metrics × 3 methods × 359 questions)
- **Total**: ~5,385 LLM calls

Adjust your method selection or question subset to control costs.
