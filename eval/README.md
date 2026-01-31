# GraphRAG Evaluation

This directory contains evaluation tools for GraphRAG search methods using LLM-as-Judge metrics.

## Dataset Format

The evaluation dataset should be a JSON file with the following structure:

```json
[
  {
    "question": "Who does the golden crucifix belong to?",
    "ground_truth": "To Coronado",
    "context": "It's the Cross of Coronado! Cortes gave it to him in 1521."
  },
  {
    "question": "What is Indy doing with his Boy Scout troop?",
    "ground_truth": "He is horseback riding",
    "context": "Riders on horseback cross the desert..."
  }
]
```

Each entry contains:
- `question`: The question to evaluate
- `ground_truth`: The expected correct answer
- `context`: Source context from the dataset

## Configuration

Create an `eval_config.yaml` file in your project root:

```yaml
dataset:
  path: "eval/test/test.json"  # Path to your dataset

indexes:
  default: "./output"  # Path to your GraphRAG index

methods:
  - "tog"
  - "local"
  - "basic"

judge:
  temperature: 0.0

output:
  dir: "eval/results"
  save_intermediate: true
```

## Running Evaluation

### Basic Usage

```bash
# Run evaluation with default config (eval_config.yaml)
graphrag eval --root ./my-project

# Specify custom config
graphrag eval --root ./my-project --config ./custom_eval_config.yaml

# Evaluate specific methods only
graphrag eval --root ./my-project --methods tog,local

# Skip LLM evaluation (collect responses only)
graphrag eval --root ./my-project --skip-evaluation

# Resume from checkpoint
graphrag eval --root ./my-project --resume

# Show progress
graphrag eval --root ./my-project --verbose
```

### Output

Results are saved to the configured output directory:

- `eval_results_summary.json` - Aggregated metrics by method
- `eval_results_detailed.json` - Detailed per-query results
- `eval_results_simple.json` - Simple results (when using --skip-evaluation)
- `checkpoint.json` - Intermediate results for resuming

### Simple Results Format

When using `--skip-evaluation`, the output JSON contains:

```json
[
  {
    "question": "Who does the golden crucifix belong to?",
    "response": "The golden crucifix belongs to Coronado.",
    "context": "It's the Cross of Coronado! Cortes gave it to him in 1521.",
    "context_text": "Retrieved context from search...",
    "ground_truth": "To Coronado",
    "method": "tog",
    "latency": 2.5,
    "llm_calls": 5,
    "prompt_tokens": 1000,
    "output_tokens": 200
  }
]
```

## Evaluation Metrics

When not using `--skip-evaluation`, the system evaluates responses using:

- **Correctness**: Does the response match the ground truth?
- **Faithfulness**: Is the response supported by the retrieved context?
- **Relevance**: Is the retrieved context relevant to the question?
- **Completeness**: Does the response fully address the question?

Each metric is scored 0 or 1 by an LLM judge.

## Example Workflow

1. Prepare your dataset in the correct JSON format
2. Create `eval_config.yaml` pointing to your dataset and index
3. Run evaluation:
   ```bash
   graphrag eval --root ./my-project --verbose
   ```
4. Check results in `eval/results/eval_results_summary.json`

## Test Dataset Statistics

Current test datasets:
- tt0097576: 102 questions
- tt0443706: 50 questions  
- tt2278388: 206 questions
