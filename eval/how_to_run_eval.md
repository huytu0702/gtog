# How to Run GraphRAG Evaluation

This guide shows the exact workflow to run `graphrag eval` in this repo, including **ToG-only** and `--skip-evaluation` mode.

## 1) Prerequisites

- Repo root: `F:/KL/gtog`
- Virtual environment exists: `F:/KL/gtog/.venv`
- API key is set in `.env` (`GRAPHRAG_API_KEY=...`)

If `graphrag` is not found, use the venv executable directly:

```bash
"/f/KL/gtog/.venv/Scripts/graphrag.exe" --help
```

## 2) Create an evaluation project folder

Example folder:

```bash
"/f/KL/gtog/.venv/Scripts/graphrag.exe" init --root "/f/KL/gtog/eval/tt0443706_tog_skip"
```

## 3) Prepare files

For `tt0443706`, copy:

- Input text to `input/`
- QA dataset JSON to project root
- `settings.yaml` from `graphrag/settings.yaml`
- `.env` from repo root

Example:

```bash
mkdir -p "/f/KL/gtog/eval/tt0443706_tog_skip/input"
cp "/f/KL/gtog/eval/tt0443706/tt0443706.txt" "/f/KL/gtog/eval/tt0443706_tog_skip/input/tt0443706.txt"
cp "/f/KL/gtog/eval/tt0443706/qa_tt0443706.json" "/f/KL/gtog/eval/tt0443706_tog_skip/qa_tt0443706.json"
cp "/f/KL/gtog/graphrag/settings.yaml" "/f/KL/gtog/eval/tt0443706_tog_skip/settings.yaml"
cp "/f/KL/gtog/.env" "/f/KL/gtog/eval/tt0443706_tog_skip/.env"
```

## 4) Create `eval_config.yaml`

`F:/KL/gtog/eval/tt0443706_tog_skip/eval_config.yaml`

```yaml
dataset:
  path: "qa_tt0443706.json"

indexes:
  tt0443706: "F:/KL/gtog/eval/tt0443706_tog_skip"

methods:
  - tog

output:
  dir: "eval_results"
  save_intermediate: true

judge:
  model: null
  temperature: 0.0
```

> Important: use an index path that points to the folder containing `settings.yaml` and `output/*.parquet`.

## 5) Build index

```bash
"/f/KL/gtog/.venv/Scripts/graphrag.exe" index --root "/f/KL/gtog/eval/tt0443706_tog_skip" --verbose
```

Verify index files exist (at least):

- `output/entities.parquet`
- `output/relationships.parquet`
- `output/text_units.parquet`

## 6) Run ToG evaluation with `--skip-evaluation`

Recommended on Windows terminal encoding:

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
"/f/KL/gtog/.venv/Scripts/python.exe" -X utf8 -m graphrag eval \
  --root "/f/KL/gtog/eval/tt0443706_tog_skip" \
  --methods tog \
  --skip-evaluation \
  --verbose
```

Output file:

- `eval_results/eval_results_simple.json` (relative to `--root` unless absolute path is set in config)

## 7) Retry only failed Q&A (optional)

If some rows have `response` starting with `ERROR:`, create a retry dataset and run eval again on failed rows only.

### 7.1 Create retry dataset

```bash
"/f/KL/gtog/.venv/Scripts/python.exe" -X utf8 -c "import json, pathlib; base=pathlib.Path('F:/KL/gtog/eval/tt0443706_tog_skip'); src=base/'eval_results'/'eval_results_simple.json'; d=json.load(open(src,encoding='utf-8')); failed=[r for r in d if str(r.get('response','')).startswith('ERROR:')]; out=[{'question':r['question'],'ground_truth':r.get('ground_truth',''),'context':r.get('context','')} for r in failed]; json.dump(out, open(base/'qa_failed_retry.json','w',encoding='utf-8'), ensure_ascii=False, indent=2); print(len(out))"
```

### 7.2 Retry config

`F:/KL/gtog/eval/tt0443706_tog_skip/eval_retry_config.yaml`

```yaml
dataset:
  path: "qa_failed_retry.json"

indexes:
  tt0443706: "F:/KL/gtog/eval/tt0443706_tog_skip"

methods:
  - tog

output:
  dir: "F:/KL/gtog/eval/tt0443706_tog_skip/eval_results_retry"
  save_intermediate: true

judge:
  model: null
  temperature: 0.0
```

### 7.3 Run retry

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
"/f/KL/gtog/.venv/Scripts/python.exe" -X utf8 -m graphrag eval \
  --root "/f/KL/gtog/eval/tt0443706_tog_skip" \
  --config "/f/KL/gtog/eval/tt0443706_tog_skip/eval_retry_config.yaml" \
  --methods tog \
  --skip-evaluation \
  --verbose
```

## 8) Common errors

### `graphrag: command not found`
Use venv executable:

```bash
"/f/KL/gtog/.venv/Scripts/graphrag.exe" ...
```

### `Could not find entities.parquet in storage!`
- Index was not built, or
- `indexes` path in eval config is wrong.

Fix: run `index` first and set `indexes.<key>` to the correct project root.

### `UnicodeDecodeError ... cp1252`
Run eval with UTF-8 flags (section 6 command).
