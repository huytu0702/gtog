# Hướng dẫn chạy Eval

## Tổng quan

Quy trình eval gồm 2 bước:

1. **Thu thập kết quả** (`graphrag eval --skip-evaluation`) — chạy search, lưu response và efficiency metrics (latency, llm_calls, input_tokens, output_tokens), không gọi LLM judge.
2. **Đánh giá với Ragas** (`ragas_runner`) — đọc file kết quả từ bước 1, tính các metrics chất lượng (context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness_custom).

---

## Bước 1: Thu thập kết quả (skip evaluation)

### Cú pháp

```bash
.venv/Scripts/python -m graphrag eval \
  --root <project_root> \
  --config <absolute_path_to_eval_config.yaml> \
  --skip-evaluation \
  --verbose
```

### Lưu ý quan trọng

- `--root` là thư mục gốc của project GraphRAG (chứa `settings.yaml`).
- `--config` phải là **đường dẫn tuyệt đối** đến file eval config (typer validate path từ CWD).
- Proxy/LLM server phải đang chạy vì ToG search cần gọi LLM để search.

### Ví dụ — subset 1 (test nhanh)

```bash
cd F:/KL/gtog-eval

.venv/Scripts/python -m graphrag eval \
  --root medical_graphrag_project \
  --config "F:/KL/gtog-eval/medical_graphrag_project/eval_config_subset_1.yaml" \
  --skip-evaluation \
  --verbose
```

### Ví dụ — subset 200 (full eval)

```bash
cd F:/KL/gtog-eval

.venv/Scripts/python -m graphrag eval \
  --root medical_graphrag_project \
  --config "F:/KL/gtog-eval/medical_graphrag_project/eval_config_subset_200.yaml" \
  --skip-evaluation \
  --verbose
```

### Resume từ checkpoint

Thêm `--resume` để tiếp tục từ checkpoint nếu bị gián đoạn:

```bash
.venv/Scripts/python -m graphrag eval \
  --root medical_graphrag_project \
  --config "F:/KL/gtog-eval/medical_graphrag_project/eval_config_subset_200.yaml" \
  --skip-evaluation \
  --resume \
  --verbose
```

### Output

| File | Mô tả |
|------|-------|
| `<output_dir>/eval_results_simple.json` | Danh sách kết quả gồm question, response, context_text, ground_truth, method, và efficiency |
| `<output_dir>/checkpoint.json` | Checkpoint để resume (nếu `save_intermediate: true`) |

Cấu trúc mỗi entry trong `eval_results_simple.json`:

```json
{
  "question": "...",
  "response": "...",
  "context": "...",
  "context_text": "...",
  "ground_truth": "...",
  "method": "tog",
  "efficiency": {
    "latency": 12.34,
    "llm_calls": 3,
    "input_tokens": 7571,
    "output_tokens": 41
  }
}
```

---

## Bước 2: Đánh giá với Ragas

### Cú pháp

```bash
.venv/Scripts/python -m graphrag.eval.ragas_runner \
  --input <path_to_eval_results_simple.json> \
  --output-dir <output_directory> \
  --settings <path_to_settings.yaml>
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--input` | *(bắt buộc)* | Path đến `eval_results_simple.json` từ bước 1 |
| `--output-dir` | Cùng thư mục với `--input` | Thư mục lưu kết quả Ragas |
| `--settings` | `backend/settings.yaml` | Path đến `settings.yaml` của GraphRAG project (dùng embedding model) |
| `--model` | `gpt-5.2` | Model LLM cho Ragas judge |
| `--base-url` | `http://127.0.0.1:8317/v1` | Base URL của OpenAI-compatible proxy |
| `--api-key` | `proxypal-local` | API key |
| `--timeout` | `120.0` | Timeout mỗi request (giây) |
| `--max-retries` | `5` | Số lần retry |
| `--metric` | Tất cả metrics | Chỉ chạy metric cụ thể (có thể lặp lại) |

### Metrics có sẵn

| Metric | Mô tả |
|--------|-------|
| `context_precision` | Độ chính xác của context được retrieve |
| `context_recall` | Độ bao phủ của context so với ground truth |
| `faithfulness` | Response có trung thực với context không |
| `answer_relevancy` | Response có liên quan đến câu hỏi không |
| `answer_correctness_custom` | LLM judge chấm đúng/sai so với ground truth |

### Ví dụ — subset 1

```bash
cd F:/KL/gtog-eval

.venv/Scripts/python -m graphrag.eval.ragas_runner \
  --input eval_results_subset_1/eval_results_simple.json \
  --output-dir eval_results_subset_1 \
  --settings medical_graphrag_project/settings.yaml
```

### Ví dụ — subset 200

```bash
cd F:/KL/gtog-eval

.venv/Scripts/python -m graphrag.eval.ragas_runner \
  --input eval_results_medical/eval_results_simple.json \
  --output-dir eval_results_medical \
  --settings medical_graphrag_project/settings.yaml
```

### Ví dụ — chỉ chạy một số metric

```bash
.venv/Scripts/python -m graphrag.eval.ragas_runner \
  --input eval_results_subset_1/eval_results_simple.json \
  --output-dir eval_results_subset_1 \
  --settings medical_graphrag_project/settings.yaml \
  --metric faithfulness \
  --metric answer_correctness_custom
```

### Output

| File | Mô tả |
|------|-------|
| `eval_results_ragas_detailed.json` | Điểm từng câu hỏi, kèm status (success/failed/skipped) |
| `eval_results_ragas_summary.json` | Điểm trung bình theo method |

Cấu trúc `eval_results_ragas_summary.json`:

```json
{
  "metadata": {
    "timestamp": "...",
    "model": "gpt-5.2",
    "base_url": "http://127.0.0.1:8317/v1"
  },
  "overall": {
    "count": 200,
    "success_count": 198,
    "fail_count": 2,
    "skip_count": 0
  },
  "by_method": {
    "tog": {
      "count": 100,
      "success_count": 99,
      "context_precision": 0.85,
      "context_recall": 0.78,
      "faithfulness": 0.91,
      "answer_relevancy": 0.88,
      "answer_correctness_custom": 0.82
    }
  }
}
```

---

## Cấu hình eval config

```yaml
# eval_config_subset_200.yaml
dataset:
  path: "F:/KL/gtog-eval/Datasets/Questions/medical_questions_eval_subset_200.json"

indexes:
  medical: "F:/KL/gtog-eval/medical_graphrag_project"

methods:
  - tog
  - local

output:
  dir: "eval_results_medical"   # relative to project root
  save_intermediate: true        # lưu checkpoint

judge:
  model: null       # không dùng (skip_evaluation mode)
  temperature: 0.0
```

---

## Workflow đầy đủ

```bash
cd F:/KL/gtog-eval

# 1. Đảm bảo proxy đang chạy trước khi chạy lệnh này

# 2. Thu thập kết quả
.venv/Scripts/python -m graphrag eval \
  --root medical_graphrag_project \
  --config "F:/KL/gtog-eval/medical_graphrag_project/eval_config_subset_200.yaml" \
  --skip-evaluation \
  --verbose

# 3. Đánh giá với Ragas
.venv/Scripts/python -m graphrag.eval.ragas_runner \
  --input eval_results_medical/eval_results_simple.json \
  --output-dir eval_results_medical \
  --settings medical_graphrag_project/settings.yaml
```
