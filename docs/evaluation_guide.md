# Hướng Dẫn Test Chức Năng Evaluation

Hướng dẫn này mô tả cách test chức năng evaluation của GraphRAG sử dụng phương pháp ToG (Think-on-Graph).

## 1. Chuẩn Bị Dữ Liệu Test

### 1.1 Tạo Thư Mục Dự Án

```bash
mkdir -p eval_test_project/input
```

### 1.2 Chuẩn Bị File Dữ Liệu

Tạo file `eval_test_project/input/test.txt` chứa nội dung văn bản cần index. Ví dụ: kịch bản phim Indiana Jones.

### 1.3 Chuẩn Bị Ground Truth

Tạo file `eval_test_project/ground_truth.json` với định dạng:

```json
[
    {
        "question": "Câu hỏi 1?",
        "ground_truth": "Câu trả lờii đúng",
        "context": "Ngữ cảnh từ tài liệu"
    },
    {
        "question": "Câu hỏi 2?",
        "ground_truth": "Câu trả lờii đúng",
        "context": "Ngữ cảnh từ tài liệu"
    }
]
```

## 2. Khởi Tạo Dự Án

### 2.1 Initialize GraphRAG Project

```bash
graphrag init --root ./eval_test_project
```

Lệnh này sẽ tạo các file cấu hình cần thiết:
- `settings.yaml` - Cấu hình chính
- `.env` - Biến môi trường
- `prompts/` - Các file prompt

### 2.2 Cấu Hình API Key

Chỉnh sửa file `eval_test_project/.env`:

```env
GRAPHRAG_API_KEY=your_api_key_here
```

## 3. Index Dữ Liệu

Chạy lệnh index để xây dựng knowledge graph:

```bash
graphrag index --root ./eval_test_project --verbose
```

Quá trình này sẽ:
- Load tài liệu từ `input/`
- Tách thành các text units
- Trích xuất entities và relationships
- Tạo community reports
- Generate embeddings

Kết quả sẽ được lưu trong `eval_test_project/output/`.

## 4. Tạo Cấu Hình Evaluation

Tạo file `eval_test_project/eval_config.yaml`:

```yaml
# eval_config.yaml
dataset:
  path: "ground_truth.json"

indexes:
  test: "eval_test_project"

methods:
  - tog

output:
  dir: "eval_results"
  save_intermediate: true

judge:
  model: null  # Sử dụng model mặc định từ settings.yaml
  temperature: 0.0
```

**Lưu ý quan trọng**: 
- `indexes.test` phải trỏ đến thư mục chứa `settings.yaml`
- Nếu chạy từ thư mục gốc, dùng `"eval_test_project"` thay vì `"."`

## 5. Chạy Evaluation

### 5.1 Chạy Với Đánh Giá LLM-as-Judge

```bash
graphrag eval --root ./eval_test_project --methods tog --verbose
```

### 5.2 Chạy Không Đánh Giá (Chỉ Lưu Kết Quả)

```bash
graphrag eval --root ./eval_test_project --methods tog --verbose --skip-evaluation
```

Hoặc dùng flag ngắn:

```bash
graphrag eval --root ./eval_test_project --methods tog -s
```

## 6. Xem Kết Quả

Kết quả được lưu tại `eval_results/eval_results_simple.json`:

```json
[
  {
    "question": "Who does the golden crucifix the grave robbers had belong to?",
    "response": "Câu trả lờii từ ToG...",
    "context": "Ngữ cảnh ground truth",
    "context_text": "Ngữ cảnh từ knowledge graph",
    "ground_truth": "To Coronado",
    "method": "tog"
  }
]
```

## 7. Các Phương Pháp Đánh Giá Khác

Có thể đánh giá nhiều phương pháp cùng lúc:

```bash
graphrag eval --root ./eval_test_project --methods tog,local,basic --verbose
```

Các phương pháp hỗ trợ:
- `tog` - Think-on-Graph (deep reasoning)
- `local` - Local search
- `basic` - Basic vector search
- `global` - Global search
- `drift` - DRIFT search

## 8. Cấu Trúc Thư Mục Sau Khi Hoàn Thành

```
eval_test_project/
├── input/
│   └── test.txt              # Tài liệu gốc
├── output/                   # Kết quả index
│   ├── entities.parquet
│   ├── relationships.parquet
│   ├── communities.parquet
│   └── lancedb/             # Vector database
├── prompts/                  # System prompts
├── ground_truth.json        # QA pairs
├── eval_config.yaml         # Cấu hình evaluation
├── settings.yaml            # Cấu hình GraphRAG
└── .env                     # API keys

eval_results/
└── eval_results_simple.json # Kết quả đánh giá
```

## 9. Troubleshooting

### Lỗi: "Could not find entities.parquet in storage!"

**Nguyên nhân**: Đường dẫn trong `indexes` không đúng.

**Giải pháp**: 
- Đảm bảo `indexes.test` trỏ đến thư mục chứa `settings.yaml`
- Kiểm tra đã chạy `graphrag index` thành công chưa

### Lỗi: "Config file not found"

**Nguyên nhân**: Thư mục index không có `settings.yaml`.

**Giải pháp**: Chạy `graphrag init` trong thư mục dự án.

### Lỗi: SSL transport

Các lỗi SSL ở cuối quá trình index thường là warnings không ảnh hưởng đến kết quả.

## 10. Tùy Chỉnh Cấu Hình ToG

Chỉnh sửa `settings.yaml` để tùy chỉnh ToG:

```yaml
tog_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  width: 3                          # Beam width
  depth: 3                          # Maximum exploration depth
  prune_strategy: llm               # llm, semantic, hoặc bm25
  num_retain_entity: 3              # Entities giữ lại mỗi level
  temperature_exploration: 0.4      # Temperature cho exploration
  temperature_reasoning: 0.0        # Temperature cho final answer
  max_context_tokens: 8000          # Maximum context tokens
  max_exploration_paths: 10         # Maximum exploration paths
```
