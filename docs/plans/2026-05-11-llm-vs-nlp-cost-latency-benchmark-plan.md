# Plan: Benchmark cost và latency giữa LLM graph extraction và NLP graph extraction

## Mục tiêu

So sánh định lượng cost và latency của quá trình GraphRAG khi tạo graph bằng LLM so với tạo graph bằng NLP, đồng thời đo tác động của hai loại graph này lên latency/cost của query.

Repo hiện tại đã hỗ trợ hai hướng indexing:

- `standard`: dùng LLM để tạo graph qua workflow `extract_graph`.
- `fast`: dùng NLP để tạo graph qua workflow `extract_graph_nlp`, sau đó vẫn có thể dùng LLM ở các bước khác như community report và query.

## Câu hỏi benchmark cần trả lời

1. Indexing bằng LLM graph extraction tốn bao nhiêu thời gian và chi phí?
2. Indexing bằng NLP graph extraction tốn bao nhiêu thời gian và chi phí?
3. Query trên graph tạo bởi LLM có latency/cost khác query trên graph tạo bởi NLP như thế nào?
4. Nếu tính cả chất lượng trả lời, NLP graph extraction có giảm chất lượng đáng kể so với LLM graph extraction không?
5. Với cùng một dataset, phương án nào có cost/latency tốt hơn cho production?

## Phạm vi benchmark

### Trong phạm vi

- Indexing end-to-end bằng `--method standard` và `--method fast`.
- Latency từng workflow trong indexing.
- Query latency/cost trên cùng bộ câu hỏi.
- Các search method chính: `local`, `basic`, `tog`; có thể thêm `global` và `drift` nếu cần.
- Token usage và estimated provider cost cho các bước có dùng LLM.
- So sánh chất lượng trả lời nếu có ground truth/eval set.

### Ngoài phạm vi ban đầu

- Tối ưu thuật toán NLP extraction.
- Thay model LLM để benchmark model-vs-model.
- Benchmark production traffic thật.
- Thay đổi kiến trúc GraphRAG hoặc query pipeline.

## Nguyên tắc benchmark

Để kết quả công bằng, hai run phải giữ nguyên:

- cùng input documents;
- cùng chunking config;
- cùng embedding model;
- cùng query model;
- cùng vector store type;
- cùng query set;
- cùng machine/runtime environment;
- cùng cache policy;
- cùng số lần chạy lặp lại.

Chỉ thay đổi graph extraction method:

- LLM mode: `graphrag index --method standard`
- NLP mode: `graphrag index --method fast`

## Cấu trúc thư mục đề xuất

```text
benchmarks/
  llm_graph/
    input/
    settings.yaml
    prompts/
    output/
    cache/
  nlp_graph/
    input/
    settings.yaml
    prompts/
    output/
    cache/
  questions/
    benchmark_questions.jsonl
  reports/
    llm_graph_index_stats.json
    nlp_graph_index_stats.json
    llm_graph_query_results.json
    nlp_graph_query_results.json
    comparison.md
```

## Cấu hình indexing

### LLM graph extraction

Dùng pipeline chuẩn:

```bash
uv run graphrag index --root ./benchmarks/llm_graph --method standard --no-cache
```

Workflow chính cần đo:

- `extract_graph`
- `finalize_graph`
- `create_communities`
- `create_community_reports`
- `generate_text_embeddings`

### NLP graph extraction

Dùng pipeline fast:

```bash
uv run graphrag index --root ./benchmarks/nlp_graph --method fast --no-cache
```

Workflow chính cần đo:

- `extract_graph_nlp`
- `prune_graph`
- `finalize_graph`
- `create_communities`
- `create_community_reports_text`
- `generate_text_embeddings`

## Cache policy

Benchmark lần đầu nên chạy với `--no-cache` để đo chi phí thật.

Nếu muốn đo production repeated-run behavior, thêm benchmark thứ hai với cache bật:

```bash
uv run graphrag index --root ./benchmarks/llm_graph --method standard
uv run graphrag index --root ./benchmarks/nlp_graph --method fast
```

Báo cáo cần tách rõ:

- cold run: cache off hoặc cache trống;
- warm run: cache có sẵn.

## Metrics cần thu thập cho indexing

### Latency

Lấy từ `output/stats.json` sau index:

- `total_runtime`
- `workflows.extract_graph.overall`
- `workflows.extract_graph_nlp.overall`
- `workflows.create_community_reports.overall`
- `workflows.create_community_reports_text.overall`
- `workflows.generate_text_embeddings.overall`

### Cost

Tính cost theo công thức:

```text
llm_cost = input_tokens / 1_000_000 * input_price_per_1m
         + output_tokens / 1_000_000 * output_price_per_1m
```

Tách thành:

```text
index_cost = graph_extraction_llm_cost
           + community_report_llm_cost
           + embedding_cost
           + infrastructure_cost
```

Với NLP graph extraction:

```text
nlp_graph_extraction_cost ≈ runtime_hours * machine_cost_per_hour
```

Lưu ý: `fast` không dùng LLM cho graph extraction, nhưng vẫn có thể dùng LLM cho community reports và query, nên không được ghi là “zero LLM cost” cho toàn bộ indexing.

## Metrics cần thu thập cho query

Chạy cùng query set trên hai root đã index:

```bash
uv run graphrag eval --root ./benchmarks/llm_graph --methods local,basic,tog --skip-evaluation
uv run graphrag eval --root ./benchmarks/nlp_graph --methods local,basic,tog --skip-evaluation
```

Dùng `--skip-evaluation` khi đo cost/latency để tránh LLM-as-Judge làm nhiễu số liệu.

Thu thập:

- `avg_latency`
- `avg_llm_calls`
- `avg_prompt_tokens`
- `avg_output_tokens`
- error rate
- timeout rate

Cost/query:

```text
query_cost_per_question = avg_prompt_tokens / 1_000_000 * input_price_per_1m
                        + avg_output_tokens / 1_000_000 * output_price_per_1m
```

## Metrics chất lượng trả lời

Nếu có ground truth, chạy eval có LLM-as-Judge riêng sau khi đã hoàn tất benchmark cost/latency:

```bash
uv run graphrag eval --root ./benchmarks/llm_graph --methods local,basic,tog
uv run graphrag eval --root ./benchmarks/nlp_graph --methods local,basic,tog
```

Thu thập:

- relevance;
- completeness;
- faithfulness nếu eval hỗ trợ;
- answer length;
- số câu lỗi hoặc không trả lời được.

Không trộn cost/latency của LLM-as-Judge vào query benchmark chính.

## Bảng báo cáo đề xuất

### Indexing summary

| Mode | Method | Total latency | Graph build latency | LLM calls | Input tokens | Output tokens | Estimated cost |
|---|---|---:|---:|---:|---:|---:|---:|
| LLM graph | standard | TBD | TBD | TBD | TBD | TBD | TBD |
| NLP graph | fast | TBD | TBD | TBD | TBD | TBD | TBD |

### Query summary

| Graph source | Query method | Avg latency | Avg LLM calls | Avg input tokens | Avg output tokens | Cost/question |
|---|---|---:|---:|---:|---:|---:|
| LLM graph | local | TBD | TBD | TBD | TBD | TBD |
| NLP graph | local | TBD | TBD | TBD | TBD | TBD |
| LLM graph | basic | TBD | TBD | TBD | TBD | TBD |
| NLP graph | basic | TBD | TBD | TBD | TBD | TBD |
| LLM graph | tog | TBD | TBD | TBD | TBD | TBD |
| NLP graph | tog | TBD | TBD | TBD | TBD | TBD |

### Quality summary

| Graph source | Query method | Relevance | Completeness | Error rate | Notes |
|---|---|---:|---:|---:|---|
| LLM graph | local | TBD | TBD | TBD | TBD |
| NLP graph | local | TBD | TBD | TBD | TBD |
| LLM graph | tog | TBD | TBD | TBD | TBD |
| NLP graph | tog | TBD | TBD | TBD | TBD |

## Các bước thực hiện

### Phase 1: Chuẩn bị benchmark dataset

1. Chọn một dataset đại diện cho production hoặc eval hiện tại.
2. Copy cùng input vào `benchmarks/llm_graph/input` và `benchmarks/nlp_graph/input`.
3. Copy cùng `settings.yaml` và `prompts` sang hai root benchmark.
4. Đảm bảo chỉ khác indexing method khi chạy command, không khác config ngầm.
5. Tạo file query set cố định.

Deliverable:

- Hai benchmark roots có input/config giống nhau.
- Query set cố định.

### Phase 2: Chạy indexing benchmark

1. Xoá hoặc tách riêng output/cache cũ.
2. Chạy `standard` với cache off.
3. Chạy `fast` với cache off.
4. Lưu `stats.json` của hai run vào `benchmarks/reports`.
5. Lặp lại 3 lần nếu cần số liệu ổn định.
6. Tính median latency thay vì chỉ lấy một lần chạy.

Deliverable:

- Index stats cho LLM graph.
- Index stats cho NLP graph.
- Bảng latency từng workflow.

### Phase 3: Thu thập token/cost indexing

1. Xác định nơi provider trả token usage hoặc log hiện tại của GraphRAG.
2. Nếu log token usage chưa đầy đủ, thêm instrumentation nhẹ quanh language model provider để ghi:
   - model;
   - operation/workflow;
   - prompt tokens;
   - completion tokens;
   - total tokens;
   - latency/request;
   - error/retry count.
3. Áp bảng giá model đang dùng để tính estimated cost.
4. Tách riêng cost của:
   - graph extraction;
   - community report;
   - embeddings.

Deliverable:

- Cost report cho indexing.
- Mapping token usage theo workflow.

### Phase 4: Chạy query benchmark

1. Chạy eval với `--skip-evaluation` trên root LLM graph.
2. Chạy eval với `--skip-evaluation` trên root NLP graph.
3. Dùng cùng `--methods` và cùng query set.
4. Lưu raw eval outputs.
5. Tính trung bình và median cho latency/token/cost theo method.

Deliverable:

- Query efficiency report cho mỗi graph source.
- Cost/question theo query method.

### Phase 5: Chạy quality benchmark

1. Chạy eval có LLM-as-Judge nếu có ground truth.
2. Tách riêng file kết quả quality khỏi file latency/cost.
3. So sánh chất lượng giữa graph LLM và graph NLP theo từng method.
4. Ghi nhận câu hỏi NLP trả lời kém hơn hoặc tốt hơn LLM.

Deliverable:

- Quality comparison report.
- Danh sách câu hỏi có chênh lệch đáng kể.

### Phase 6: Tổng hợp kết luận

1. Tổng hợp index cost/latency.
2. Tổng hợp query cost/latency.
3. Tổng hợp quality delta.
4. Đưa ra recommendation:
   - dùng LLM graph khi cần chất lượng/semantic relationship tốt hơn;
   - dùng NLP graph khi cần index nhanh/rẻ;
   - dùng hybrid nếu NLP đủ tốt cho phần lớn dữ liệu và LLM chỉ dùng cho tài liệu quan trọng.

Deliverable:

- `benchmarks/reports/comparison.md`
- Recommendation cuối cùng cho production/eval.

## Rủi ro và cách kiểm soát

### Cache làm sai lệch số liệu

Mitigation:

- Chạy cold benchmark với `--no-cache`.
- Lưu rõ cache state trong report.

### LLM-as-Judge làm tăng query cost giả

Mitigation:

- Dùng `--skip-evaluation` cho cost/latency benchmark.
- Chạy quality eval riêng.

### NLP graph có graph topology khác quá nhiều

Mitigation:

- Báo cáo thêm số node, số edge, số community, density.
- Không chỉ so latency/cost; phải so cả quality.

### Provider latency biến động

Mitigation:

- Chạy nhiều lần.
- Lấy median và p95 nếu đủ mẫu.
- Ghi lại thời điểm chạy và model/provider.

### So sánh end-to-end bị lẫn nhiều chi phí

Mitigation:

- Tách hai góc nhìn:
  - graph extraction only;
  - full indexing pipeline.

## Instrumentation đề xuất nếu số liệu token chưa đủ

Nếu eval/index hiện chưa ghi đủ token usage theo workflow, thêm một lớp logging ở language model provider để ghi JSONL:

```json
{"phase":"index","workflow":"extract_graph","model":"...","prompt_tokens":1234,"output_tokens":567,"latency":2.34,"success":true}
```

File output đề xuất:

```text
output/llm_usage.jsonl
```

Các field tối thiểu:

- `phase`: `index` hoặc `query`
- `workflow` hoặc `query_method`
- `model`
- `prompt_tokens`
- `output_tokens`
- `total_tokens`
- `latency`
- `success`
- `error_type`
- `retry_count`

## Definition of Done

Benchmark được xem là hoàn tất khi có đủ:

- Hai root benchmark chạy thành công: LLM graph và NLP graph.
- Index latency report từ `stats.json`.
- Query efficiency report với cùng query set.
- Estimated cost cho indexing và query.
- Quality comparison nếu có ground truth.
- Final recommendation dựa trên cost, latency và quality.
