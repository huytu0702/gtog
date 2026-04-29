# Vì Sao ToG Tốt Hơn Local Search Ở Complex Reasoning

Tài liệu này giải thích vì sao **ToG Search** cho kết quả tốt hơn **Local Search** trên nhóm câu hỏi **Complex Reasoning**, dựa trên:

- Kết quả đánh giá trong hình bạn cung cấp
- Mã nguồn tại `graphrag/query/structured_search/tog_search/`
- Mã nguồn tại `graphrag/query/structured_search/local_search/`

## 1. Kết quả trong hình cho thấy điều gì?

Ở nhóm **Complex Reasoning**, các chỉ số là:

| Method | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness |
|---|---:|---:|---:|---:|---:|
| Local Search | 0.98 | 0.81 | 0.82 | 0.90 | 0.84 |
| ToG Search | 0.99 | 0.98 | 0.83 | 0.91 | 0.95 |

Chênh lệch đáng chú ý nhất:

- **Context Recall**: `+0.17`
- **Answer Correctness**: `+0.11`
- **Faithfulness**: `+0.01`
- **Answer Relevancy**: `+0.01`

Điều này nói lên một điểm rất rõ: **ToG không chỉ lấy đúng ngữ cảnh, mà còn lấy đủ chuỗi bằng chứng hơn Local Search**, nên khi câu hỏi đòi hỏi suy luận nhiều bước, câu trả lời cuối cùng chính xác hơn hẳn.

## 2. Local Search mạnh ở đâu, và giới hạn ở đâu?

### Cách Local Search hoạt động

`LocalSearch.search()` gọi `context_builder.build_context(...)`, sau đó nhét toàn bộ context đã build vào **một prompt duy nhất** để LLM trả lời:

- `graphrag/query/structured_search/local_search/search.py`
- `graphrag/query/structured_search/local_search/mixed_context.py`

Trong `LocalSearchMixedContext.build_context(...)`, pipeline chính là:

1. Map query sang một tập entity liên quan
2. Lấy community reports, entity tables, relationship tables, text units
3. Cắt bớt mọi thứ để vừa **một cửa sổ ngữ cảnh cố định**
4. Giao cho LLM tổng hợp câu trả lời trong một lần

### Điểm giới hạn của Local Search với complex reasoning

Complex reasoning thường cần dạng suy luận:

`A -> B -> C`

hoặc

`nguyên nhân -> trung gian -> hệ quả`

hoặc

`thực thể 1 <-> quan hệ <-> thực thể 2 <-> quan hệ <-> thực thể 3`

Nhưng Local Search có 3 giới hạn cấu trúc:

1. **Flat retrieval, không có exploration nhiều bước**
   `build_context()` chọn entity và gom context quanh chúng, nhưng không có vòng lặp mở rộng theo depth để lần theo các hop quan hệ.

2. **Bị nén vào một context window cố định**
   Trong `mixed_context.py`, token budget bị chia theo tỷ lệ:
   - `text_unit_prop`
   - `community_prop`
   - phần còn lại cho local context

   Khi đến ngưỡng token, context bị dừng sớm. Với complex reasoning, chỉ thiếu một mắt xích trung gian là câu trả lời có thể vẫn "có vẻ hợp lý" nhưng bị thiếu logic hoặc sai kết luận.

3. **Quan hệ được lấy theo top-k cục bộ**
   `top_k_relationships` trong local context giúp kiểm soát chi phí, nhưng cũng làm Local Search dễ bỏ sót những quan hệ trung gian cần cho suy luận nhiều bước.

Nói ngắn gọn: **Local Search giỏi trả lời quanh một vùng entity**, nhưng không được thiết kế để chủ động truy vết một chuỗi reasoning sâu.

## 3. ToG làm gì khác về mặt thuật toán?

ToG không dừng ở việc "lấy context quanh entity". Nó có một vòng lặp tìm đường trên graph.

Các file chính:

- `graphrag/query/structured_search/tog_search/search.py`
- `graphrag/query/structured_search/tog_search/exploration.py`
- `graphrag/query/structured_search/tog_search/pruning.py`
- `graphrag/query/structured_search/tog_search/reasoning.py`
- `graphrag/query/structured_search/tog_search/state.py`

### 3.1. ToG bắt đầu từ entity, nhưng không dừng ở entity đó

Trong `search.py`, ToG:

1. Tìm **starting entities**
2. Tạo các `ExplorationNode`
3. Lặp theo `depth`
4. Với mỗi node đang ở frontier:
   - lấy tất cả quan hệ liên quan
   - chấm điểm quan hệ
   - chấm điểm entity đích
   - mở rộng sang hop tiếp theo

Đây là khác biệt cốt lõi: **ToG được thiết kế để đi tiếp trên graph**, còn Local Search chủ yếu **thu gom context quanh vùng gần query**.

### 3.2. ToG có beam search nên vừa sâu vừa có chọn lọc

Trong `state.py`, `prune_current_frontier()` giữ lại các node có score cao nhất theo `beam_width`.

Ý nghĩa:

- Không nổ số lượng path theo cấp số nhân
- Vẫn giữ được nhiều giả thuyết song song
- Có thể theo dõi nhiều chuỗi bằng chứng khác nhau trước khi chốt

Complex reasoning rất cần điều này, vì một câu hỏi thường không lộ ngay đường đúng ở hop đầu tiên.

### 3.3. ToG chấm điểm theo hai tầng: relation rồi entity

Trong `pruning.py`, ToG không chỉ hỏi "quan hệ nào có vẻ liên quan?" mà còn hỏi tiếp "entity đích nào đáng giữ lại trong path hiện tại?".

Trong `search.py`, score cuối cho một hop được tính kiểu:

`combined_score = node.score * hop_score`

với `hop_score` phụ thuộc cả:

- score của relation
- score của entity đích

Điều này rất hợp với complex reasoning vì nó giảm khả năng đi nhầm sang một nhánh nghe có vẻ liên quan ở bề mặt nhưng không giúp hoàn thành chuỗi suy luận.

### 3.4. ToG reasoning trên path, không chỉ trên context phẳng

Trong `reasoning.py`, ToG format đầu vào cho LLM thành ba lớp:

- `CHUNKS`
- `ENTITIES`
- `RELATIONSHIPS`

LLM không chỉ thấy một đống text chunk. Nó thấy:

- entity nào đã được đi qua
- quan hệ nào nối các entity đó
- path nào đã được khám phá

Vì vậy câu trả lời cuối cùng có cấu trúc reasoning rõ hơn và ít bị "đoán nối" hơn Local Search.

## 4. Vì sao ToG thắng đúng ở Context Recall và Answer Correctness?

### Context Recall tăng mạnh (`0.81 -> 0.98`)

Đây là dấu hiệu điển hình của multi-hop reasoning.

Local Search vẫn có **Context Precision rất cao** (`0.98`), nghĩa là những gì nó lấy ra đa phần đúng chủ đề. Vấn đề là nó **không lấy đủ** những mảnh context cần thiết để hoàn thành chuỗi suy luận.

ToG tăng recall vì:

- mở rộng theo nhiều hop
- lấy cả incoming và outgoing relations trong `exploration.py`
- giữ nhiều path song song bằng beam search
- thu thập text units gắn với cả entity lẫn relationship trong path

Nói cách khác, ToG ít bị bỏ sót "mắt xích trung gian".

### Answer Correctness tăng mạnh (`0.84 -> 0.95`)

Correctness tăng là hệ quả trực tiếp của recall tăng.

Với complex reasoning, câu trả lời đúng không chỉ cần "đúng chủ đề", mà phải ghép đúng chuỗi:

- sự kiện nào gây ra điều gì
- thực thể nào liên quan qua trung gian nào
- thứ tự logic giữa các mệnh đề là gì

ToG tạo được chuỗi này tốt hơn vì nó reasoning trên graph path đã được chọn lọc, thay vì để LLM tự suy diễn từ một context phẳng đã bị cắt ngắn.

## 5. Vì sao Faithfulness chỉ tăng nhẹ nhưng vẫn hợp lý?

Faithfulness tăng từ `0.82` lên `0.83`, không lớn như Correctness. Điều này hợp lý.

Lý do:

- Cả hai phương pháp đều đã khá grounded trên context retrieval
- Điểm khác biệt lớn nhất không phải "bịa ít hơn", mà là **ToG lấy được đủ bằng chứng hơn**

Nói cách khác:

- **Local Search**: thường trả lời đúng hướng nhưng dễ thiếu mắt xích
- **ToG**: vẫn grounded như vậy, nhưng thêm được các path quan trọng nên kết luận cuối chính xác hơn

## 6. Ví dụ loại câu hỏi mà ToG có lợi thế

Một câu hỏi complex reasoning trong tập dữ liệu có dạng:

> Why is detecting either the Philadelphia chromosome or BCR::ABL1 gene sufficient for diagnosing CML?

Để trả lời tốt, hệ thống phải nối được chuỗi kiểu:

1. Philadelphia chromosome và BCR::ABL1 cùng bắt nguồn từ một translocation
2. Cả hai đều là biomarkers của CML
3. Vì cùng quy về cùng cơ chế bệnh sinh nên phát hiện một trong hai là đủ cho chẩn đoán

Đây không phải retrieval một mẩu đơn lẻ. Đây là **ghép nhiều fact thành một chain**.

ToG hợp với kiểu câu hỏi này hơn vì nó được thiết kế để lần theo các quan hệ đó từng bước.

## 7. Kết luận ngắn gọn

ToG tốt hơn Local Search ở complex reasoning không phải vì prompt "hay hơn" một chút, mà vì **kiến trúc truy xuất và suy luận khác bản chất**:

- **Local Search**: one-shot, context phẳng, giới hạn bởi token window, mạnh ở truy vấn entity-centric
- **ToG**: iterative graph exploration, beam search, relation/entity pruning, reasoning trên path nhiều bước

Vì complex reasoning cần **đủ chuỗi bằng chứng** hơn là chỉ cần **đúng vài mẩu ngữ cảnh**, ToG đạt:

- recall cao hơn
- answer correctness cao hơn
- reasoning minh bạch hơn

Đó là lý do kết quả trong hình cho thấy ToG vượt Local Search rõ nhất chính ở nhóm **Complex Reasoning**.
