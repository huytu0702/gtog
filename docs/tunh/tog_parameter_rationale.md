# Lý do lựa chọn các thông số cấu hình ToG Search

## 4.x. Cấu hình thực nghiệm của thuật toán ToG Search

Bảng 4.1 trình bày các thông số chính của thuật toán ToG Search được sử dụng trong hệ thống. Việc lựa chọn từng thông số đều dựa trên sự cân nhắc giữa chất lượng câu trả lời và chi phí tính toán, đồng thời có tham khảo đề xuất từ bài báo gốc ToG (Sun et al., ICLR 2024).

---

### `width = 3` — Số nhánh tối đa giữ lại ở mỗi mức

Tham số `width` kiểm soát số lượng quan hệ được giữ lại tại mỗi nút trong quá trình duyệt đồ thị tri thức theo chiều rộng (beam search). Sau khi LLM chấm điểm toàn bộ các quan hệ lân cận của một thực thể, chỉ `width` quan hệ có điểm cao nhất được mở rộng sang bước tiếp theo.

Giá trị `width = 3` được chọn nhằm duy trì sự đa dạng đủ để bao phủ các hướng lý luận khác nhau, trong khi vẫn hạn chế số lượng đường khám phá bùng nổ theo cấp số nhân khi tăng độ sâu. Với `width = 1`, thuật toán suy thoái về tìm kiếm tham lam (greedy search), dễ bị kẹt ở nhánh cục bộ. Với `width ≥ 5`, số lượng nhánh tăng quá nhanh, làm tăng đáng kể số lần gọi LLM mà không cải thiện chất lượng rõ rệt. Giá trị 3 phản ánh cân bằng hợp lý được đề xuất trong nghiên cứu gốc.

---

### `depth = 3` — Độ sâu khám phá tối đa

Tham số `depth` xác định số bước nhảy tối đa (hop) mà thuật toán thực hiện trên đồ thị tri thức trước khi chuyển sang pha suy luận cuối. Mỗi bước nhảy tương ứng với việc duyệt sang một thực thể lân cận thông qua một quan hệ.

Lựa chọn `depth = 3` xuất phát từ nhận định rằng phần lớn các câu hỏi phức tạp trong miền nghiên cứu có thể được giải đáp thông qua tối đa ba bước suy luận trung gian (ví dụ: *Chủ thể → Hành động → Đối tượng → Kết quả*). Tăng độ sâu lên 4 hoặc 5 không chỉ làm tăng đáng kể số lần gọi LLM theo cấp số nhân (tỉ lệ `width^depth`), mà còn có nguy cơ đưa vào ngữ cảnh các thực thể quá xa với câu hỏi ban đầu, làm loãng chất lượng lý luận. Ngược lại, độ sâu bằng 1 hoặc 2 không đủ để khám phá các mối quan hệ nhiều bước vốn là thế mạnh của phương pháp ToG.

---

### `num_retain_entity = 5` — Số ứng viên giữ lại sau bước relation scoring

Trước khi LLM đánh giá chất lượng các thực thể lân cận, hệ thống thực hiện một bước lọc sơ bộ bằng cách lấy mẫu ngẫu nhiên `num_retain_entity` thực thể từ tập ứng viên ban đầu (nếu tập ứng viên vượt quá ngưỡng này). Bước này được triển khai trực tiếp trong mã nguồn tại `search.py` nhằm phù hợp với cơ chế lọc hai tầng của bài báo gốc ToG:

```python
if len(candidate_data) > self.num_retain_entity:
    candidate_data = random.sample(candidate_data, self.num_retain_entity)
```

Giá trị `num_retain_entity = 5` cung cấp đủ độ đa dạng cho LLM đánh giá (nhiều hơn `width = 3` để tránh trùng lặp với kết quả relation scoring), trong khi vẫn kiểm soát kích thước prompt gửi lên LLM. Nếu giá trị này quá nhỏ (ví dụ bằng `width`), bước entity scoring trở nên thừa. Nếu quá lớn, prompt sẽ phình to không cần thiết.

---

### `prune_strategy = 'llm'` — Chiến lược tỉa nhánh

Hệ thống hỗ trợ ba chiến lược tỉa nhánh: `llm`, `semantic` (độ tương đồng embedding), và `bm25` (khớp từ vựng). Chiến lược `llm` sử dụng mô hình ngôn ngữ lớn để chấm điểm mức độ liên quan của từng quan hệ và thực thể so với câu hỏi gốc.

Chiến lược `llm` được lựa chọn vì nó có khả năng nắm bắt ngữ nghĩa sâu hơn so với `bm25` (chỉ dựa trên khớp từ vựng) và linh hoạt hơn so với `semantic` (phụ thuộc vào chất lượng biểu diễn embedding của mô hình). Đặc biệt, trong bài toán trả lời câu hỏi trên đồ thị tri thức, sự liên quan của một quan hệ thường phụ thuộc vào ngữ cảnh câu hỏi theo cách mà chỉ LLM mới có thể đánh giá chính xác. Đây là lựa chọn nhất quán với thiết kế gốc của ToG.

---

### `max_relations_for_llm = 10` — Số quan hệ tối đa gửi vào LLM mỗi lần chấm điểm

Tham số này giới hạn số lượng quan hệ được đưa vào một lần gọi LLM trong bước relation scoring. Khi một thực thể có hàng chục quan hệ trong đồ thị, việc gửi tất cả cùng lúc sẽ làm prompt quá dài và tăng chi phí token.

Giá trị `max_relations_for_llm = 10` được xác định dựa trên thực nghiệm: đây là ngưỡng đủ lớn để LLM có cái nhìn tổng quan về các quan hệ liên quan nhất, đồng thời đủ nhỏ để prompt vẫn nằm trong giới hạn ngữ cảnh hiệu quả của mô hình. Các quan hệ có trọng số (weight) thấp hơn sẽ bị loại bỏ trước khi gửi lên LLM, giúp tập trung vào các kết nối có ý nghĩa hơn trong đồ thị.

---

### `temperature_exploration = 0.4` — Nhiệt độ ở pha khám phá

Tham số nhiệt độ `temperature` trong pha khám phá ảnh hưởng đến mức độ đa dạng trong phản hồi chấm điểm của LLM khi đánh giá các quan hệ và thực thể.

Giá trị `temperature_exploration = 0.4` được chọn nhằm đạt được sự cân bằng giữa *tính nhất quán* (kết quả ổn định khi cùng đầu vào) và *tính đa dạng* (LLM không bị thiên kiến quá mạnh về một hướng duy nhất). Nhiệt độ bằng 0.0 sẽ khiến LLM luôn chọn cùng một tập quan hệ trong mọi lần chạy, có thể dẫn đến bỏ sót các đường lý luận thay thế có giá trị. Nhiệt độ cao hơn 0.5 có nguy cơ tạo ra các điểm số không nhất quán, ảnh hưởng đến độ tin cậy của bước tỉa nhánh.

---

### `temperature_reasoning = 0.0` — Nhiệt độ ở pha reasoning cuối

Pha reasoning cuối là nơi LLM tổng hợp toàn bộ ngữ cảnh đã khám phá và đưa ra câu trả lời chính thức cho người dùng. Yêu cầu đặt ra ở đây là tính *xác định* và *nhất quán*: cùng một ngữ cảnh phải tạo ra cùng một câu trả lời.

Do đó, `temperature_reasoning = 0.0` — tương đương với chế độ tham lam (greedy decoding) — là lựa chọn phù hợp. Điều này đảm bảo câu trả lời cuối cùng mang tính khẳng định, dựa trên bằng chứng, và có thể tái hiện (reproducible), phù hợp với bối cảnh hệ thống trả lời câu hỏi học thuật và tra cứu tri thức.

---

### `max_context_tokens = 8000` — Giới hạn ngữ cảnh

Tham số này xác định số lượng token tối đa của phần ngữ cảnh được đưa vào prompt pha reasoning. Ngữ cảnh bao gồm các đường khám phá, mô tả thực thể, mô tả quan hệ, và các đoạn văn bản liên quan từ đồ thị tri thức.

Giá trị `max_context_tokens = 8000` được chọn dựa trên giới hạn cửa sổ ngữ cảnh của mô hình GPT-4o-mini được sử dụng trong hệ thống (tối đa 128k token), nhằm để lại đủ không gian cho system prompt, lịch sử hội thoại, và phần sinh câu trả lời. Giá trị này đảm bảo phần ngữ cảnh đủ phong phú để hỗ trợ lý luận, trong khi vẫn không vượt quá giới hạn thực tế của cửa sổ ngữ cảnh hiệu quả.

---

### `max_exploration_paths = 10` — Số đường khám phá tối đa duy trì

Sau mỗi bước mở rộng, hệ thống duy trì danh sách các đường khám phá (exploration paths) từ thực thể gốc đến các thực thể đã duyệt. Tham số `max_exploration_paths` giới hạn số đường tối đa được đưa vào ngữ cảnh của pha reasoning.

Giá trị `max_exploration_paths = 10` đảm bảo rằng ngữ cảnh gửi đến LLM đủ đa dạng để bao quát nhiều góc độ lý luận khác nhau, trong khi không vượt quá giới hạn `max_context_tokens`. Trên thực tế, với `width = 3` và `depth = 3`, số đường khám phá tiềm năng tối đa là 3³ = 27, nhưng không phải tất cả đều được mở rộng đầy đủ do cơ chế tỉa nhánh. Ngưỡng 10 phản ánh số lượng đường đủ để phủ các bằng chứng quan trọng trong phần lớn các câu hỏi thực tế.

---

## Tổng kết

Các thông số trên được lựa chọn nhằm đạt được ba mục tiêu đồng thời: (1) đảm bảo chất lượng lý luận đa bước trên đồ thị tri thức, (2) kiểm soát chi phí gọi LLM trong điều kiện triển khai thực tế, và (3) duy trì tính tái hiện và ổn định của hệ thống. Các giá trị này có thể được điều chỉnh tùy theo yêu cầu cụ thể của tập dữ liệu và mô hình ngôn ngữ được sử dụng.
