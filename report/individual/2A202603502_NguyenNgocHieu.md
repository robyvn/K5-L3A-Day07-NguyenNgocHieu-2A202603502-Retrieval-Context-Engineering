# Báo cáo Day 7 — Agentic RAG

**Học viên:** Nguyễn Ngọc Hiếu  
**MSSV:** 2A202603502  
**Lớp:** L3A  
**Dự án:** Agentic Retrieval & Context Engineering  

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - **Chủ đề corpus:** Văn bản quy phạm pháp luật — **Luật Đường bộ số 35/2024/QH15** (Quốc hội khóa XV thông qua ngày 27/06/2024, công bố trên Công báo Chính phủ số 983 + 984 ngày 25/08/2024).
  - **Quyền sử dụng:** Theo Điều 19 Luật Sở hữu trí tuệ Việt Nam, văn bản quy phạm pháp luật thuộc phạm vi công cộng (**Public Domain**), được phép sử dụng, nghiên cứu và trích dẫn hoàn toàn tự do.
  - **Quy mô corpus:** 69 trang PDF chuẩn hóa, gồm **306 chunks** (kế thừa Chiến lược B từ Day 6: phân tách theo ranh giới Điều luật và đoạn văn bản kép `\n\n`, max size 800 ký tự). Dữ liệu được trích xuất ra file chuẩn `data/day6_chunks.json`.
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** **PostgreSQL 16 + pgvector 0.8.7** chạy trên container Docker local qua cổng `55433` (kết nối qua `psycopg`).
  - **Embedding model:** `google/embeddinggemma-2` (Google DeepMind, 768 dimensions, $L_2$ normalized).
  - **Metric tìm kiếm:** Cosine Similarity ($1 - \text{cosine distance} \Leftrightarrow$ toán tử `<=>` của pgvector).
- **Agent model, cách chạy và phân chia dev/test:**
  - **Model:** Google Gemini (`gemini-3.1-flash-lite`) gọi trực tiếp qua REST API (JSON Schema mode) với độ trễ thấp và độ tin cậy cao.
  - **Kiến trúc Agent:** Vòng lặp ReAct (`starter/agentic.py`), bộ điều phối `GeminiController` tự động quyết định hành động `search` hoặc `answer` kèm `citations` `[chunk-id]`.
  - **Phân chia benchmark:** Tổng cộng **24 câu hỏi** chuẩn:
    - **Tập Dev (17 câu):** 15 câu có đáp án từ Day 6 (`q01`–`q15`) + 2 câu thiếu bằng chứng (`q_dev_unans_01`, `q_dev_unans_02`).
    - **Tập Test (7 câu):** 5 câu có đáp án từ Day 6 (`q16`–`q20`) + 2 câu thiếu bằng chứng (`q_test_unans_01`, `q_test_unans_02`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án:** Bằng chứng (`gold evidence`) trích dẫn nguyên văn từng ký tự từ Điều luật và trang văn bản tương ứng. Chấm điểm theo thang `0/1/2` (2 = trả lời đúng, đủ và có citation hợp lệ; 1 = đúng một phần; 0 = sai, thiếu bằng chứng hoặc từ chối nhầm).
  - **Câu thiếu evidence:** Câu hỏi liên quan mật thiết đến chủ đề giao thông đường bộ nhưng nội dung nằm ngoài phạm vi Luật Đường bộ (như mức phạt tiền vượt đèn đỏ thuộc Nghị định 100, quy trình cấp GPLX quốc tế, phí bảo hiểm bắt buộc dân sự). Tiêu chí chấm: Agent phải chủ động từ chối (`status: abstain`), không được suy đoán hoặc bịa đặt (hallucination).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF Lexical, top-k=3, budget=800 chars | Phân tích từ vựng đơn thuần bằng TF-IDF không có embedding | dev | 9/15 (60.0%) | 11/15 (73.3%) | 9/15 (60.0%) | ~3.1 ms / 0 USD |
| **Phương pháp 2** | Dense (pgvector + embeddinggemma-2), top-k=3, budget=800 | Truy vấn không gian vector 768 chiều từ Day 6 | dev | 13/15 (86.7%) | 15/15 (100.0%) | 13/15 (86.7%) | ~25.0 ms / 0 USD |
| **Phương pháp 3** | Hybrid (Dense + TF-IDF RRF), top-k=3, budget=800 | Hợp nhất thứ hạng Reciprocal Rank Fusion ($k=60$) | dev | 12/15 (80.0%) | 14/15 (93.3%) | 12/15 (80.0%) | ~28.2 ms / 0 USD |
| **Cải thiện 1** | Dense (pgvector), top-k=3, **budget=1500 chars** | Tăng context budget để không làm cụt các Điều luật dài | dev | 13/15 (86.7%) | 15/15 (100.0%) | 14/15 (93.3%) | ~25.0 ms / 0 USD |
| **Cải thiện 2** | Dense (pgvector), top-k=3, **budget=2500 chars** | Tối đa hóa khả năng chứa toàn bộ 3 chunk trong context | dev | 13/15 (86.7%) | 15/15 (100.0%) | 15/15 (100.0%) | ~25.0 ms / 0 USD |
| **Cấu hình đã chọn** | Dense (pgvector) / Hybrid, top-k=3, budget=1500 chars | Cấu hình cân bằng tối ưu giữa độ phủ context và token budget | **test** | 3/5 (60.0%) | 4/5 (80.0%) | 4/5 (80.0%) | ~26.5 ms / 0 USD |
| **Baseline** | TF-IDF Lexical, top-k=3, budget=800 chars | Lexical baseline trên tập test độc lập | **test** | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | ~3.2 ms / 0 USD |

### Phân tích hai lỗi trên tập Dev, nguyên nhân, cách thử và lý do chọn cấu hình:

1. **Lỗi 1 (Câu `q02` - Hiện tượng Query lặp trong Multi-turn Retrieval):**
   - *Triệu chứng:* Ở câu `q02` (*"Theo Luật Đường bộ 2024, kết cấu hạ tầng đường bộ bao gồm những thành phần nào?"*), lượt search 1 Agent tìm kiếm chuỗi `"kết cấu hạ tầng đường bộ bao gồm những thành phần nào"`. Retriever trả về 3 chunk chứa từ khóa chung chung (Điều 1, Điều 21 và Phụ lục) nhưng không có chunk mục tiêu `p1_struct_002` (Điều 2 khoản 4).
   - *Nguyên nhân:* Ở lượt 2, Agent không thay đổi chiến thuật từ khóa mà tiếp tục lặp lại đúng câu truy vấn ban đầu, dẫn đến kết quả trả về trùng lặp, chạm ngưỡng `max_searches` và buộc phải từ chối (`abstain`).
   - *Biện pháp cải thiện:* Cần bổ sung cơ chế Query Reformulation / Query Rewriting (ví dụ: tự động bổ sung `"Điều 2"` hoặc `"giải thích từ ngữ"` khi kết quả trước chưa thỏa mãn).

2. **Lỗi 2 (Câu `q04` - Hạn chế của Top-K hẹp đối với các từ khóa phổ biến):**
   - *Triệu chứng:* Ở câu `q04` (*"Hành vi nào đối với kết cấu hạ tầng đường bộ bị nghiêm cấm tại khoản 1 Điều 7?"*), từ khóa "nghiêm cấm" và "kết cấu hạ tầng" xuất hiện rải rác ở rất nhiều Điều luật khác nhau (Điều 7, Điều 21, Điều 50).
   - *Nguyên nhân:* Khi đặt `top_k=3`, điểm số tương đồng ngữ nghĩa bị phân tán cho các điều luật bảo vệ khác, đẩy chunk vàng về rank 4 hoặc 5, khiến Agent không nhận được evidence trong context.
   - *Biện pháp cải thiện:* Mở rộng candidate pool lên `top_k=5` sau đó áp dụng bộ lọc số Điều luật hoặc reranker để đưa đúng Điều 7 vào context.

- **Lý do chốt cấu hình:** Chọn **Dense Retrieval kết hợp Context Budget 1500 ký tự** làm nền tảng vì đạt Evidence Hit@3 cao nhất (**100% trên Dev** và **80% trên Test**), đồng thời Context Evidence Rate đạt tới **93.3%** mà không làm tăng quá nhiều kích thước prompt gửi tới LLM.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** Agent được trang bị tool `search(query, filters, top_k)`.
  - **Số lượt gọi:** Cho phép tối đa `max_searches = 3`. Agent dừng ngay khi tìm đủ thông tin và chuyển sang action `answer`. Nếu vượt quá giới hạn mà chưa đủ bằng chứng, hệ thống trả về `abstain` an toàn (`event: search_limit`).
  - **Cách chọn context:** Áp dụng thuật toán `pack_context` với cơ chế khử trùng lặp (`deduplication` theo `chunk_id`), xếp các chunk theo thứ tự điểm số giảm dần và đóng gói ngữ cảnh trong ngân sách ký tự quy định (`budget = 1500`).
  - **Xử lý lỗi:** Bắt lỗi kết nối mạng, chuẩn hóa JSON trả về từ model; nếu citation không khớp với bất kỳ chunk nào trong context, Agent tự động chuyển về trạng thái `abstain` để chống hallucination.

### Bảng tổng hợp đánh giá Agent (kết quả từ `starter/agent_metrics.py`):

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Agent Gemini (Dev)** | 16/17 (94.1%) | 1.8 lượt/câu | 0: 6 câu \| 1: 0 câu \| 2: 9 câu (ĐTB: **1.2/2.0**) | 9/9 (**100.0%**) | 9/9 (**100.0%**) | 2/2 (**100.0%**) | 6/15 |
| **Agent Gemini (Test)** | 5/7 (71.4%) | 1.6 lượt/câu | 0: 3 câu \| 1: 0 câu \| 2: 2 câu (ĐTB: **0.8/2.0**) | 2/2 (**100.0%**) | 2/2 (**100.0%**) | 2/2 (**100.0%**) | 3/5 |

### Trích xuất minh chứng một số câu điển hình:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q01` | Có đáp án (Dev) | `search("phạm vi điều chỉnh Luật Đường bộ 2024")` $\rightarrow$ Trả về chunk `luat_duong_bo_2024_p1_struct_000` ở Rank 1 | *"Luật Đường bộ số 35/2024/QH15 quy định về hoạt động đường bộ và quản lý nhà nước về hoạt động đường bộ [luat_duong_bo_2024_p1_struct_000]."* | **2/2 điểm**. Trả lời chính xác, trích dẫn đúng nguyên văn bằng chứng. |
| `q18` | Có đáp án (Test) | `search("tạm dừng khai thác đường cao tốc là trường hợp như thế nào")` $\rightarrow$ Trả về `luat_duong_bo_2024_p46_struct_001` | *"Theo Điều 51 Luật Đường bộ 2024, tạm dừng khai thác đường cao tốc là trường hợp tạm không cho phương tiện tham gia giao thông... [luat_duong_bo_2024_p46_struct_001]"* | **2/2 điểm**. Trả lời đúng trọn vẹn ngữ nghĩa và dẫn nguồn chuẩn. |
| `q_dev_unans_01` | Thiếu evidence (Dev) | `search("mức xử phạt vượt đèn đỏ xe máy")` $\rightarrow$ Không có điều luật quy định tiền phạt vi phạm hành chính | *"Không đủ bằng chứng để trả lời."* (Citations: `[]`) | **Abstention đúng (True)**. Agent nhận diện chính xác câu hỏi nằm ngoài văn bản luật và từ chối trả lời. |
| `q_test_unans_01` | Thiếu evidence (Test) | `search("quy trình cấp giấy phép lái xe quốc tế IDP")` $\rightarrow$ Tài liệu không có quy trình chi tiết | *"Không đủ bằng chứng để trả lời."* (Citations: `[]`) | **Abstention đúng (True)**. Từ chối chuẩn xác, không bịa đặt quy trình. |
| `q02` | Có đáp án (Dev) | `search("kết cấu hạ tầng đường bộ bao gồm...")` $\rightarrow$ Bị lọt khỏi top-3 do nhiễu từ khóa | *"Không đủ bằng chứng để trả lời."* | **0/2 điểm (Từ chối nhầm)**. Do retrieval chưa lấy được chunk Điều 2, Agent trung thực từ chối thay vì bịa câu trả lời. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - File dữ liệu chunks: `data/day6_chunks.json`
  - File benchmark: `data/questions.json` và `data/agent_cases.json`
  - Trace & review Dev: `results/agent_dev/traces.json`, `results/agent_dev/reviews.json`, `results/agent_dev/summary.json`
  - Trace & review Test: `results/agent_test/traces.json`, `results/agent_test/reviews.json`, `results/agent_test/summary.json`
  - Thời gian chạy trung bình: **~2.5 - 3.2 giây/câu** (bao gồm cả độ trễ mạng gọi LLM).
  - Chi phí: **0 USD** (database chạy local, gọi Gemini API trong hạn mức miễn phí).
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - *Đánh đổi Chất lượng vs Độ trễ:* Thử nghiệm cho thấy khi tăng context budget từ 800 lên 1500 ký tự, Context Evidence Rate tăng từ 86.7% lên 93.3% mà thời gian xử lý của LLM chỉ tăng thêm khoảng 200ms.
  - *Độ tin cậy của Citation:* 100% số câu mà Agent sinh câu trả lời đều có citation hợp lệ và được bằng chứng hỗ trợ trực tiếp (`citation_validity = 100%`).
  - *Giới hạn hệ thống:* Hiện tại Agent chưa có bộ phân tích truy vấn để tự động viết lại câu hỏi (query reformulation) khi lượt tìm kiếm trước đó chưa mang lại thông tin mong muốn, dẫn tới tỷ lệ từ chối nhầm còn ở mức 40%.

---

## Kết luận

Bài lab Day 7 đã hoàn thành các yêu cầu của phần cơ bản:
1. Kết nối thành công dữ liệu 306 chunks và pgvector từ Day 6.
2. Xây dựng và đo lường độc lập 3 phương pháp retrieval (TF-IDF, Dense, Hybrid) qua các mức context budget.
3. Tích hợp Agent với LLM thật (`gemini-3.1-flash-lite`), thực thi 24 câu hỏi benchmark độc lập (Dev/Test), trích dẫn nguồn chuẩn xác 100% và nhận diện đúng các trường hợp thiếu thông tin (100% abstention đúng).

