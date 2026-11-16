# Báo cáo cá nhân Day 7 — Agentic RAG trên *Emma*

Nguyễn Trung Hải · MSSV 2A202603728 · Lớp L3B

## Dữ liệu và benchmark

Corpus là *Emma* của Jane Austen, Project Gutenberg #158, gồm 55 chương và 878.458 ký tự sau chuẩn hóa. Mỗi chương là một đơn vị nguồn. Kho A của LAB06 chia văn bản theo cửa sổ 500 ký tự và overlap 50, tổng cộng 1.971 chunk. Bài làm giữ nguyên ID, văn bản, nguồn, số chương và offset từ LAB06. Gutenberg ghi tác phẩm thuộc phạm vi công cộng ở Hoa Kỳ. [Nguồn tác phẩm](https://www.gutenberg.org/ebooks/158) · [Dữ liệu và cách chia](../../data/DAY07.md)

Dense dùng Chroma 1.5.9 với HNSW cosine và FastEmbed 0.9.1. Embedding là BAAI/bge-small-en-v1.5, bản ONNX lượng tử hóa của Qdrant revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`, 384 chiều. Agent dùng Qwen2.5-3B-Instruct Q4_K_M qua llama.cpp b12359, chạy cục bộ trên NVIDIA RTX 4060 Laptop bằng Vulkan. [Thông số và lệnh tái lập](../../RUN_DAY07.md)

Dev có 15 câu có đáp án từ LAB06 và 2 câu mới thiếu bằng chứng. Test là 7 câu mới, gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Tất cả 20 đoạn gold đã được xác nhận trên bản gốc và nằm trọn trong ít nhất một chunk đúng chương. Với bốn câu thiếu bằng chứng, dò từ khóa trên đủ 55 chương và kiểm tra lại các cảnh liên quan. Agent chỉ nhận câu hỏi, lịch sử tìm kiếm và context, còn đáp án chuẩn được giữ ở bộ chấm. [Bộ câu hỏi và đoạn chuẩn](../../data/questions.json)

## Agent và context

Model phát một action JSON chọn tìm kiếm, trả lời hoặc từ chối. Tool nhận query tiếng Anh, tìm trên toàn bộ corpus và ghi kết quả có nguồn, chương, ID và điểm. Prompt yêu cầu bắt đầu bằng query gốc và cho phép viết lại khi cần. Benchmark lưu riêng ranking của query gốc ngay cả khi model thực tế viết lại câu hỏi. Mỗi lượt thêm kết quả theo thứ tự xếp hạng, bỏ ID trùng và bỏ các đoạn nguyên khối không vừa ngân sách ký tự. Context giữ ưu tiên cho các đoạn đã lấy ở lượt trước. Agent dừng khi có câu trả lời hoặc từ chối, hoặc chạm giới hạn hai lượt tìm. Wrapper định dạng danh sách citation model chọn và chỉ chấp nhận ID có trong context. Lỗi provider, truy xuất hoặc định dạng được ghi vào trace. [Mã benchmark](../../src/benchmark.py) · [Vòng lặp agent](../../starter/agentic.py)

Giới hạn hai lượt là chung cho mọi cấu hình. Các lần chạy chính thức chỉ quan sát được 0 hoặc 1 lượt tìm cho mỗi câu. Với D17 và T06, model chọn từ chối ngay. Các lần này vẫn được tính vào mẫu số tool use. Khi model từ chối bằng văn bản nhưng đính kèm ID, audit chấm hành vi từ chối theo nội dung và chấm citation riêng. Ví dụ, D16 của dense từ chối đoán serial, nhưng đoạn về chiếc piano khác không chứng minh sự thiếu dữ kiện.

## Kết quả retrieval và context

Ba phương pháp là TF-IDF cosine, dense cosine và hybrid RRF. Hybrid hợp nhất tối đa 30 ứng viên từ mỗi retriever, tính mỗi đóng góp là `1 / (60 + rank)`. Không áp dụng metadata filter. Mọi hàng báo Hit@1 và Hit@3 của query gốc, dùng cùng cutoff. Context Evidence Rate so bằng chứng chuẩn với context ở lần gọi model cuối. Q09 có bằng chứng thay thế ở đầu chương 13, được tính khi đọc thủ công dù tiêu chí khớp gold tự động không ghi hit.

| Cấu hình | Top-k / ngân sách ký tự | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Retrieval gốc / toàn lượt, trung vị | Token vào / ra |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline TF-IDF | 3 / 1.800 | dev | 3/15 (20,0%) | 6/15 (40,0%) | 5/15 (33,3%) | 6,8 ms / 1,13 s | 16.236 / 1.372 |
| Dense | 3 / 1.800 | dev | 8/15 (53,3%) | 9/15 (60,0%) | 10/15 (66,7%) | 9,5 ms / 1,09 s | 15.876 / 1.327 |
| Hybrid | 3 / 1.800 | dev | 5/15 (33,3%) | 8/15 (53,3%) | 8/15 (53,3%) | 15,9 ms / 1,07 s | 16.050 / 1.238 |
| **Hybrid-wide** | 6 / 3.600 | dev | 5/15 (33,3%) | 8/15 (53,3%) | 11/15 (73,3%) | 15,9 ms / 1,18 s | 24.400 / 1.279 |
| Baseline TF-IDF | 3 / 1.800 | test | 1/5 (20%) | 1/5 (20%) | 1/5 (20%) | 6,9 ms / 0,93 s | 6.210 / 386 |
| **Hybrid-wide** | 6 / 3.600 | test | 1/5 (20%) | 1/5 (20%) | 1/5 (20%) | 16,4 ms / 1,14 s | 9.299 / 468 |

Tất cả cấu hình có chi phí dịch vụ **0 USD** và **0 lỗi** khi chạy, không tính điện và phần cứng. Thời gian toàn lượt tính cả truy xuất query gốc, tool và model, loại thời gian tải, khởi động model và mở index. Token là tổng được server trả về cho toàn bộ câu hỏi của mỗi hàng, gồm các vòng quyết định. [Kết quả dev](../../results/dev) · [Kết quả test](../../results/test)

## Chấm agent và citation

Điểm grounded answer 0–2 xét tính đúng, đủ và bằng chứng trong context. Citation được kiểm tra riêng theo đoạn được model chọn. Nhãn và lý do do Codex đọc câu trả lời và đối chiếu nội dung nguồn, không dùng một model judge tự động. Mỗi thư mục có `traces.jsonl`, `reviews.json`, `summary.json` và `agent-summary.json`. Người nộp có thể kiểm tra trực tiếp từng nhãn từ trace và [bộ câu hỏi](../../data/questions.json).

| Cấu hình | Split | Tool use | Câu có 0 / 1 / 2 lượt tìm | Grounded 0 / 1 / 2 · trung bình | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | dev | 16/17 | 1 / 16 / 0 | 11 / 0 / 4 · 0,53/2 | 4/21 | 15/15 | 1/2 | 1/15 |
| Dense | dev | 16/17 | 1 / 16 / 0 | 4 / 2 / 9 · 1,33/2 | 9/19 | 15/15 | 2/2 | 0/15 |
| Hybrid | dev | 16/17 | 1 / 16 / 0 | 7 / 1 / 7 · 1,00/2 | 8/18 | 15/15 | 2/2 | 0/15 |
| **Hybrid-wide** | dev | 16/17 | 1 / 16 / 0 | 5 / 1 / 9 · 1,27/2 | 8/17 | 15/15 | 2/2 | 0/15 |
| Baseline | test | 6/7 | 1 / 6 / 0 | 5 / 0 / 0 · 0,00/2 | 0/5 | 5/5 | 2/2 | 0/5 |
| **Hybrid-wide** | test | 6/7 | 1 / 6 / 0 | 4 / 0 / 1 · 0,40/2 | 1/6 | 6/6 | 1/2 | 0/5 |

Đã đọc và chấm toàn bộ 5 câu test có đáp án cho mỗi cấu hình. Phân bố chất lượng context mức 0 / 1 / 2 là **2 / 2 / 1** với baseline và **1 / 3 / 1** với hybrid-wide. Cả hai có đúng một context chứa trọn bằng chứng, ở T02. [Nhãn baseline](../../results/test/baseline/reviews.json) · [Nhãn hybrid-wide](../../results/test/hybrid-wide/reviews.json)

## Thử nghiệm trên dev và quyết định

- **TF-IDF → dense.** Giả thuyết là embedding tìm được đoạn có cùng nghĩa dù khác cách diễn đạt. Q03 dùng từ “service”. TF-IDF trả cảnh nhân vật giúp đỡ Harriet, còn dense tìm được chương 2 về việc Weston gia nhập dân quân và trả lời đúng. Hit@3 tăng từ 6/15 lên 9/15. Dense vẫn sai Q14 vì ngữ cảnh chỉ chứa suy đoán về người tặng đàn.
- **TF-IDF → hybrid.** Giả thuyết là phối hợp thứ hạng từ khóa và ngữ nghĩa giữ được cả tên riêng lẫn mô tả. Hybrid đưa bằng chứng đúng vào context ở 8/15 câu so với 5/15 của TF-IDF. Ở Q12, baseline có đoạn về Bristol trong ranking query gốc nhưng tool chạy query rút gọn và trả đoạn gọi Miss Hawkins là người Bath. Hybrid giữ đúng đoạn phân biệt nơi ở Bristol và nơi qua mùa đông là Bath. So với dense, hybrid giảm Hit@3 từ 9/15 xuống 8/15 và điểm grounded từ 1,33 xuống 1,00.
- **Hybrid → hybrid-wide.** Giả thuyết là lấy sáu đoạn và tăng ngân sách giữ được bằng chứng ngoài top 3. Q05 đưa đoạn đúng về Mr. Elton vào context, nhưng model vẫn cite đoạn chỉ nói vật liệu bức tranh. Q04 đưa đoạn nói rõ tiệc đông người và giờ muộn vào context, nhưng model chọn đoạn nói về một bữa tối khác. Context Evidence Rate tăng từ 8/15 lên 11/15. Prompt token tăng từ 16.050 lên 24.400.

Chốt **hybrid-wide** trước khi chạy test theo thứ tự ưu tiên đã công bố. Dense và hybrid-wide cùng đạt 9/15 câu mức 2 và 2/2 câu từ chối đúng. Hybrid-wide hơn một câu về tỷ lệ context có gold (11/15 so với 10/15). Dense đạt Hit@1, Hit@3 và điểm grounded trung bình cao hơn, đồng thời dùng ít token hơn. [Tiêu chí và cấu hình chốt](../../data/DAY07.md)

## Ví dụ và giới hạn

| Câu test | Hành vi quan sát được | Nhãn và căn cứ |
|---|---|---|
| T02, có đáp án | Hybrid-wide gọi tìm kiếm và trả “Until the autumn [pg158:42:0:500]”. Baseline trả “Until August [pg158:41:0:500]”. | Hybrid-wide mức 2, citation hỗ trợ câu trả lời. Baseline mức 0 vì tháng Tám nói về nhà Campbell, dù đoạn đúng về Suckling đã ở trong context. |
| T07, thiếu bằng chứng | Baseline tìm một lượt rồi từ chối. Hybrid-wide khẳng định người mẹ đã gọi ông Perry. | Baseline từ chối đúng. Hybrid-wide không nêu được phí và thêm chi tiết sai vì đoạn nguồn chỉ nói bà suýt gọi. |
| T04, lỗi truy xuất và gán nhầm nhân vật | Cả hai trả lời là bé trai và trích chương 2. | Mức 0. Đoạn trích là về con trai của ông Weston với vợ đầu. Đoạn chuẩn ở chương 53 nêu bé gái của Mrs. Weston. |

Trên test, hybrid-wide tăng số câu mức 2 từ **0/5 lên 1/5** nhưng từ chối đúng giảm từ **2/2 xuống 1/2**. Bốn câu trả lời sai của cấu hình được chọn đều thiếu bằng chứng chuẩn trong context. Số câu ít nên một câu tương ứng 20 điểm phần trăm ở các chỉ số trên câu có đáp án. Truyện có nhiều sự kiện và nhân vật lặp lại trong ba tập, làm các đoạn tương tự cạnh tranh với cảnh được hỏi. D16 cũng cho thấy model có thể đọc nhầm số trong ID thành dữ kiện. Cơ hội thử tiếp trên một bộ dev phù hợp là tách ID khỏi nội dung chứng cứ, yêu cầu model xác định mệnh đề được nguồn hỗ trợ, và đo lại khả năng tìm bổ sung khi kết quả đầu tiên không giải đáp được.

Codex hỗ trợ xây dựng chương trình, tạo câu mới, chạy benchmark, đọc trace và soạn báo cáo. Nhãn được công bố để học viên và người chấm đối chiếu. [Cách chạy lại](../../RUN_DAY07.md) · [Trace baseline](../../results/test/baseline/traces.jsonl) · [Trace hybrid-wide](../../results/test/hybrid-wide/traces.jsonl)
