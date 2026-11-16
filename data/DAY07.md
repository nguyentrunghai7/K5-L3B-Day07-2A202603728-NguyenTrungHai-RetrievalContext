# Dữ liệu và cách chia

Dùng *Emma* của Jane Austen, Project Gutenberg #158, theo snapshot và manifest đã chốt trong LAB06. Kho Chroma A có 1.971 chunk, dùng cửa sổ 500 ký tự và overlap 50. Trường `page` là số chương liên tục từ 1 đến 55, không phải số trang in. Metadata `volume`, `chapter`, `start`, `end` chỉ vị trí trong bản gốc. Nguồn ghi tác phẩm thuộc phạm vi công cộng tại Hoa Kỳ. Tải từ trang nguồn và đọc thông báo giấy phép Project Gutenberg để tái tạo. [Trang nguồn](https://www.gutenberg.org/ebooks/158)

Tập dev dùng 15 câu có đáp án Q01–Q15 từ LAB06 và hai câu mới D16–D17. Tập test mới có năm câu có đáp án T01–T05 và hai câu thiếu bằng chứng T06–T07. Gold nằm trong `questions.json` và chỉ được đọc để xác minh, đối chiếu sau truy xuất và chấm. Các filter từ bộ câu LAB06 được bỏ khi chuyển vì mọi phép chạy LAB07 tìm trên cùng toàn bộ corpus.

## Thiết kế chốt trước khi chạy

Baseline là TF-IDF. Hai cấu hình thử cải thiện là dense dùng BGE để tìm câu diễn đạt khác nguyên văn, và hybrid để kết hợp khớp từ với tương đồng ngữ nghĩa. Thử thêm hybrid-wide với kỳ vọng giữ các bằng chứng nằm ngoài ba kết quả đầu. Hybrid dùng Reciprocal Rank Fusion với hằng số 60 và tối đa 30 kết quả từ mỗi phương pháp. Dùng cùng model trả lời và giới hạn hai lượt tìm cho mọi cấu hình.

Trên dev, ưu tiên số câu có đáp án đạt mức grounded answer 2, rồi số câu thiếu evidence từ chối đúng, Context Evidence Rate, Evidence Hit@3, Hit@1 và thời gian trung vị thấp hơn. Lỗi hệ thống không được tính như một lần từ chối đúng. Sau khi chốt, chạy baseline và cấu hình được chọn trên test.

Chốt từ dev: **hybrid-wide**, RRF, 6 kết quả mỗi lượt tìm, ngân sách tổng 3.600 ký tự, tối đa 2 lượt. Dense và hybrid-wide cùng 9/15 câu mức 2 và 2/2 câu từ chối đúng. Hybrid-wide đạt Context Evidence Rate 11/15, dense đạt 10/15. Cặp so sánh trên test là baseline và hybrid-wide.

Hit tự động yêu cầu đúng nguồn, đúng chương và đoạn evidence chuẩn nằm trọn trong chunk. Một câu có thể có bằng chứng tương đương khác nên đánh giá thủ công xem cả context và từng citation. Ngữ cảnh đạt 0 nếu không có bằng chứng, 1 nếu có thông tin liên quan nhưng thiếu điều kiện hoặc nguồn, 2 nếu có đủ bằng chứng và nguồn. Điểm grounded answer xét nội dung trả lời và bằng chứng trong context. Citation được chấm riêng, dựa vào ID mà model thực sự chọn và nội dung của đoạn tương ứng.

Trong lần chạy thử trên dev, model đôi khi chọn đúng ID nhưng quên đặt ID trong văn bản trả lời, khiến wrapper loại cả câu. Bản dùng cho bảng so sánh lấy danh sách ID do model chọn và hiển thị thành citation, sau đó kiểm tra ID có trong context. Tất cả cấu hình dùng cùng bản này.
