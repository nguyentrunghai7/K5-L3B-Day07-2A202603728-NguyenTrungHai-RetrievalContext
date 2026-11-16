# Chạy lại bài nộp Day 7

Nguyễn Trung Hải · MSSV 2A202603728 · Lớp L3B

## Chuẩn bị corpus và index

Đặt repo LAB06 cạnh LAB07 hoặc truyền đường dẫn bằng `--lab06`. Bài dùng commit LAB06 `4cc546a63eede00fad6e32d69fe5996ea2acae44`. Nếu đã có `.lab06/chroma` và `.lab06/models` từ lần chạy Day 6, có thể dùng lại. Nếu cài mới, từ thư mục LAB06 chạy các lệnh sau. Trình nhập liệu tự lấy bản Gutenberg nếu cần và đối chiếu SHA-256 với manifest. [Repo LAB06](https://github.com/nguyentrunghai7/K5-L3B-Day06-NguyenTrungHai-2A202603728-DataIngestionEngineering/tree/4cc546a63eede00fad6e32d69fe5996ea2acae44)

```powershell
python -m venv starter/.venv
& starter/.venv/Scripts/python -m pip install -r starter/requirements-lab.txt
$env:HF_HUB_DISABLE_XET = '1'
& starter/.venv/Scripts/python -m starter.src.run
```

## Khởi động model trả lời

Tải llama.cpp Windows build `b12359` và model `Qwen2.5-3B-Instruct-Q4_K_M.gguf` ở revision `f302c64a2269a69fb27b2f9473b362f5bb8e78d8`. Giải nén llama.cpp vào `.lab07/vulkan/`, lưu model thành `.lab07/qwen.gguf`. Dùng bản CPU và `--n-gpu-layers 0` nếu máy thiếu GPU hỗ trợ Vulkan. Model, index và cache nằm ngoài Git. [llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b12359) · [Model GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/tree/f302c64a2269a69fb27b2f9473b362f5bb8e78d8)

SHA-256 của GGUF đã dùng là `9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94`. Dùng `llama-server.exe --list-devices` để xác định ID GPU. Với bản CPU, bỏ tham số `--device`.

```powershell
.lab07/vulkan/llama-server.exe -m .lab07/qwen.gguf --host 127.0.0.1 --port 8087 --ctx-size 4096 --parallel 1 --device Vulkan0 --n-gpu-layers 99
```

## Benchmark và chấm

Từ thư mục LAB07, mở terminal thứ hai. Dùng Python của LAB06 để giữ nguyên phiên bản Chroma và FastEmbed. Đặt biến môi trường offline sau khi cache model embedding đã có đủ. Chỉ định thư mục đầu ra mới bằng `--out` để giữ trace đã nộp. Vì nhãn phụ thuộc từng câu trả lời sinh ra, cần kiểm tra lại chúng khi chạy lại model.

```powershell
python -m unittest discover -s tests -v
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$labPython = '../LAB06/starter/.venv/Scripts/python.exe'
& $labPython src/benchmark.py --split dev --configs baseline dense hybrid hybrid-wide --out .lab07/replay
```

Đọc báo cáo để xem cấu hình đã chọn cho test. Dùng `--split test --configs baseline <cấu hình đã chọn>` với cùng lệnh trên. Trong trace, `original_top3` là ranking với query gốc, `tool_calls` ghi query do agent gửi, `model_calls` giữ số token và văn bản model trả về. Chạy `python src/review_metrics.py results/test/baseline` để tổng hợp trace và nhãn đã lưu.

Xem cách chia, định nghĩa hit và tiêu chí chọn trong [thiết kế benchmark](data/DAY07.md). Báo cáo cá nhân nằm tại [report/individual/2A202603728_NguyenTrungHai.md](report/individual/2A202603728_NguyenTrungHai.md).
