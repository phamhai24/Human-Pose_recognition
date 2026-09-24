# Pose Studio

Demo nhận diện hành động với **React/TypeScript (FE)** và **FastAPI + MediaPipe + LSTM (BE)**. Dùng model có sẵn, không huấn luyện lại.

![Pose Studio trên desktop](docs/demo-desktop.png)

## Cấu trúc thư mục

```text
be/                 Backend FastAPI: nạp model, tracking, WebSocket, tests
fe/                 Frontend React + TypeScript + Vite, unit test và Playwright E2E
scripts/            Script PowerShell khởi động BE/FE
model_weight/       Trọng số LSTM (best_lstm_model.keras)
recognition_lstm/   MediaPipe pose_landmarker_heavy.task và script nhận diện gốc
workplace_dataset/  Dữ liệu CSV sáu hành động
create_dataset/     Notebook tạo dữ liệu
train_model.py      Script huấn luyện gốc
docs/               Ảnh giao diện và báo cáo kiểm chứng
```

## Chạy demo trên Windows

Mở hai terminal tại thư mục dự án:

```powershell
# Terminal 1
.\scripts\start-be.ps1

# Terminal 2
.\scripts\start-fe.ps1
```

Mở **http://127.0.0.1:5173**. Đợi **Mô hình sẵn sàng**, chọn Webcam hoặc Video tải lên, bấm **Bắt đầu nhận diện**. Dừng phiên để giải phóng camera; `Ctrl+C` trong terminal để tắt server.

Nếu PowerShell chặn script, chạy trực tiếp:

```powershell
# Terminal 1, tại repo root
.\.venv\Scripts\python.exe -m uvicorn be.app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 2097152 --ws-max-queue 1

# Terminal 2
cd fe
npm.cmd run dev
```

## Cài đặt lần đầu

Cần **Python 3.11** và **Node.js 22.12+ hoặc 24 LTS**. Model lưu bằng Keras 2.15; không dùng Python 3.12 cho bộ dependencies này.

```powershell
git clone https://github.com/phamhai24/Human-Pose_recognition.git
cd Human-Pose_recognition
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r be/requirements-dev.txt
cd fe
npm.cmd ci
```

Nếu `py` không tìm thấy Python 3.11, dùng đường dẫn đầy đủ tới Python 3.11 để tạo venv. Hai asset bắt buộc:

- `model_weight/best_lstm_model.keras`
- `recognition_lstm/pose_landmarker_heavy.task`

Backend tính đường dẫn từ repo root. Asset MediaPipe có thể tải từ [kho model chính thức](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task).

## Tính năng

- Webcam hoặc video MP4/WebM cục bộ; vẽ khung xương, chọn người xem kết quả.
- Sáu hành động: ngồi làm việc, ngồi ngả lưng, nằm ngủ, gác chân, đứng dậy, đi lại.
- Xác suất từng nhãn, FPS thực tế, thời gian xử lý, tiến độ 10 frame.
- Lịch sử sau ba dự đoán cùng nhãn; lưu 500 sự kiện, hiển thị 100 gần nhất, xuất CSV toàn bộ lịch sử còn trong phiên.
- Nằm ngủ và gác chân được đánh dấu “Cần chú ý”; không đánh giá năng suất.
- Giao diện tiếng Việt responsive, font cục bộ, hướng dẫn và thư viện tư thế.

Ảnh trước khi chạy là **minh họa**, không phải kết quả model. Không tự bật camera, không ghi hình hoặc lưu video. Video được đọc trong browser, từng frame JPEG gửi tới backend local. Dừng phiên giữ lịch sử trong bộ nhớ tab; tải lại trang sẽ mất lịch sử.

## Kiến trúc và API

```text
fe/ React + TypeScript + Vite
  Camera/video → JPEG → WebSocket → kết quả + overlay + lịch sử
be/ FastAPI
  JPEG → MediaPipe (33 landmark) → tracker riêng từng người
       → 10 × 132 đặc trưng → Bidirectional LSTM → 6 xác suất
```

- `GET /api/health`: trạng thái model; `WS /api/recognize`: binary JPEG → JSON `result`/`error`.
- API docs: `http://127.0.0.1:8000/docs`.
- Tối đa hai người mỗi frame, bốn phiên; tracker và buffer riêng từng phiên.
- Client tối đa 10 frame/giây, chỉ một frame đang chờ; cạnh dài resize xuống 960 px.
- JPEG tối đa 2 MiB, ảnh tối đa 2.073.600 pixel; khóa model và giới hạn concurrency.
- Mất người, ghép không chắc chắn hoặc gián đoạn quá hai giây sẽ reset chuỗi.

Vite dev/preview proxy `/api` tới `127.0.0.1:8000`. Tùy chọn `VITE_API_URL` trong `fe/.env` đổi backend URL. BE hỗ trợ `POSE_MODEL_PATH`, `POSE_LANDMARKER_PATH`, `POSE_ALLOWED_ORIGINS` (origin ngăn cách dấu phẩy). Mặc định cho localhost/127.0.0.1 cổng 5173/4173. Bản demo chạy local, chưa có xác thực cho public deployment.

## Kiểm thử

```powershell
# Repo root
.\.venv\Scripts\python.exe -m pytest be/tests -q
.\.venv\Scripts\python.exe -m be.scripts.smoke_model

# Trong fe/
npm.cmd test
npm.cmd run build
npx.cmd playwright install chromium
npx.cmd playwright test
```

Playwright cần FE và BE đang chạy. Test video thật cần biến môi trường trỏ tới video có người:

```powershell
$env:POSE_TEST_VIDEO = 'C:\path\to\sample.mp4'
npx.cmd playwright test
```

Ảnh kiểm thử ở `fe/test-results/` (không commit). Kết quả kiểm chứng gần nhất: [docs/verification.md](docs/verification.md). Xem production build bằng `npm.cmd run preview`, mở `http://127.0.0.1:4173`.

## Giới hạn

- Độ tin cậy là xác suất model, không phải độ chính xác kiểm định.
- Smoke test chạy được sáu CSV; chuỗi đầu `di_lai.csv` được model gốc dự đoán là đứng dậy. Ứng dụng giữ kết quả thật.
- Tracking dựa trên vị trí, không nhận dạng danh tính; che khuất/giao nhau có thể reset hoặc đổi ID.
- Kết quả phụ thuộc ánh sáng, góc camera, toàn thân trong khung hình và nhịp frame.
- Video phân tích khi phát, không phải offline toàn bộ frame; xử lý chậm sẽ bỏ qua frame ở giữa.
- TensorFlow/MediaPipe cũ có thể in cảnh báo deprecation; kiểm tra health/log để phân biệt lỗi.

## Pipeline gốc

Giữ `create_dataset/`, `workplace_dataset/`, `train_model.py`, `recognition_lstm/` và trọng số. Entry point demo mới là `fe/` + `be/`. `requirements.txt` gốc dành cho pipeline cũ; dùng `be/requirements-dev.txt` cho ứng dụng mới. Train/notebook cần pandas/scikit-learn trong môi trường riêng hoặc cài bổ sung. Không ghi đè model đang dùng khi huấn luyện thử.

Tham khảo: [TensorFlow](https://www.tensorflow.org/install/source), [Vite](https://vite.dev/guide/), [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
