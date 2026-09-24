# Bàn giao Pose Studio — 2026-09-24

Đã triển khai hai module `fe/` và `be/` dùng model gốc. Chạy theo [README](../README.md). Ảnh giao diện: [desktop](demo-desktop.png), [mobile](demo-mobile.png).

| Kiểm tra | Kết quả |
| --- | --- |
| Backend pytest | 20 passed |
| Frontend Vitest | 7 passed |
| Playwright Chromium | 6 passed, không skip trong lượt bàn giao |
| TypeScript + Vite build | Thành công |
| npm audit | 0 vulnerabilities được báo cáo |
| uv pip check | 79 packages tương thích |
| API health / frontend HTTP | ready / 200 |
| PowerShell startup scripts | Parse không lỗi; lệnh server tương ứng đã chạy |

Model smoke test đọc 10 hàng đầu mỗi CSV trong sáu dataset, xác minh output sáu xác suất hữu hạn và tổng gần 1. MediaPipe được kiểm tra với frame trống. Đây là kiểm tra vận hành, không phải đánh giá accuracy.

E2E video dùng WebM lặp [ảnh pose mẫu công khai của MediaPipe](https://storage.googleapis.com/mediapipe-assets/pose.jpg). FE đọc video, gửi JPEG qua WebSocket tới MediaPipe/LSTM thật; xác nhận kết quả, lịch sử, download CSV, dừng/giải phóng nguồn và xóa lịch sử. Đây là fixture kỹ thuật, không phải video hành động tự nhiên có ground truth.

Camera ảo 960×540 kiểm tra overlay giữ tỷ lệ 16:9 trên mobile, kể cả khi mở rộng. Các test khác kiểm tra từ chối camera, hủy trong lúc chờ quyền, backend lỗi và bố cục desktop/mobile.

## Review

Review độc lập không tìm thấy blocker backend. Hai vấn đề được nâng mức ưu tiên vì ảnh hưởng demo nhiều người và đã sửa với regression test fail rồi pass:

- Bảng tự đổi người khi detection đổi thứ tự: mặc định chọn track nhỏ nhất; giữ lựa chọn thủ công khi người đó còn xuất hiện.
- Nhãn cảnh báo nhấp nháy: chỉ đổi nhãn hiển thị sau ba dự đoán liên tiếp, dùng cùng trạng thái với lịch sử. Xác suất vẫn cập nhật từng frame.

Kiểm tra trực quan còn phát hiện tràn thư viện hành động sau khi tăng chữ và lệch tỷ lệ canvas/video trên mobile. Cả hai đã sửa, E2E fail rồi pass.

## Quyết định và giới hạn

- Làm trong checkout hiện tại trên nhánh `feat/pose-studio` để kết quả hiện trong IDE, thay vì worktree khác. Đổi lại không có checkout cách ly; giữ nguyên thay đổi notebook của người dùng.
- Dùng Python 3.11.9 có sẵn cho venv riêng. Không thay Python hệ thống, chuyển đổi hoặc train lại model.
- Dùng headless test vì browser tương tác không kết nối. Chưa thử webcam vật lý, tracking người thật khi giao nhau/che khuất, hoặc cài trên máy Windows hoàn toàn mới. Các điều kiện đó cần kiểm chứng thực tế.
- Đã commit và push lên nhánh `feat/pose-studio`; chưa merge vào `main`.
- Điểm wording nhỏ chưa sửa sau review: khi HTTP server hoạt động nhưng model không nạp được, thanh trạng thái ghi “Backend chưa kết nối”. Bấm bắt đầu hiển thị lỗi model chi tiết; API health phân biệt đúng.
- Model gốc nhầm chuỗi đầu `di_lai.csv` thành đứng dậy. Ảnh yoga E2E ngoài miền nơi làm việc cũng có nhãn không phù hợp. Không sửa kết quả để làm đẹp demo; confidence cao không bảo đảm nhãn đúng.
- TensorFlow/MediaPipe cũ và Starlette test client có cảnh báo deprecation, không làm thất bại kiểm thử.

Các giới hạn webcam, crossing, cài máy mới và accuracy mà reviewer không đánh giá được giữ rõ ở trên. Agent triển khai đã tự xem ảnh và chạy bộ kiểm tra cuối.
