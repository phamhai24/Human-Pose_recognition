"""Thu thập dữ liệu khung xương từ webcam cho sáu hành động.

Chuyển từ notebook create_dataset/create_data.ipynb.
Chạy tại repo root: python ml/collect_data.py
CSV được ghi (nối tiếp) vào thư mục data/.
"""
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Khởi tạo thư viện MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Danh sách hành động cần thu thập
actions = ["ngoi_lam_viec", "ngoi_nga_lung", "nam_ngu", "gac_chan", "dung_day", "di_lai"]
KEY_TO_ACTION = {ord(str(i + 1)): action for i, action in enumerate(actions)}

# Số khung hình cần thu thập mỗi hành động
no_of_frames = 600
fps_limit = 60  # Giới hạn FPS


def extract_landmarks(results):
    """Trích xuất thông số khung xương từ MediaPipe."""
    if results.pose_landmarks:
        return [coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]
    return []


def save_data(data):
    """Lưu dữ liệu vào CSV; nếu file đã tồn tại thì append."""
    DATA_DIR.mkdir(exist_ok=True)
    for action, lm_list in data.items():
        if lm_list:
            filename = DATA_DIR / f"{action}.csv"
            df = pd.DataFrame(lm_list)
            if os.path.exists(filename):
                df.to_csv(filename, mode="a", header=False, index=False)
            else:
                df.to_csv(filename, index=False)
    print(f"Dữ liệu đã lưu thành công vào {DATA_DIR}!")


def main():
    cap = cv2.VideoCapture(0)
    data = {action: [] for action in actions}
    frame_count = {action: 0 for action in actions}  # Đếm số frame đã thu thập
    current_label = None  # Mặc định không ghi dữ liệu
    recording = False  # Cờ kiểm soát trạng thái ghi
    prev_time = 0

    print("Nhấn số để bắt đầu ghi hành động:")
    print("1: Ngồi làm việc | 2: Ngồi ngả lưng | 3: Nằm ngủ | 4: Gác chân | 5: Đứng dậy | 6: Đi lại | SPACE: Dừng ghi | Q: Thoát ")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Giới hạn tốc độ FPS
        curr_time = time.time()
        if curr_time - prev_time < 1 / fps_limit:
            continue
        prev_time = curr_time

        # Xử lý ảnh với MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Nếu đang ghi, trích xuất dữ liệu
        if recording and current_label is not None:
            lm = extract_landmarks(results)
            if lm:
                data[current_label].append(lm)
                frame_count[current_label] += 1
                print(f"Đang ghi '{current_label.upper()}': {frame_count[current_label]}/{no_of_frames} frames")

                cv2.putText(frame, f"Recording: {current_label.upper()} ({frame_count[current_label]}/{no_of_frames})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Nếu đủ số frame thì tự động dừng ghi
                if frame_count[current_label] >= no_of_frames:
                    print(f"Đã thu thập đủ {no_of_frames} frames cho '{current_label.upper()}'!")
                    recording = False

        # Vẽ khung xương lên ảnh
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4))

        # Hiển thị thông tin hành động hiện tại
        if current_label:
            cv2.putText(frame, f"Label: {current_label.upper()}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Pose Detection", frame)

        # Chuyển đổi hành động khi nhấn phím
        key = cv2.waitKey(1) & 0xFF
        if key in KEY_TO_ACTION:
            current_label = KEY_TO_ACTION[key]
            recording = True
        elif key == ord(" "):  # Nhấn SPACE để dừng ghi
            recording = False
            if current_label:
                print(f"⏸ Dừng ghi hành động '{current_label.upper()}'")
        elif key == ord("q"):
            break

    save_data(data)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
