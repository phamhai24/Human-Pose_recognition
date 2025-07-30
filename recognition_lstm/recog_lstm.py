import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Load model đã train
model = tf.keras.models.load_model(r"model_weight/best_lstm_model.keras")

# Khởi tạo Pose Landmarker API
base_options = mp.tasks.BaseOptions(model_asset_path=r"recognition_lstm/pose_landmarker_heavy.task")
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=2,  # Nhận diện tối đa 2 người
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# Biến global
labels = {}
n_time_steps = 10  # Số khung hình đầu vào
lm_dict = {}
warmup_frames = 30
cap = cv2.VideoCapture(0)

# Các lớp hành động
classes = ["LAM VIEC", "NGA LUNG", "NAM NGU", "GAC CHAN", "DUNG DAY", "DI LAI"]

# Đo thời gian
prev_time = 0

# Điều kiện cảnh báo
def check_violation(label):
    return label in ["NAM NGU", "GAC CHAN"]

def detect_action(model, lm_list):
    global label
    lm_array = np.array(lm_list).reshape(1, n_time_steps, -1)  # Reshape dữ liệu đầu vào
    results = model.predict(lm_array)  # Dự đoán
    predicted_class = np.argmax(results)  # Lấy class có xác suất cao nhất
    label = classes[predicted_class]  # Gán nhãn
    return label

# Vòng lặp chính
frame_count = 0
while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # Nhận diện nhiều người từ PoseLandmarker API
    detection_result = detector.detect(mp_image)
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Vẽ khung xương cho tất cả người phát hiện
    annotated_image = np.copy(img)
    if detection_result.pose_landmarks:
        # Duyệt qua danh sách các đối tượng PoseLandmark (mỗi đối tượng là một người)
        for idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
            # Kiểm tra và vẽ khung xương cho từng người
            if pose_landmarks:  # Kiểm tra nếu pose_landmarks không rỗng
                # Create a NormalizedLandmarkList and add landmarks to it
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                landmark_list.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility
                    ) for landmark in pose_landmarks
                ])
                
                # Draw landmarks
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    landmark_list,
                    solutions.pose.POSE_CONNECTIONS
                )

                lm_list = []
                for lm in pose_landmarks:
                    lm_list.extend([lm.x, lm.y, lm.z, lm.visibility])  # Ghi cả 4 tọa độ

                # Thu thập đủ dữ liệu và đưa vào mô hình
                if len(lm_list) == 132:
                    if len(lm_dict) < n_time_steps:
                        lm_dict[len(lm_dict)] = lm_list
                    else:
                        lm_dict = {i: lm_dict[i+1] for i in range(n_time_steps - 1)}
                        lm_dict[n_time_steps - 1] = lm_list

                    if len(lm_dict) == n_time_steps:
                        label = detect_action(model, list(lm_dict.values()))
                        labels[idx] = label

    # Hiển thị nhãn cho các người nhận diện
    for idx, label in labels.items():
        pos = (10, 30 * (idx + 1))

        # Vẽ khung viền cho nhãn
        cv2.rectangle(annotated_image, (pos[0] - 5, pos[1] - 25), (pos[0] + 400, pos[1] + 5), (0, 0, 0), -1)
        cv2.putText(annotated_image, f"Person {idx}: {label}", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị cảnh báo nếu có
        if check_violation(label):
            cv2.putText(annotated_image, " - VI PHAM!", (pos[0] + 250, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị FPS
    cv2.putText(annotated_image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hiển thị ảnh cuối cùng
    cv2.imshow("Pose Classification", annotated_image)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()