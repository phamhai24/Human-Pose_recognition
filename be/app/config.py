import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv('POSE_MODEL_PATH', ROOT / 'model_weight/best_lstm_model.keras'))
LANDMARKER_PATH = Path(os.getenv('POSE_LANDMARKER_PATH', ROOT / 'recognition_lstm/pose_landmarker_heavy.task'))
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 1920 * 1080
ALLOWED_ORIGINS = set(os.getenv('POSE_ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173').split(','))
CLASSES = ['Ngồi làm việc', 'Ngồi ngả lưng', 'Nằm ngủ', 'Gác chân', 'Đứng dậy', 'Đi lại']
