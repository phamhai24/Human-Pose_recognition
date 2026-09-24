import threading
import numpy as np
from .config import MODEL_PATH, LANDMARKER_PATH


def validate_sequence(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (10, 132) or not np.isfinite(array).all():
        raise ValueError('Expected 10 frames with 132 finite features')
    return array[None, ...]


def validate_probabilities(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.shape != (6,) or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)
            or not np.isclose(array.sum(), 1., atol=1e-3)):
        raise ValueError('Model must return six valid probabilities')
    return array.tolist()


class ModelRuntime:
    def __init__(self):
        import tensorflow as tf
        import mediapipe as mp
        if not MODEL_PATH.is_file() or not LANDMARKER_PATH.is_file():
            raise FileNotFoundError('Missing LSTM model or pose landmarker asset')
        self.model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        if tuple(self.model.input_shape[1:]) != (10, 132) or self.model.output_shape[-1] != 6:
            raise ValueError('Model shape is incompatible with Pose Studio')
        self.mp = mp
        self.lock = threading.Lock()
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=2,
            )
        )

    def predict(self, sequence):
        batch = validate_sequence(sequence)
        with self.lock:
            output = self.model(batch, training=False).numpy()[0]
        return validate_probabilities(output)

    def detect(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with self.lock:
            result = self.detector.detect(image)
        return [[dict(x=p.x, y=p.y, z=p.z, visibility=p.visibility) for p in pose]
                for pose in result.pose_landmarks]

    def close(self):
        with self.lock:
            self.detector.close()
