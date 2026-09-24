import cv2
import numpy as np
from fastapi.testclient import TestClient
from be.app.main import create_app
from be.app.config import MAX_FRAME_BYTES


class Runtime:
    def detect(self, rgb):
        return [[dict(x=.3, y=.5, z=0., visibility=1.) for _ in range(33)]]

    def predict(self, sequence):
        assert np.asarray(sequence).shape == (10, 132)
        return [.8, .04, .04, .04, .04, .04]

    def close(self):
        pass


def jpeg():
    return cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))[1].tobytes()


def test_health_reports_model_failure():
    def broken():
        raise FileNotFoundError('missing')
    with TestClient(create_app(broken)) as client:
        assert client.get('/api/health').json()['status'] == 'unavailable'
        with client.websocket_connect('/api/recognize') as ws:
            assert ws.receive_json()['code'] == 'model_unavailable'


def test_bad_frames_do_not_kill_connection():
    with TestClient(create_app(Runtime)) as client:
        with client.websocket_connect('/api/recognize') as ws:
            ws.send_bytes(b'not-jpeg')
            assert ws.receive_json()['code'] == 'invalid_frame'
            ws.send_bytes(b'x' * (MAX_FRAME_BYTES + 1))
            assert ws.receive_json()['code'] == 'frame_too_large'
            ws.send_bytes(jpeg())
            assert ws.receive_json()['people'][0]['bufferCount'] == 1


def test_prediction_warms_up_and_new_session_starts_empty():
    with TestClient(create_app(Runtime)) as client:
        with client.websocket_connect('/api/recognize') as a:
            for i in range(10):
                a.send_bytes(jpeg())
                result = a.receive_json()
                assert result['people'][0]['bufferCount'] == i + 1
            assert result['people'][0]['labelIndex'] == 0
            with client.websocket_connect('/api/recognize') as b:
                b.send_bytes(jpeg())
                assert b.receive_json()['people'][0]['labelIndex'] is None


def test_inference_failure_is_recoverable():
    class BrokenRuntime(Runtime):
        def predict(self, sequence):
            raise RuntimeError('private failure')
    with TestClient(create_app(BrokenRuntime)) as client:
        with client.websocket_connect('/api/recognize') as ws:
            for _ in range(10):
                ws.send_bytes(jpeg())
                result = ws.receive_json()
            assert result['code'] == 'inference_failed'
            assert 'private' not in result['message']
            ws.send_bytes(jpeg())
            assert ws.receive_json()['people'][0]['bufferCount'] == 1


def test_disallowed_origin_is_rejected():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with TestClient(create_app(Runtime)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/api/recognize', headers={'origin': 'https://untrusted.example'}):
                pass
