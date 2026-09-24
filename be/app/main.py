import asyncio
from contextlib import asynccontextmanager
import io
import logging
import time

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .config import ALLOWED_ORIGINS, MAX_FRAME_BYTES, MAX_PIXELS
from .inference import ModelRuntime
from .schemas import Health, error
from .tracking import SessionTracker

log = logging.getLogger('pose-studio')


class InvalidFrame(ValueError):
    pass


def decode_frame(data):
    try:
        with Image.open(io.BytesIO(data)) as header:
            if header.format != 'JPEG' or header.width * header.height > MAX_PIXELS:
                raise InvalidFrame('Ảnh phải là JPEG, tối đa 1920 × 1080 pixel.')
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidFrame('Không đọc được ảnh JPEG.')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidFrame('Không đọc được ảnh JPEG.') from exc


def process_frame(runtime, tracker, data, frame_id):
    started = time.perf_counter()
    rgb = decode_frame(data)
    tracks = tracker.update(runtime.detect(rgb))
    people = []
    for track in tracks:
        if len(track.frames) == 10:
            track.probabilities = runtime.predict(list(track.frames))
            track.label_index = int(np.argmax(track.probabilities))
        people.append(dict(id=track.id, landmarks=track.landmarks, bufferCount=len(track.frames),
                           labelIndex=track.label_index, probabilities=track.probabilities))
    return dict(type='result', frameId=frame_id,
                processingMs=round((time.perf_counter() - started) * 1000, 1), people=people)


def create_app(runtime_factory=ModelRuntime):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = None
        app.state.message = 'Model chưa sẵn sàng.'
        app.state.inference_gate = asyncio.Semaphore(1)
        app.state.sessions = 0
        try:
            app.state.runtime = await run_in_threadpool(runtime_factory)
            app.state.message = 'Model và bộ nhận diện tư thế đã sẵn sàng.'
        except Exception:
            log.exception('Model initialization failed')
            app.state.message = 'Không nạp được model. Kiểm tra hai file model và môi trường Python theo README.'
        yield
        if app.state.runtime is not None:
            await run_in_threadpool(app.state.runtime.close)

    app = FastAPI(title='Pose Studio', version='1.0.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS), allow_methods=['GET'])

    @app.get('/api/health', response_model=Health)
    async def health():
        return Health(status='ready' if app.state.runtime else 'unavailable', message=app.state.message)

    @app.websocket('/api/recognize')
    async def recognize(ws: WebSocket):
        origin = ws.headers.get('origin')
        if origin and origin not in ALLOWED_ORIGINS:
            await ws.close(code=1008)
            return
        await ws.accept()
        if app.state.runtime is None:
            await ws.send_json(error('model_unavailable', app.state.message))
            await ws.close(code=1013)
            return
        if app.state.sessions >= 4:
            await ws.send_json(error('server_busy', 'Đã đủ phiên demo. Hãy đóng một phiên rồi thử lại.'))
            await ws.close(code=1013)
            return
        app.state.sessions += 1
        tracker = SessionTracker()
        frame_id = 0
        last_frame_at = None
        try:
            while True:
                message = await ws.receive()
                if message['type'] == 'websocket.disconnect':
                    break
                data = message.get('bytes')
                if data is None or not data:
                    await ws.send_json(error('invalid_frame', 'Hãy gửi một frame JPEG dạng binary.'))
                    continue
                if len(data) > MAX_FRAME_BYTES:
                    await ws.send_json(error('frame_too_large', 'Frame vượt giới hạn 2 MiB.'))
                    continue
                now = time.monotonic()
                if last_frame_at is not None and now - last_frame_at > 2:
                    tracker.reset()
                last_frame_at = now
                frame_id += 1
                try:
                    async with app.state.inference_gate:
                        result = await run_in_threadpool(process_frame, app.state.runtime, tracker, data, frame_id)
                    await ws.send_json(result)
                except InvalidFrame as exc:
                    tracker.reset()
                    await ws.send_json(error('invalid_frame', str(exc)))
                except Exception:
                    tracker.reset()
                    log.exception('Frame inference failed')
                    await ws.send_json(error('inference_failed', 'Không xử lý được frame. Hãy dừng và thử lại.'))
        except WebSocketDisconnect:
            pass
        finally:
            tracker.reset()
            app.state.sessions -= 1

    return app


app = create_app()
