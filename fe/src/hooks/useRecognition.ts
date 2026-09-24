import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, socketUrl } from "../api";
import { collectEvents, stablePeople, type Transition } from "../history";
import type { ActivityEvent, FrameResult, Status } from "../types";

export function useRecognition() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const generation = useRef(0);
  const resources = useRef<{
    socket?: WebSocket;
    stream?: MediaStream;
    url?: string;
    interval?: number;
    timeout?: number;
  }>({});
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState<FrameResult | null>(null);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [fps, setFps] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [notice, setNotice] = useState("");
  const transitions = useRef(new Map<number, Transition>());

  const release = useCallback(() => {
    generation.current++;
    const r = resources.current;
    window.clearInterval(r.interval);
    window.clearTimeout(r.timeout);
    if (r.socket) {
      r.socket.onopen =
        r.socket.onmessage =
        r.socket.onclose =
        r.socket.onerror =
          null;
      r.socket.close();
    }
    r.stream?.getTracks().forEach((t) => t.stop());
    if (r.url) URL.revokeObjectURL(r.url);
    const video = videoRef.current;
    if (video) {
      video.onended = video.onerror = null;
      if (video.srcObject || video.getAttribute("src")) video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
    }
    resources.current = {};
    transitions.current.clear();
  }, []);

  const stop = useCallback(() => {
    release();
    setStatus("idle");
    setResult(null);
    setFps(0);
    setError("");
  }, [release]);

  const start = useCallback(
    async (source: "camera" | File) => {
      release();
      const token = generation.current;
      const current = () => token === generation.current;
      const fail = (message: string) => {
        if (!current()) return;
        release();
        setResult(null);
        setFps(0);
        setError(message);
        setStatus("error");
      };
      setError("");
      setNotice("");
      setResult(null);
      setElapsed(0);
      setFps(0);
      setStatus("connecting");
      const video = videoRef.current;
      if (!video) {
        fail("Không mở được vùng phát video. Hãy tải lại trang.");
        return;
      }
      try {
        const health = await getHealth();
        if (!current()) return;
        if (health.status !== "ready") throw new Error(health.message);
        if (source === "camera") {
          if (!navigator.mediaDevices?.getUserMedia)
            throw new Error(
              "Trình duyệt chưa hỗ trợ camera. Hãy mở ứng dụng bằng localhost hoặc HTTPS.",
            );
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 960 }, height: { ideal: 540 } },
            audio: false,
          });
          if (!current()) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }
          resources.current.stream = stream;
          video.srcObject = stream;
        } else {
          const url = URL.createObjectURL(source);
          resources.current.url = url;
          video.src = url;
        }
        resources.current.timeout = window.setTimeout(
          () =>
            fail("Nguồn video không phản hồi. Hãy thử camera hoặc video khác."),
          15000,
        );
        await video.play();
        if (!current()) return;
        clearTimeout(resources.current.timeout);
        const socket = new WebSocket(socketUrl());
        resources.current.socket = socket;
        resources.current.timeout = window.setTimeout(
          () => fail("Kết nối quá lâu. Kiểm tra backend rồi thử lại."),
          15000,
        );
        socket.onerror = () =>
          fail("Không kết nối được backend. Hãy kiểm tra server rồi thử lại.");
        socket.onclose = () =>
          fail("Kết nối đã đóng. Bấm bắt đầu để mở phiên mới.");
        video.onerror = () =>
          fail(
            "Không đọc được video. Hãy chọn MP4 hoặc WebM tương thích trình duyệt.",
          );
        video.onended = () => {
          if (current()) {
            stop();
            setNotice(
              "Video đã kết thúc. Lịch sử nhận diện được giữ lại bên dưới.",
            );
          }
        };
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Trình duyệt không hỗ trợ xử lý ảnh.");
        let pending = false,
          previousVideoTime = -1,
          count = 0,
          measureAt = performance.now(),
          startedAt = performance.now();
        socket.onopen = () => {
          if (!current()) return;
          clearTimeout(resources.current.timeout);
          setStatus("running");
          startedAt = performance.now();
          measureAt = startedAt;
          resources.current.interval = window.setInterval(() => {
            if (!current()) return;
            setElapsed(Math.floor((performance.now() - startedAt) / 1000));
            if (
              pending ||
              video.paused ||
              video.readyState < 2 ||
              !video.videoWidth ||
              video.currentTime === previousVideoTime
            )
              return;
            previousVideoTime = video.currentTime;
            const scale = Math.min(
              1,
              960 / Math.max(video.videoWidth, video.videoHeight),
            );
            canvas.width = Math.round(video.videoWidth * scale);
            canvas.height = Math.round(video.videoHeight * scale);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            pending = true;
            resources.current.timeout = window.setTimeout(
              () =>
                fail(
                  "Xử lý frame quá lâu. Hãy dừng các phiên khác và thử lại.",
                ),
              15000,
            );
            canvas.toBlob(
              (blob) => {
                if (!current()) return;
                if (!blob || socket.readyState !== WebSocket.OPEN) {
                  fail("Không gửi được frame. Hãy thử lại.");
                  return;
                }
                socket.send(blob);
              },
              "image/jpeg",
              0.8,
            );
          }, 100);
        };
        socket.onmessage = (event) => {
          if (!current()) return;
          clearTimeout(resources.current.timeout);
          pending = false;
          try {
            const data = JSON.parse(event.data);
            if (data.type === "error") {
              fail(data.message);
              return;
            }
            if (data.type !== "result" || !Array.isArray(data.people))
              throw new Error("Invalid response");
            const frame = data as FrameResult;
            const additions = collectEvents(
              frame.people,
              transitions.current,
              Date.now(),
            );
            setResult({
              ...frame,
              people: stablePeople(frame.people, transitions.current),
            });
            if (additions.length)
              setEvents((old) => [...additions, ...old].slice(0, 500));
            count++;
            const now = performance.now();
            if (now - measureAt >= 1000) {
              setFps((count * 1000) / (now - measureAt));
              count = 0;
              measureAt = now;
            }
          } catch {
            fail("Backend trả dữ liệu không hợp lệ. Hãy khởi động lại phiên.");
          }
        };
      } catch (e) {
        const name = e instanceof DOMException ? e.name : "";
        fail(
          name === "NotAllowedError"
            ? "Chưa có quyền truy cập camera. Hãy cho phép camera trong trình duyệt hoặc chọn video."
            : name === "NotFoundError"
              ? "Không tìm thấy camera. Hãy kết nối camera hoặc chọn video."
              : name === "NotReadableError"
                ? "Camera đang được ứng dụng khác sử dụng. Hãy đóng ứng dụng đó và thử lại."
                : e instanceof Error
                  ? e.message === "Failed to fetch"
                    ? "Chưa kết nối được backend. Hãy chạy server rồi thử lại."
                    : e.message
                  : "Không mở được nguồn video. Hãy thử lại.",
        );
      }
    },
    [release, stop],
  );

  useEffect(() => release, [release]);
  const clearEvents = useCallback(() => setEvents([]), []);
  return {
    videoRef,
    status,
    error,
    result,
    events,
    fps,
    elapsed,
    notice,
    start,
    stop,
    clearEvents,
  };
}
