import {
  useEffect,
  useRef,
  useState,
  type RefObject,
  type CSSProperties,
} from "react";
import { Camera, Expand, ScanLine, Video } from "lucide-react";
import type { FrameResult, Status } from "../types";
import { drawPeople } from "../pose";
import { StudioScene } from "./StudioScene";
export function CameraStage({
  videoRef,
  result,
  status,
  overlay,
  mirror,
  fps,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  result: FrameResult | null;
  status: Status;
  overlay: boolean;
  mirror: boolean;
  fps: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [ratio, setRatio] = useState(16 / 9);
  const [expanded, setExpanded] = useState(false);
  const running = status === "running";
  useEffect(() => {
    if (!canvas.current || !videoRef.current) return;
    canvas.current.width = videoRef.current.videoWidth || 960;
    canvas.current.height = videoRef.current.videoHeight || 540;
    drawPeople(canvas.current, overlay ? (result?.people ?? []) : [], mirror);
  }, [result, overlay, mirror, videoRef]);
  useEffect(() => {
    if (!expanded) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [expanded]);
  return (
    <div className={`stage-shell ${expanded ? "expanded" : ""}`} ref={stage}>
      <div
        className={`camera-stage ${running ? "live-stage" : ""}`}
        style={
          {
            aspectRatio: running ? ratio : 800 / 430,
            "--stage-ratio": running ? ratio : 800 / 430,
          } as CSSProperties
        }
      >
        <video
          ref={videoRef}
          muted
          playsInline
          className={`${running ? "visible" : ""} ${mirror ? "mirrored" : ""}`}
          onLoadedMetadata={() => {
            if (videoRef.current?.videoWidth)
              setRatio(
                videoRef.current.videoWidth / videoRef.current.videoHeight,
              );
          }}
        />
        <canvas ref={canvas} className={running ? "visible" : ""} />
        {!running && <StudioScene />}
        <div className="stage-top">
          <span className={`stage-tag ${running ? "live" : ""}`}>
            <span />
            {running
              ? "Đang nhận diện"
              : status === "connecting"
                ? "Đang kết nối…"
                : "Không gian xem trước"}
          </span>
          <span className="stage-source">
            {mirror ? <Camera size={14} /> : <Video size={14} />}{" "}
            {mirror ? "Webcam" : "Video"}
          </span>
        </div>
        {!running && (
          <div className="illustration-caption">
            <ScanLine size={14} />
            Minh họa tư thế · Camera chưa bật
          </div>
        )}
        <div className="stage-bottom">
          <span>
            {running
              ? `${result?.people.length ?? 0} người trong khung hình`
              : "Sẵn sàng khám phá chuyển động"}
          </span>
          <div>
            <span className="fps-indicator">
              {running ? fps.toFixed(1) : "—"} FPS
            </span>
            <button
              className="stage-expand"
              title={expanded ? "Thu nhỏ (Esc)" : "Mở rộng khung hình"}
              aria-label={
                expanded ? "Thu nhỏ khung hình" : "Mở rộng khung hình"
              }
              onClick={() => setExpanded(!expanded)}
            >
              <Expand size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
