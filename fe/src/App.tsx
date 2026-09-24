import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Camera,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  FileVideo,
  Focus,
  Info,
  LoaderCircle,
  Play,
  ScanLine,
  ShieldCheck,
  Square,
  Upload,
  Users,
  Video,
  X,
  Zap,
} from "lucide-react";
import { getHealth } from "./api";
import { useRecognition } from "./hooks/useRecognition";
import { COLORS, LABELS, type Health } from "./types";
import { Sidebar } from "./components/Sidebar";
import { CameraStage } from "./components/CameraStage";
import { PredictionPanel } from "./components/PredictionPanel";
import { EventTimeline } from "./components/EventTimeline";
import { PoseFigure } from "./components/PoseFigure";

const ACTION_DESCRIPTIONS = [
  "Tư thế ngồi trước bàn",
  "Tựa lưng về phía sau",
  "Tư thế nằm nghỉ",
  "Đặt chân lên cao",
  "Chuyển sang tư thế đứng",
  "Chuyển động bước chân",
];
function duration(seconds: number) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export default function App() {
  const recognition = useRecognition();
  const { status, result, events, error, fps, elapsed, notice } = recognition;
  const [health, setHealth] = useState<Health | null>(null);
  const [source, setSource] = useState<"camera" | "file">("camera");
  const [file, setFile] = useState<File | null>(null);
  const [overlay, setOverlay] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [activeNav, setActiveNav] = useState("studio");
  const [modal, setModal] = useState<"help" | "model" | number | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const busy = status === "connecting" || status === "running";

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const value = await getHealth();
        if (alive) setHealth(value);
      } catch {
        if (alive)
          setHealth({
            status: "unavailable",
            message: "Chưa kết nối backend. Hãy chạy server theo hướng dẫn.",
          });
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 10000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (modal !== null) dialog.current?.showModal();
    else dialog.current?.close();
  }, [modal]);

  function navigate(id: string) {
    setActiveNav(id);
    document
      .getElementById(id)
      ?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "start",
      });
  }
  function changeSource(next: "camera" | "file") {
    recognition.stop();
    setSource(next);
    setSelectedId(null);
  }
  function begin() {
    setSelectedId(null);
    if (source === "camera") void recognition.start("camera");
    else if (file) void recognition.start(file);
    else fileInput.current?.click();
  }

  return (
    <div className="app-shell">
      <a href="#studio" className="skip-link">
        Đến nội dung chính
      </a>
      <Sidebar
        active={activeNav}
        onNavigate={navigate}
        onHelp={() => setModal("help")}
        onAbout={() => setModal("model")}
      />
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Không gian làm việc</span>
            <ChevronRight size={14} />
            <strong>Phòng nhận diện</strong>
          </div>
          <div className="topbar-right">
            <span className="local-label">
              <ShieldCheck size={15} />
              Xử lý cục bộ
            </span>
            <button
              className="icon-button"
              aria-label="Hướng dẫn sử dụng"
              onClick={() => setModal("help")}
            >
              <CircleHelp size={19} />
            </button>
            <span className="avatar">CV</span>
          </div>
        </header>
        <main>
          <section id="studio" className="studio-section">
            <div className="page-heading">
              <div>
                <div className="heading-kicker">
                  <span />
                  Computer vision, in motion
                </div>
                <h1>
                  Hiểu từng chuyển động<span>.</span>
                </h1>
                <p>
                  Biến những hành động thường ngày thành thông tin, ngay trước
                  mắt bạn.
                </p>
              </div>
              <button
                className="button secondary guide-button"
                onClick={() => setModal("help")}
              >
                <Play size={15} />
                Hướng dẫn demo
                <ArrowUpRight size={15} />
              </button>
            </div>

            <div className="session-strip">
              <div className="session-intro">
                <span className="session-icon">
                  <ScanLine size={20} />
                </span>
                <div>
                  <strong>Phòng nhận diện trực tiếp</strong>
                  <span>Webcam hoặc video của bạn</span>
                </div>
              </div>
              <div className="session-status">
                <span
                  className={`connection-dot ${health?.status === "ready" ? "ready" : health ? "offline" : ""}`}
                />
                <span>
                  {health?.status === "ready"
                    ? "Mô hình sẵn sàng"
                    : health
                      ? "Backend chưa kết nối"
                      : "Đang kiểm tra mô hình"}
                </span>
                <span className="strip-divider" />
                <span className="model-tag">MediaPipe + LSTM</span>
              </div>
            </div>

            <div className="studio-grid">
              <section
                className="camera-panel panel"
                aria-label="Nguồn và khung hình nhận diện"
              >
                <div className="camera-toolbar">
                  <div className="source-tabs" aria-label="Nguồn nhận diện">
                    <button
                      aria-pressed={source === "camera"}
                      className={source === "camera" ? "selected" : ""}
                      onClick={() => changeSource("camera")}
                    >
                      <Camera size={16} />
                      Webcam
                    </button>
                    <button
                      aria-pressed={source === "file"}
                      className={source === "file" ? "selected" : ""}
                      onClick={() => changeSource("file")}
                    >
                      <FileVideo size={16} />
                      Video tải lên
                    </button>
                  </div>
                  <span className="session-time">
                    <Clock3 size={14} />
                    {duration(elapsed)}
                  </span>
                </div>
                <CameraStage
                  videoRef={recognition.videoRef}
                  result={result}
                  status={status}
                  overlay={overlay}
                  mirror={source === "camera"}
                  fps={fps}
                />
                {source === "file" && (
                  <div className="file-row">
                    <FileVideo size={18} />
                    <span>
                      {file?.name ?? "Chọn video MP4 hoặc WebM để bắt đầu"}
                    </span>
                    <button
                      className="text-button"
                      onClick={() => fileInput.current?.click()}
                      disabled={busy}
                    >
                      {file ? "Đổi video" : "Chọn video"}
                      <Upload size={14} />
                    </button>
                  </div>
                )}
                <input
                  ref={fileInput}
                  className="sr-only"
                  type="file"
                  accept="video/*"
                  aria-label="Chọn video để nhận diện"
                  onChange={(e) => {
                    const selected = e.target.files?.[0];
                    if (selected) {
                      recognition.stop();
                      setFile(selected);
                      setSource("file");
                    }
                    e.target.value = "";
                  }}
                />
                <div className="camera-controls">
                  <div className="overlay-control">
                    <button
                      className={`switch ${overlay ? "on" : ""}`}
                      role="switch"
                      aria-checked={overlay}
                      aria-label="Hiển thị khung xương"
                      onClick={() => setOverlay(!overlay)}
                    >
                      <span />
                    </button>
                    <span>Hiển thị khung xương</span>
                    <span
                      className="control-help"
                      title="Vẽ các điểm cơ thể và liên kết trên video"
                    >
                      <Info size={14} />
                    </span>
                  </div>
                  <button
                    className={`button ${busy ? "stop-button" : "primary"}`}
                    onClick={busy ? recognition.stop : begin}
                  >
                    {status === "connecting" ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : busy ? (
                      <Square size={14} />
                    ) : (
                      <Play size={16} />
                    )}{" "}
                    {status === "connecting"
                      ? "Hủy kết nối"
                      : busy
                        ? "Dừng nhận diện"
                        : "Bắt đầu nhận diện"}
                  </button>
                </div>
                {error && (
                  <div className="feedback error" role="alert">
                    <Info size={18} />
                    <span>{error}</span>
                  </div>
                )}
                {notice && !busy && (
                  <div className="feedback" role="status">
                    <Check size={18} />
                    <span>{notice}</span>
                  </div>
                )}
                <div className="camera-metrics">
                  <div>
                    <span className="metric-icon blue">
                      <Users size={18} />
                    </span>
                    <span>
                      Trong khung hình
                      <strong>
                        {result?.people.length ?? "—"} <small>người</small>
                      </strong>
                    </span>
                  </div>
                  <div>
                    <span className="metric-icon teal">
                      <Zap size={18} />
                    </span>
                    <span>
                      Thời gian xử lý
                      <strong>
                        {result ? Math.round(result.processingMs) : "—"}{" "}
                        <small>ms / frame</small>
                      </strong>
                    </span>
                  </div>
                  <div>
                    <span className="metric-icon lavender">
                      <Focus size={18} />
                    </span>
                    <span>
                      Chuỗi nhận diện
                      <strong>
                        {result?.people.length
                          ? Math.max(...result.people.map((p) => p.bufferCount))
                          : "—"}{" "}
                        <small>/ 10 frame</small>
                      </strong>
                    </span>
                  </div>
                </div>
                <p className="camera-hint">
                  <ShieldCheck size={14} />
                  Video chỉ được xử lý trong phiên. Không ghi hình, không lưu
                  trữ.
                </p>
              </section>
              <PredictionPanel
                people={result?.people ?? []}
                selectedId={selectedId}
                select={setSelectedId}
                running={status === "running"}
              />
            </div>
          </section>

          <section id="actions" className="action-section">
            <div className="section-heading">
              <div>
                <h2>Một mô hình. Sáu hành động.</h2>
                <p>Những tư thế quen thuộc mà Pose Studio có thể nhận diện.</p>
              </div>
              <button className="text-button" onClick={() => setModal("model")}>
                Khám phá mô hình
                <ArrowUpRight size={15} />
              </button>
            </div>
            <div className="action-library">
              {LABELS.map((label, i) => (
                <button
                  className="action-card"
                  key={label}
                  onClick={() => setModal(i)}
                  style={{ "--action-color": COLORS[i] } as React.CSSProperties}
                >
                  <div className="action-art">
                    <PoseFigure pose={i} color={COLORS[i]} />
                  </div>
                  <span>{label}</span>
                  <small>
                    {i === 2 || i === 3
                      ? "Hành động cần chú ý"
                      : "Nhận diện chuyển động"}
                  </small>
                  <ArrowUpRight className="action-arrow" size={14} />
                </button>
              ))}
            </div>
          </section>
          <EventTimeline events={events} clear={recognition.clearEvents} />
          <footer className="page-footer">
            <span>
              <Activity size={15} />
              Pose Studio <span className="footer-separator">/</span> Thị giác
              máy tính trong từng chuyển động
            </span>
            <button className="text-button" onClick={() => setModal("model")}>
              Về bản demo
              <ArrowUpRight size={13} />
            </button>
          </footer>
        </main>
      </div>
      <dialog
        ref={dialog}
        className="info-dialog"
        onCancel={() => setModal(null)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setModal(null);
        }}
      >
        <div className="dialog-content">
          <button
            className="icon-button dialog-close"
            aria-label="Đóng"
            onClick={() => setModal(null)}
          >
            <X size={20} />
          </button>
          {modal === "help" ? (
            <>
              <span className="dialog-icon">
                <Video size={27} />
              </span>
              <h2>Sẵn sàng cho lần demo đầu?</h2>
              <p>
                Chỉ vài bước để quan sát cách mô hình đọc chuyển động của bạn.
              </p>
              <ol className="help-steps">
                <li>
                  <strong>Chọn nguồn hình ảnh</strong>
                  <p>
                    Dùng webcam và cho phép truy cập camera, hoặc chọn video
                    MP4/WebM trên máy.
                  </p>
                </li>
                <li>
                  <strong>Bắt đầu nhận diện</strong>
                  <p>
                    Đảm bảo toàn thân nằm trong khung hình, đủ ánh sáng và tối
                    đa hai người. Đợi thu thập đủ 10 frame.
                  </p>
                </li>
                <li>
                  <strong>Khám phá kết quả</strong>
                  <p>
                    Xem khung xương, chọn từng người để xem xác suất. Lịch sử
                    ghi nhận sau 3 lần dự đoán cùng hành động.
                  </p>
                </li>
              </ol>
              <div className="dialog-note">
                <Info size={18} />
                <span>
                  Nếu backend chưa kết nối, mở terminal tại thư mục dự án và
                  chạy <code>.\scripts\start-be.ps1</code>.
                </span>
              </div>
              <button className="button primary" onClick={() => setModal(null)}>
                Đã hiểu, bắt đầu thôi
                <ChevronRight size={16} />
              </button>
            </>
          ) : typeof modal === "number" ? (
            <>
              <div className="dialog-pose">
                <PoseFigure pose={modal} color={COLORS[modal]} />
              </div>
              <h2>{LABELS[modal]}</h2>
              <p>
                {ACTION_DESCRIPTIONS[modal]}. Mô hình kết hợp vị trí cơ thể qua
                10 khung hình liên tiếp để đưa ra dự đoán.
              </p>
              {(modal === 2 || modal === 3) && (
                <div className="dialog-note amber">
                  <Info size={18} />
                  <span>
                    Được đánh dấu “Cần chú ý” trong bản demo. Đây là mô tả hành
                    động, không phải đánh giá năng suất.
                  </span>
                </div>
              )}
              <p className="muted">
                Các tư thế tương tự, góc camera hoặc che khuất có thể làm thay
                đổi kết quả.
              </p>
              <button
                className="button primary"
                onClick={() => {
                  setModal(null);
                  navigate("studio");
                }}
              >
                Thử nhận diện
                <Play size={15} />
              </button>
            </>
          ) : (
            <>
              <span className="dialog-icon">
                <Cpu size={27} />
              </span>
              <h2>Chuyển động, qua góc nhìn AI.</h2>
              <p>
                Pose Studio kết hợp bộ trích xuất tư thế MediaPipe và mô hình
                LSTM đã huấn luyện trong dự án.
              </p>
              <div className="model-flow">
                <span>Khung hình</span>
                <ChevronRight size={15} />
                <span>33 điểm cơ thể</span>
                <ChevronRight size={15} />
                <span>6 hành động</span>
              </div>
              <dl className="model-details">
                <div>
                  <dt>Đầu vào</dt>
                  <dd>10 frame × 132 đặc trưng</dd>
                </div>
                <div>
                  <dt>Mô hình</dt>
                  <dd>Bidirectional LSTM</dd>
                </div>
                <div>
                  <dt>Số người</dt>
                  <dd>Tối đa 2 người / frame</dd>
                </div>
                <div>
                  <dt>Chế độ xử lý</dt>
                  <dd>Cục bộ, không lưu video</dd>
                </div>
              </dl>
              <div className="dialog-note">
                <Info size={18} />
                <span>
                  Độ tin cậy là xác suất do mô hình trả về, không phải độ chính
                  xác đã kiểm định. Nhãn có thể nhầm khi góc quay hoặc hành động
                  khác dữ liệu huấn luyện.
                </span>
              </div>
              <button className="button primary" onClick={() => setModal(null)}>
                Quay lại phòng nhận diện
                <ChevronRight size={16} />
              </button>
            </>
          )}
        </div>
      </dialog>
    </div>
  );
}
