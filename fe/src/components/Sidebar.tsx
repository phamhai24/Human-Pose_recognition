import {
  Activity,
  AudioLines,
  CircleHelp,
  History,
  ScanLine,
  Video,
  ArrowUpRight,
  Sparkles,
} from "lucide-react";
export function Sidebar({
  onHelp,
  onAbout,
  active,
  onNavigate,
}: {
  onHelp: () => void;
  onAbout: () => void;
  active: string;
  onNavigate: (id: string) => void;
}) {
  return (
    <aside className="sidebar">
      <a className="brand" href="#studio" onClick={() => onNavigate("studio")}>
        <span className="brand-symbol">
          <Activity size={25} />
        </span>
        <span>
          pose<span className="brand-light">studio</span>
          <small>Chuyển động thành thông tin</small>
        </span>
      </a>
      <div className="workspace">
        <span className="workspace-icon">
          <ScanLine size={20} />
        </span>
        <div>
          Không gian demo<small>Nhận diện hành động</small>
        </div>
        <span className="workspace-dot" />
      </div>
      <p className="nav-caption">Khám phá</p>
      <nav aria-label="Điều hướng chính">
        <button
          className={active === "studio" ? "nav-item active" : "nav-item"}
          onClick={() => onNavigate("studio")}
        >
          <Video size={19} />
          Phòng nhận diện
          <span className="nav-indicator" />
        </button>
        <button
          className={active === "history" ? "nav-item active" : "nav-item"}
          onClick={() => onNavigate("history")}
        >
          <History size={19} />
          Lịch sử phiên
        </button>
        <button
          className={active === "actions" ? "nav-item active" : "nav-item"}
          onClick={() => onNavigate("actions")}
        >
          <AudioLines size={19} />
          Thư viện hành động
        </button>
      </nav>
      <div className="sidebar-bottom">
        <div className="side-note">
          <span className="note-icon">
            <Sparkles size={19} />
          </span>
          <h3>
            Mỗi chuyển động,
            <br />
            một câu chuyện.
          </h3>
          <p>Khám phá cách AI hiểu những hành động thường ngày.</p>
          <button onClick={onAbout}>
            Tìm hiểu mô hình <ArrowUpRight size={16} />
          </button>
        </div>
        <button className="help-link" onClick={onHelp}>
          <CircleHelp size={18} />
          Hướng dẫn sử dụng
        </button>
        <div className="project-signature">
          <span className="signature-icon">
            <ScanLine size={18} />
          </span>
          <div>
            Computer Vision Lab<small>Pose Studio / v1.0</small>
          </div>
        </div>
      </div>
    </aside>
  );
}
