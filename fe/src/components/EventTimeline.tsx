import { Download, History, Trash2 } from "lucide-react";
import { COLORS, LABELS, isAlert, type ActivityEvent } from "../types";
import { exportHistory } from "../history";
export function EventTimeline({
  events,
  clear,
}: {
  events: ActivityEvent[];
  clear: () => void;
}) {
  return (
    <section id="history" className="panel history-panel">
      <div className="panel-heading">
        <div>
          <h2>
            <History size={18} />
            Lịch sử phiên <span className="count-badge">{events.length}</span>
          </h2>
          <p>Các thay đổi hành động được ghi lại trong phiên này.</p>
        </div>
        <div className="history-actions">
          <button
            className="icon-button"
            disabled={!events.length}
            onClick={clear}
            aria-label="Xóa lịch sử"
            title="Xóa lịch sử"
          >
            <Trash2 size={16} />
          </button>
          <button
            className="button secondary small"
            disabled={!events.length}
            onClick={() => exportHistory(events)}
          >
            <Download size={15} />
            Xuất CSV
          </button>
        </div>
      </div>
      {!events.length ? (
        <div className="history-empty">
          <div className="empty-history-icon">
            <History size={25} />
          </div>
          <div>
            <h3>Mọi chuyển động bắt đầu từ đây</h3>
            <p>
              Bật camera hoặc chọn video. Lịch sử sẽ xuất hiện khi hành động
              được nhận diện ổn định.
            </p>
          </div>
          <span className="empty-dashes">— — —</span>
        </div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Thời gian</th>
                <th>Đối tượng</th>
                <th>Hành động</th>
                <th>Độ tin cậy</th>
                <th>Ghi chú</th>
              </tr>
            </thead>
            <tbody>
              {events.slice(0, 100).map((e) => (
                <tr key={e.id}>
                  <td>{new Date(e.at).toLocaleTimeString("vi-VN")}</td>
                  <td>Người {e.personId}</td>
                  <td>
                    <span className="action-label">
                      <i style={{ background: COLORS[e.labelIndex] }} />
                      {LABELS[e.labelIndex]}
                    </span>
                  </td>
                  <td>{(e.confidence * 100).toFixed(1)}%</td>
                  <td>
                    <span
                      className={
                        isAlert(e.labelIndex)
                          ? "status-pill warning"
                          : "status-pill"
                      }
                    >
                      {isAlert(e.labelIndex) ? "Cần chú ý" : "Đã ghi nhận"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="history-footer">
        <span>Lưu tối đa 500 sự kiện trong phiên · Không lưu video</span>
        <span>Ổn định qua 3 lần dự đoán</span>
      </div>
    </section>
  );
}
