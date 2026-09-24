import { ChartNoAxesCombined, CircleDot, Users, ScanLine } from "lucide-react";
import { COLORS, LABELS, isAlert, type Person } from "../types";
import { PoseFigure } from "./PoseFigure";
export function PredictionPanel({
  people,
  selectedId,
  select,
  running,
}: {
  people: Person[];
  selectedId: number | null;
  select: (id: number) => void;
  running: boolean;
}) {
  const person =
    people.find((p) => p.id === selectedId) ??
    people.reduce<Person | undefined>(
      (first, next) => (!first || next.id < first.id ? next : first),
      undefined,
    );
  const label = person?.labelIndex ?? null;
  const confidence = label !== null ? person?.probabilities?.[label] : null;
  return (
    <section
      className="prediction-panel panel"
      aria-labelledby="prediction-title"
    >
      <div className="panel-heading">
        <h2 id="prediction-title">
          <ChartNoAxesCombined size={18} />
          Kết quả nhận diện
        </h2>
        <span className="small-dot" />
      </div>
      <div className="person-select">
        <Users size={16} />
        {person ? (
          <select
            aria-label="Chọn người"
            value={person.id}
            onChange={(e) => select(Number(e.target.value))}
          >
            {people.map((p) => (
              <option key={p.id} value={p.id}>
                Người {p.id}
              </option>
            ))}
          </select>
        ) : (
          <span>Chưa có người được nhận diện</span>
        )}
      </div>
      <div className={`prediction-focus ${isAlert(label) ? "attention" : ""}`}>
        <div className="prediction-figure">
          <PoseFigure
            pose={label ?? 4}
            color={label !== null ? COLORS[label] : "#a0b1d0"}
          />
        </div>
        <span className="prediction-eyebrow">
          {label !== null ? "Hành động hiện tại" : "Đang chờ chuyển động"}
        </span>
        <h3>
          {label !== null
            ? LABELS[label]
            : person
              ? "Đang quan sát…"
              : "Chưa có kết quả"}
        </h3>
        <span className="confidence-chip">
          <CircleDot size={13} />
          {confidence != null
            ? `${(confidence * 100).toFixed(1)}% độ tin cậy`
            : person
              ? `${person.bufferCount}/10 khung hình`
              : "Bắt đầu phiên để nhận diện"}
        </span>
        {isAlert(label) && <p className="alert-caption">Hành động cần chú ý</p>}
      </div>
      <div className="probability-heading">
        <span>Phân bố xác suất</span>
        <span>6 hành động</span>
      </div>
      <div className="probabilities">
        {LABELS.map((name, i) => (
          <div className="probability" key={name}>
            <div>
              <span>
                <i style={{ background: COLORS[i] }} />
                {name}
              </span>
              <strong>
                {person?.probabilities
                  ? `${(person.probabilities[i] * 100).toFixed(1)}%`
                  : "—"}
              </strong>
            </div>
            <div className="probability-track">
              <span
                style={{
                  width: `${(person?.probabilities?.[i] ?? 0) * 100}%`,
                  background: COLORS[i],
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="prediction-foot">
        <ScanLine size={15} />
        {running
          ? "Kết quả cập nhật theo từng khung hình"
          : "Dữ liệu thật từ phiên nhận diện của bạn"}
      </div>
    </section>
  );
}
