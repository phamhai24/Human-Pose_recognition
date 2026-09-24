export type Landmark = { x: number; y: number; z: number; visibility: number };
export type Person = {
  id: number;
  landmarks: Landmark[];
  bufferCount: number;
  labelIndex: number | null;
  probabilities: number[] | null;
};
export type FrameResult = {
  type: "result";
  frameId: number;
  processingMs: number;
  people: Person[];
};
export type Health = { status: "ready" | "unavailable"; message: string };
export type Status = "idle" | "connecting" | "running" | "error";
export type ActivityEvent = {
  id: string;
  at: number;
  personId: number;
  labelIndex: number;
  confidence: number;
};
export const LABELS = [
  "Ngồi làm việc",
  "Ngồi ngả lưng",
  "Nằm ngủ",
  "Gác chân",
  "Đứng dậy",
  "Đi lại",
];
export const COLORS = [
  "#3563e9",
  "#9674d8",
  "#e3904d",
  "#e06e86",
  "#16a694",
  "#61a2ce",
];
export const isAlert = (index: number | null) => index === 2 || index === 3;
