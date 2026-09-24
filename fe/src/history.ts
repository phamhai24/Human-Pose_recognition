import { LABELS, type ActivityEvent, type Person } from "./types";
export type Transition = {
  candidate: number;
  count: number;
  emitted: number | null;
};
export function collectEvents(
  people: Person[],
  state: Map<number, Transition>,
  at: number,
): ActivityEvent[] {
  const present = new Set(people.map((p) => p.id));
  for (const id of state.keys()) if (!present.has(id)) state.delete(id);
  const events: ActivityEvent[] = [];
  for (const p of people) {
    if (p.labelIndex === null || !p.probabilities) continue;
    const old = state.get(p.id);
    const next = {
      candidate: p.labelIndex,
      count: old?.candidate === p.labelIndex ? old.count + 1 : 1,
      emitted: old?.emitted ?? null,
    };
    if (next.count >= 3 && next.emitted !== p.labelIndex) {
      events.push({
        id: `${at}-${p.id}`,
        at,
        personId: p.id,
        labelIndex: p.labelIndex,
        confidence: p.probabilities[p.labelIndex],
      });
      next.emitted = p.labelIndex;
    }
    state.set(p.id, next);
  }
  return events;
}
export function csvCell(value: string): string {
  const safe = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  return /[",\n\r]/.test(safe) ? `"${safe.replaceAll('"', '""')}"` : safe;
}
export function stablePeople(
  people: Person[],
  state: Map<number, Transition>,
): Person[] {
  return people.map((person) => ({
    ...person,
    labelIndex:
      person.labelIndex === null
        ? null
        : (state.get(person.id)?.emitted ?? null),
  }));
}
export function exportHistory(events: ActivityEvent[]) {
  const rows = [
    ["Thời gian", "Người", "Hành động", "Độ tin cậy"],
    ...events.map((e) => [
      new Date(e.at).toISOString(),
      String(e.personId),
      LABELS[e.labelIndex],
      (e.confidence * 100).toFixed(1) + "%",
    ]),
  ];
  const url = URL.createObjectURL(
    new Blob(
      ["\uFEFF" + rows.map((r) => r.map(csvCell).join(",")).join("\r\n")],
      { type: "text/csv;charset=utf-8" },
    ),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "pose-studio-history.csv";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
