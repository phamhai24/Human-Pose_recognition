import type { Person } from "./types";
export const CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 29],
  [29, 31],
  [28, 30],
  [30, 32],
  [15, 17],
  [15, 19],
  [16, 18],
  [16, 20],
];
export function projectPoint(
  point: { x: number; y: number },
  width: number,
  height: number,
  mirror: boolean,
) {
  return { x: (mirror ? 1 - point.x : point.x) * width, y: point.y * height };
}
export function drawPeople(
  canvas: HTMLCanvasElement,
  people: Person[],
  mirror: boolean,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  people.forEach((person, index) => {
    const color = index === 0 ? "#40e8c7" : "#ffc776";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(2, canvas.width / 350);
    for (const [a, b] of CONNECTIONS) {
      const pa = person.landmarks[a],
        pb = person.landmarks[b];
      if (!pa || !pb || pa.visibility < 0.4 || pb.visibility < 0.4) continue;
      const start = projectPoint(pa, canvas.width, canvas.height, mirror),
        end = projectPoint(pb, canvas.width, canvas.height, mirror);
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    }
    for (const p of person.landmarks) {
      if (p.visibility < 0.4) continue;
      const { x, y } = projectPoint(p, canvas.width, canvas.height, mirror);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(3, canvas.width / 220), 0, Math.PI * 2);
      ctx.fill();
    }
    const head = person.landmarks[0];
    if (head) {
      const p = projectPoint(head, canvas.width, canvas.height, mirror);
      ctx.font = "bold 18px sans-serif";
      const x = Math.max(6, Math.min(canvas.width - 120, p.x - 42)),
        y = Math.max(30, p.y - 30);
      ctx.fillStyle = "#172b4d";
      ctx.fillRect(x - 6, y - 22, 112, 30);
      ctx.fillStyle = color;
      ctx.fillText(`Người ${person.id}`, x, y);
    }
  });
}
