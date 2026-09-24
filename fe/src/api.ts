import type { Health } from "./types";
const base = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
export async function getHealth(): Promise<Health> {
  const response = await fetch(`${base}/api/health`, {
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) throw new Error("Backend không phản hồi.");
  return response.json();
}
export function socketUrl() {
  const url = new URL(`${base}/api/recognize`, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
