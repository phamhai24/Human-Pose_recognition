import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useRecognition } from "../hooks/useRecognition";
vi.mock("../api", () => ({
  getHealth: async () => ({ status: "ready", message: "" }),
  socketUrl: () => "ws://localhost/api/recognize",
}));
afterEach(() => vi.unstubAllGlobals());
it("camera denial returns an actionable error and allows another start", async () => {
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: async () => {
        throw new DOMException("denied", "NotAllowedError");
      },
    },
  });
  const { result } = renderHook(() => useRecognition());
  const video = document.createElement("video");
  act(() => {
    result.current.videoRef.current = video;
  });
  await act(async () => {
    await result.current.start("camera");
  });
  expect(result.current.status).toBe("error");
  expect(result.current.error).toContain("quyền");
  act(() => result.current.stop());
  expect(result.current.status).toBe("idle");
});
it("stopping while camera permission is pending releases the late stream", async () => {
  let resolve!: (stream: MediaStream) => void;
  const stopped = vi.fn();
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: () =>
        new Promise((r) => {
          resolve = r;
        }),
    },
  });
  const { result } = renderHook(() => useRecognition());
  result.current.videoRef.current = document.createElement("video");
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.start("camera");
  });
  await waitFor(() => expect(resolve).toBeTypeOf("function"));
  act(() => result.current.stop());
  await act(async () => {
    resolve({ getTracks: () => [{ stop: stopped }] } as unknown as MediaStream);
    await pending;
  });
  expect(stopped).toHaveBeenCalledOnce();
  expect(result.current.status).toBe("idle");
});
