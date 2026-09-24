import { describe, expect, it } from "vitest";
import { projectPoint } from "../pose";
describe("overlay", () => {
  it("mirrors overlay around the same center as the webcam", () => {
    expect(projectPoint({ x: 0.25, y: 0.5 }, 800, 450, true)).toEqual({
      x: 600,
      y: 225,
    });
    expect(projectPoint({ x: 0.25, y: 0.5 }, 800, 450, false)).toEqual({
      x: 200,
      y: 225,
    });
  });
});
