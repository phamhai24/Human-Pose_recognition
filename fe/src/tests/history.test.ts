import { expect, it } from "vitest";
import { csvCell, collectEvents } from "../history";
import type { Person } from "../types";
const person = (labelIndex: number | null): Person => ({
  id: 1,
  landmarks: [],
  bufferCount: 10,
  labelIndex,
  probabilities: [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
});
it("escapes CSV delimiters and prevents spreadsheet formula execution", () => {
  expect(csvCell('a,"b"')).toBe('"a,""b"""');
  expect(csvCell("=1+1")).toBe("'=1+1");
});
it("logs only stable transitions and clears absent people", () => {
  const state = new Map();
  expect(collectEvents([person(null)], state, 0)).toHaveLength(0);
  expect(collectEvents([person(0)], state, 100)).toHaveLength(0);
  expect(collectEvents([person(0)], state, 200)).toHaveLength(0);
  expect(collectEvents([person(0)], state, 300)).toHaveLength(1);
  expect(collectEvents([person(0)], state, 400)).toHaveLength(0);
  collectEvents([], state, 500);
  expect(state.size).toBe(0);
});
