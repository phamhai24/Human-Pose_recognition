import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { PredictionPanel } from "../components/PredictionPanel";
import { collectEvents, stablePeople } from "../history";
import type { Person } from "../types";

const person = (id: number, labelIndex = 0): Person => ({
  id,
  labelIndex,
  bufferCount: 10,
  probabilities: [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
  landmarks: [],
});
afterEach(cleanup);
it("default selected person remains the earliest track when detection order changes", () => {
  const props = { selectedId: null, select: () => {}, running: true };
  const { rerender } = render(
    <PredictionPanel {...props} people={[person(1), person(2)]} />,
  );
  expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("1");
  rerender(<PredictionPanel {...props} people={[person(2), person(1)]} />);
  expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("1");
  rerender(
    <PredictionPanel
      {...props}
      selectedId={2}
      people={[person(1), person(2)]}
    />,
  );
  expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("2");
});
it("displayed action does not flicker into an alert after one transient prediction", () => {
  const state = new Map();
  const display = (label: number) => {
    const people = [person(1, label)];
    collectEvents(people, state, Date.now());
    return stablePeople(people, state)[0].labelIndex;
  };
  expect(display(0)).toBeNull();
  expect(display(0)).toBeNull();
  expect(display(0)).toBe(0);
  expect(display(2)).toBe(0);
  expect(display(0)).toBe(0);
  expect(display(2)).toBe(0);
  expect(display(2)).toBe(0);
  expect(display(2)).toBe(2);
  collectEvents([], state, 0);
  expect(display(0)).toBeNull();
});
