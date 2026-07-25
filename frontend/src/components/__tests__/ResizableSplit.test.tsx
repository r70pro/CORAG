import React from "react";
import { render, screen } from "@testing-library/react";
import { ResizableSplit } from "../ResizableSplit";

describe("ResizableSplit Component", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test("renders children with initial fallback sizes without hydration mismatch", () => {
    render(
      <ResizableSplit storageKey="test_split" initialSizes={[60, 40]}>
        <div>Panel A</div>
        <div>Panel B</div>
      </ResizableSplit>
    );

    const panelA = screen.getByText("Panel A").parentElement;
    expect(panelA).toHaveStyle("flex-basis: 60%");
  });

  test("restores split sizes from localStorage after mount", () => {
    localStorage.setItem("kirag_split_test_restore", JSON.stringify([30, 70]));

    render(
      <ResizableSplit storageKey="test_restore" initialSizes={[50, 50]}>
        <div>Left Panel</div>
        <div>Right Panel</div>
      </ResizableSplit>
    );

    const leftPanel = screen.getByText("Left Panel").parentElement;
    expect(leftPanel).toHaveStyle("flex-basis: 30%");
  });
});
