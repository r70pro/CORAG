import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
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

  test("restores split sizes from localStorage after mount", async () => {
    localStorage.setItem("kirag_split_test_restore", JSON.stringify([30, 70]));

    render(
      <ResizableSplit storageKey="test_restore" initialSizes={[50, 50]}>
        <div>Left Panel</div>
        <div>Right Panel</div>
      </ResizableSplit>
    );

    const leftPanel = screen.getByText("Left Panel").parentElement;
    await waitFor(() => {
      expect(leftPanel).toHaveStyle("flex-basis: 30%");
    });
  });

  test("maintains restored split sizes across parent re-renders with new inline initialSizes references", async () => {
    localStorage.setItem("kirag_split_test_rerender", JSON.stringify([45, 55]));

    const { rerender } = render(
      <ResizableSplit storageKey="test_rerender" initialSizes={[65, 35]}>
        <div>Top Section</div>
        <div>Bottom Log</div>
      </ResizableSplit>
    );

    const topSection = screen.getByText("Top Section").parentElement;
    await waitFor(() => {
      expect(topSection).toHaveStyle("flex-basis: 45%");
    });

    // Re-render with new array instance of same values
    rerender(
      <ResizableSplit storageKey="test_rerender" initialSizes={[65, 35]}>
        <div>Top Section Updated</div>
        <div>Bottom Log Updated</div>
      </ResizableSplit>
    );

    const updatedTop = screen.getByText("Top Section Updated").parentElement;
    expect(updatedTop).toHaveStyle("flex-basis: 45%");
  });
});
