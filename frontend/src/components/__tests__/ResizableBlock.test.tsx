import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResizableBlock } from "../ResizableBlock";

describe("ResizableBlock Component", () => {
  test("renders title and children correctly", () => {
    render(
      <ResizableBlock id="test_block" title="Test Block Title">
        <div>Content Inside Block</div>
      </ResizableBlock>
    );

    expect(screen.getByText("Test Block Title")).toBeInTheDocument();
    expect(screen.getByText("Content Inside Block")).toBeInTheDocument();
  });

  test("toggles collapse state on clicking collapse handle button", () => {
    render(
      <ResizableBlock id="test_collapse" title="Collapsible Block">
        <div>Content To Hide</div>
      </ResizableBlock>
    );

    expect(screen.getByText("Content To Hide")).toBeInTheDocument();
    const collapseBtn = screen.getByTitle("Collapse Block");
    fireEvent.click(collapseBtn);

    expect(screen.queryByText("Content To Hide")).not.toBeInTheDocument();
  });
});
