import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { PdfInspector } from "../PdfInspector";

jest.mock("@/lib/api", () => ({
  fetchDocumentRuns: jest.fn().mockResolvedValue([
    { run_name: "run_001", files_count: 2 }
  ]),
  fetchRunFiles: jest.fn().mockResolvedValue(["document1.md", "document2.md"]),
  fetchMarkdownContent: jest.fn().mockResolvedValue("# Sample Content"),
  fetchDocumentInfo: jest.fn().mockResolvedValue({
    total_pages: 5,
    page_ranges: [[0, 10, 1], [10, 20, 2]],
    pages_markdown: { "1": "# Page 1 Content", "2": "# Page 2 Content" },
  }),
}));

describe("PdfInspector Component", () => {
  test("renders dual-pane PDF & Markdown inspector", async () => {
    render(<PdfInspector />);

    expect(screen.getByText("📄 Select Processed Document Run")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Sync Scroll")).toBeInTheDocument();
    });
  });
});
