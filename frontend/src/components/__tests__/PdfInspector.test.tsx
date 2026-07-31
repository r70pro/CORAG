import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { PdfInspector } from "../PdfInspector";
import {
  fetchDocumentInfo,
  fetchDocumentRuns,
  fetchMarkdownContent,
  fetchRunFiles,
} from "@/lib/api";

jest.mock("@/lib/api", () => ({
  apiUrl: (path: string) => path,
  apiPathSegment: (value: string) => encodeURIComponent(value),
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
  downloadRunMarkdownZip: jest.fn().mockResolvedValue({ success: true }),
}));

describe("PdfInspector Component", () => {
  beforeEach(() => {
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_001", files_count: 2 }
    ]);
    jest.mocked(fetchMarkdownContent).mockResolvedValue("# Sample Content");
    jest.mocked(fetchDocumentInfo).mockResolvedValue({
      total_pages: 5,
      page_ranges: [[0, 10, 1], [10, 20, 2]],
      pages_markdown: { "1": "# Page 1 Content", "2": "# Page 2 Content" },
    });
  });

  test("renders dual-pane PDF & Markdown inspector", async () => {
    render(<PdfInspector />);

    expect(screen.getByText("📄 Select Processed Document Run")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Sync Scroll")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Download All \(ZIP\)/i })).toBeInTheDocument();
    });
  });

  test("does not inject active HTML from document markdown", async () => {
    const malicious =
      "<table><tbody><tr><td><img src='x' onerror='window.phase0Xss=1'></td></tr></tbody></table>";
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_001", files: ["document.md"] }
    ]);
    jest.mocked(fetchMarkdownContent).mockResolvedValue(malicious);
    jest.mocked(fetchDocumentInfo).mockResolvedValue({
      total_pages: 1,
      page_ranges: [[0, malicious.length, 1]],
      pages_markdown: { "1": malicious },
    });

    render(<PdfInspector />);

    await waitFor(() => {
      expect(screen.getAllByText(malicious).length).toBeGreaterThan(0);
    });
    expect(document.querySelector("img[onerror]")).not.toBeInTheDocument();
  });

  test("renders strict OCR tables as React elements", async () => {
    const table = "<table><thead><tr><th>Item</th><th>Value</th></tr></thead><tbody><tr><td>Safe</td><td>&lt;text&gt;</td></tr></tbody></table>";
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_001", files: ["document.md"] }
    ]);
    jest.mocked(fetchMarkdownContent).mockResolvedValue(table);
    jest.mocked(fetchDocumentInfo).mockResolvedValue({
      total_pages: 1,
      page_ranges: [[0, table.length, 1]],
      pages_markdown: { "1": table },
    });

    render(<PdfInspector />);

    await waitFor(() => {
      expect(screen.getAllByRole("table").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("<text>").length).toBeGreaterThan(0);
  });

  test("does not request a PDF for an external Markdown-only run", async () => {
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_external", files: ["external.md"], has_pdf: false }
    ]);

    render(<PdfInspector />);

    expect(await screen.findByText("No original PDF is attached to this Markdown run.")).toBeInTheDocument();
    expect(screen.queryByTitle("Source PDF Viewer")).not.toBeInTheDocument();
  });

  test("clears the old filename before loading files for a different run", async () => {
    let resolveFiles: (files: string[]) => void = () => undefined;
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_old", display_name: "Old", files: ["old.md"], has_pdf: true },
      { run_name: "run_new", display_name: "New", files: ["new.md"], has_pdf: true },
    ]);
    jest.mocked(fetchRunFiles).mockImplementation(
      () => new Promise<string[]>((resolve) => { resolveFiles = resolve; }),
    );

    render(<PdfInspector />);
    await waitFor(() => {
      expect(jest.mocked(fetchMarkdownContent)).toHaveBeenCalledWith("run_old", "old.md");
    });
    jest.mocked(fetchMarkdownContent).mockClear();
    const selector = screen.getByLabelText("📄 Select Processed Document Run") as HTMLSelectElement;
    fireEvent.change(selector, { target: { value: "run_new" } });

    expect(jest.mocked(fetchMarkdownContent)).not.toHaveBeenCalledWith("run_new", "old.md");
    resolveFiles(["new.md"]);
    await waitFor(() => {
      expect(jest.mocked(fetchMarkdownContent)).toHaveBeenCalledWith("run_new", "new.md");
    });
  });

  test("renders line breaks inside otherwise strict OCR table cells", async () => {
    const table = "<table><tr><th>Event</th></tr><tr><td>First line<br>Second line</td></tr></table>";
    jest.mocked(fetchDocumentRuns).mockResolvedValue([
      { run_name: "run_001", files: ["document.md"] }
    ]);
    jest.mocked(fetchMarkdownContent).mockResolvedValue(table);
    jest.mocked(fetchDocumentInfo).mockResolvedValue({
      total_pages: 1,
      page_ranges: [[0, table.length, 1]],
      pages_markdown: { "1": table },
    });

    render(<PdfInspector />);

    await waitFor(() => {
      expect(screen.getAllByRole("table").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/First line\s+Second line/).length).toBeGreaterThan(0);
  });
});
