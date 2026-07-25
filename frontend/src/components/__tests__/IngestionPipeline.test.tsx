import React from "react";
import { render, screen } from "@testing-library/react";
import { IngestionPipeline } from "../IngestionPipeline";

jest.mock("@/lib/api", () => ({
  triggerIngestSSE: jest.fn(),
  stopPipelineRun: jest.fn().mockResolvedValue({ success: true }),
  updateSettings: jest.fn().mockResolvedValue({ success: true }),
  fetchSettings: jest.fn().mockResolvedValue({ server_url: "http://localhost:8000", model_name: "test-model" }),
}));

describe("IngestionPipeline Component", () => {
  test("renders PDF document ingestion interface", async () => {
    render(<IngestionPipeline />);

    expect(await screen.findByText("Ingestion Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/High-throughput document OCR processing/i)).toBeInTheDocument();
    expect(screen.getByText("Pipeline Settings")).toBeInTheDocument();
  });

  test("calculates progress percentage correctly from progress HTML without false 100%", async () => {
    const { triggerIngestSSE } = require("@/lib/api");
    let sseCallback: (data: unknown) => void = () => {};

    triggerIngestSSE.mockImplementation((_opts: unknown, onData: (data: unknown) => void) => {
      sseCallback = onData;
    });

    const { act } = require("@testing-library/react");
    await act(async () => {
      render(<IngestionPipeline />);
    });

    const startButton = screen.getByText("Start Batch Processing");
    await act(async () => {
      startButton.click();
      sseCallback({
        progress_html: "<div style='width:100%;'><span>0/10 Pages</span><span>0%</span></div>",
        completed_pages: 0,
        failed_pages: 0,
      });
    });

    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  test("parses and updates per-file status dynamically from SSE event payload", async () => {
    const { triggerIngestSSE } = require("@/lib/api");
    let sseCallback: (data: unknown) => void = () => {};

    triggerIngestSSE.mockImplementation((_opts: unknown, onData: (data: unknown) => void) => {
      sseCallback = onData;
    });

    const { act } = require("@testing-library/react");
    await act(async () => {
      render(<IngestionPipeline />);
    });

    const startButton = screen.getByText("Start Batch Processing");
    await act(async () => {
      startButton.click();
      sseCallback({
        file_status_html: "<table><tbody><tr><td>test_doc.pdf</td><td>15</td><td><span>✓ Done</span></td></tr></tbody></table>",
        upload_manifest_html: "<table><tbody><tr><td>test_doc.pdf</td><td>15</td><td>1.2 MB</td></tr></tbody></table>",
      });
    });

    expect(screen.getByText("test_doc.pdf")).toBeInTheDocument();
    expect(screen.getByText("✓ Done")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
  });
});

