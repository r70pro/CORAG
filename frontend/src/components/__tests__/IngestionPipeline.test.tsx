import React from "react";
import { render, screen, act, fireEvent } from "@testing-library/react";
import { triggerIngestSSE } from "@/lib/api";
import { IngestionPipeline } from "../IngestionPipeline";

jest.mock("@/lib/api", () => ({
  triggerIngestSSE: jest.fn(),
  stopPipelineRun: jest.fn().mockResolvedValue({ success: true }),
  updateSettings: jest.fn().mockResolvedValue({ success: true }),
  fetchSettings: jest.fn().mockResolvedValue({ server_url: "http://localhost:8000", model_name: "test-model" }),
}));

const mockTriggerIngestSSE = triggerIngestSSE as jest.Mock;

describe("IngestionPipeline Component", () => {
  test("renders PDF document ingestion interface", async () => {
    render(<IngestionPipeline />);

    expect(await screen.findByText("Ingestion Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/High-throughput document OCR processing/i)).toBeInTheDocument();
    expect(screen.getByText("Pipeline Settings")).toBeInTheDocument();
  });

  test("calculates progress percentage correctly from progress HTML without false 100%", async () => {
    let sseCallback: (data: unknown) => void = () => {};

    mockTriggerIngestSSE.mockImplementation((_opts: unknown, onData: (data: unknown) => void) => {
      sseCallback = onData;
    });

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
    let sseCallback: (data: unknown) => void = () => {};

    mockTriggerIngestSSE.mockImplementation((_opts: unknown, onData: (data: unknown) => void) => {
      sseCallback = onData;
    });

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

  test("preserves a pre-flight failure when the SSE stream closes", async () => {
    let sseCallback: (data: unknown) => void = () => {};
    let completeCallback: () => void = () => {};
    mockTriggerIngestSSE.mockImplementation(
      (
        _opts: unknown,
        onData: (data: unknown) => void,
        _onError: (error: unknown) => void,
        onComplete: () => void,
      ) => {
        sseCallback = onData;
        completeCallback = onComplete;
      },
    );

    render(<IngestionPipeline />);
    fireEvent.click(await screen.findByText("Start Batch Processing"));
    await act(async () => {
      sseCallback({
        log_text: "Pre-flight check failed",
        status_badge: "<span class='badge-failed'>Model Mismatch</span>",
      });
      completeCallback();
    });

    expect(screen.getByText("● Model Mismatch")).toBeInTheDocument();
    expect(screen.getByText("[Terminated] Pipeline processing ended with errors.")).toBeInTheDocument();
    expect(screen.queryByText("[Complete] Pipeline batch processing finished.")).not.toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
