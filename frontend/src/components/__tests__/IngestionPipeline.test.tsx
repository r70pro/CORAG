import React from "react";
import { render, screen } from "@testing-library/react";
import { IngestionPipeline } from "../IngestionPipeline";

jest.mock("@/lib/api", () => ({
  triggerIngestSSE: jest.fn(),
  stopPipelineRun: jest.fn().mockResolvedValue({ success: true }),
  updateSettings: jest.fn().mockResolvedValue({ success: true }),
}));

describe("IngestionPipeline Component", () => {
  test("renders PDF document ingestion interface", () => {
    render(<IngestionPipeline />);

    expect(screen.getByText("Ingestion Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/High-throughput document OCR processing/i)).toBeInTheDocument();
    expect(screen.getByText("Pipeline Settings")).toBeInTheDocument();
  });
});
