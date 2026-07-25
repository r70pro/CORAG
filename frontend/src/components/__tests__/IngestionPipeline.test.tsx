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
});
