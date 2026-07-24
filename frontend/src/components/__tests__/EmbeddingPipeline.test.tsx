import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { EmbeddingPipeline } from "../EmbeddingPipeline";

jest.mock("@/lib/api", () => ({
  fetchEmbeddingTelemetry: jest.fn().mockResolvedValue({
    active_device: "CUDA GPU",
    device_target: "auto",
    qdrant_points: 100,
    collection_name: "cases",
    vector_dim: 1024,
    redis_cached_count: 50,
  }),
  saveEmbeddingConfig: jest.fn().mockResolvedValue({ success: true }),
  purgeVectorCache: jest.fn().mockResolvedValue({ success: true }),
  fetchPipelineRuns: jest.fn().mockResolvedValue([]),
  indexRun: jest.fn().mockResolvedValue({ success: true }),
  indexAllRuns: jest.fn().mockResolvedValue({ success: true }),
  uploadMarkdownFiles: jest.fn().mockResolvedValue({ success: true }),
}));

describe("EmbeddingPipeline Component", () => {
  test("renders embedding configuration header and parameters", async () => {
    render(<EmbeddingPipeline />);

    expect(screen.getByText("Embedding Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/Vector embedding generation/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/Qdrant Points Count/i)).toBeInTheDocument();
    });
  });
});
