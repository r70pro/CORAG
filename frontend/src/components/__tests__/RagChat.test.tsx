import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { RagChat } from "../RagChat";

jest.mock("@/lib/api", () => ({
  triggerRagChatSSE: jest.fn(),
  startRagInfra: jest.fn().mockResolvedValue({ success: true }),
  stopRagInfra: jest.fn().mockResolvedValue({ success: true }),
  fetchRagInfraStatus: jest.fn().mockResolvedValue({
    postgres: "healthy",
    redis: "healthy",
    minio: "healthy",
    qdrant: "healthy",
  }),
  fetchCorpusStats: jest.fn().mockResolvedValue({
    indexed_runs: 1,
    indexed_documents: 2,
    total_chunks: 50,
    unique_authors: 3,
  }),
  fetchPipelineRuns: jest.fn().mockResolvedValue([]),
  indexRun: jest.fn().mockResolvedValue({ success: true }),
  indexAllRuns: jest.fn().mockResolvedValue({ success: true }),
  exportChatHistory: jest.fn().mockResolvedValue({ success: true }),
  updateSettings: jest.fn().mockResolvedValue({ success: true }),
}));

describe("RagChat Component", () => {
  test("renders RAG analysis chat interface and input area", async () => {
    render(<RagChat />);

    expect(screen.getAllByText(/RAG Processing Workstation/i).length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i)).toBeInTheDocument();
    });
  });

  test("updates query input on typing", async () => {
    render(<RagChat />);

    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Extract injury timeline" } });

    await waitFor(() => {
      expect(input).toHaveValue("Extract injury timeline");
    });
  });
});
