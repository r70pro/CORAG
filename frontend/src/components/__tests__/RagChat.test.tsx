import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { RagChat } from "../RagChat";
import { triggerRagChatSSE } from "@/lib/api";

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
  beforeEach(() => {
    jest.mocked(triggerRagChatSSE).mockReset();
  });

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

  test("does not seed the chat with fabricated verification details", () => {
    render(<RagChat />);

    expect(screen.queryByText(/Gavin Weekes/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/2024AL0008570/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Pages 1-3/i)).not.toBeInTheDocument();
  });

  test("exposes every documented analysis mode", () => {
    render(<RagChat />);

    for (const name of [
      "💬 Free Q&A",
      "📋 Timeline",
      "🏥 Injury Summary",
      "🔍 Inconsistency Finder",
      "💊 Medication Tracker",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });

  test("omits empty optional filters from a RAG request", async () => {
    render(<RagChat />);

    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Build a timeline" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Query" }));

    await waitFor(() => {
      expect(triggerRagChatSSE).toHaveBeenCalledWith(
        expect.objectContaining({
          query: "Build a timeline",
          case_id: undefined,
          doc_type: undefined,
          author: undefined,
          date_from: undefined,
          date_to: undefined,
        }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      );
    });
  });

  test("flushes streamed text and leaves the cleared composer settled when DONE completes", async () => {
    jest.mocked(triggerRagChatSSE).mockImplementation(
      (_payload, onChunk, _onError, onComplete) => {
        onChunk("Timeline complete");
        onComplete();
        return { cancel: jest.fn() };
      },
    );
    render(<RagChat />);

    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Build a timeline" } });
    const send = screen.getByRole("button", { name: "Send Query" });
    fireEvent.click(send);

    await waitFor(() => {
      expect(screen.getByText("Timeline complete")).toBeInTheDocument();
      expect(send).toHaveAccessibleName("Send Query");
      expect(send).toBeDisabled();
    });
  });
});
