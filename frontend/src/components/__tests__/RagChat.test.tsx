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
  deleteRagChatHistory: jest.fn().mockResolvedValue({ success: true }),
}));

describe("RagChat Component", () => {
  beforeEach(() => {
    jest.mocked(triggerRagChatSSE).mockReset();
    localStorage.clear();
  });

  test("restores persisted chat messages after a browser refresh", async () => {
    localStorage.setItem("kirag_rag_chat_threads_v1", JSON.stringify([{
      id: "persisted-thread",
      title: "Prior causation analysis",
      createdAt: "2026-08-01T00:00:00.000Z",
      updatedAt: "2026-08-01T01:00:00.000Z",
      analysisMode: "🧬 Causation Analysis",
      activeCase: "case-1",
      messages: [{
        id: "answer-1",
        sender: "bot",
        text: "This answer survived the refresh.",
        timestamp: "11:20 AM",
      }],
    }]));

    render(<RagChat />);

    expect(await screen.findByText("This answer survived the refresh.")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Chat thread" })).toHaveValue("persisted-thread");
  });

  test("allows a persisted chat thread to be deleted", async () => {
    localStorage.setItem("kirag_rag_chat_threads_v1", JSON.stringify([{
      id: "delete-me",
      title: "Delete me",
      createdAt: "2026-08-01T00:00:00.000Z",
      updatedAt: "2026-08-01T01:00:00.000Z",
      analysisMode: "💬 Free Q&A",
      activeCase: "",
      messages: [],
    }]));
    jest.spyOn(window, "confirm").mockReturnValue(true);
    render(<RagChat />);
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Chat thread" })).toHaveValue("delete-me"));

    fireEvent.click(screen.getByRole("button", { name: "Delete Chat Thread" }));

    const stored = JSON.parse(localStorage.getItem("kirag_rag_chat_threads_v1") || "[]");
    expect(stored).toHaveLength(1);
    expect(stored[0].id).not.toBe("delete-me");
    jest.restoreAllMocks();
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
      "🌐 General Knowledge",
      "💬 Free Q&A",
      "📋 Timeline",
      "🏥 Injury Summary",
      "🔍 Inconsistency Finder",
      "💊 Medication Tracker",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });

  test("General Knowledge sends no case, filters, or reranker settings", async () => {
    render(<RagChat />);

    fireEvent.click(screen.getByRole("button", { name: "🌐 General Knowledge" }));
    const input = screen.getByPlaceholderText(/Ask a general-knowledge question/i);
    fireEvent.change(input, { target: { value: "What is Markdown?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Query" }));

    await waitFor(() => {
      expect(triggerRagChatSSE).toHaveBeenCalledWith(
        expect.objectContaining({
          query: "What is Markdown?",
          mode: "🌐 General Knowledge",
          case_id: undefined,
          doc_type: undefined,
          author: undefined,
          date_from: undefined,
          date_to: undefined,
          use_reranker: false,
          reranker_model: undefined,
          reranker_device: undefined,
        }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      );
    });
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
        expect.any(Function),
        expect.any(Function),
      );
    });
  });

  test("forwards metadata filters to the next RAG request", async () => {
    render(<RagChat />);
    fireEvent.change(screen.getByLabelText("Document Type"), { target: { value: "radiology_report" } });
    fireEvent.change(screen.getByLabelText("Author"), { target: { value: "Capital Radiology" } });
    fireEvent.change(screen.getByLabelText("Date From"), { target: { value: "2024-01-01" } });
    fireEvent.change(screen.getByLabelText("Date To"), { target: { value: "2024-12-31" } });
    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Find imaging" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Query" }));

    await waitFor(() => {
      expect(triggerRagChatSSE).toHaveBeenCalledWith(
        expect.objectContaining({
          doc_type: "radiology_report",
          author: "Capital Radiology",
          date_from: "2024-01-01",
          date_to: "2024-12-31",
        }),
        expect.any(Function),
        expect.any(Function),
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

  test("shows safe pipeline activity without exposing model reasoning", async () => {
    jest.mocked(triggerRagChatSSE).mockImplementation(
      (_payload, onChunk, _onError, _onComplete, onStatus) => {
        onStatus?.({
          type: "status",
          stage: "retrieving",
          message: "Searching indexed medical records…",
          progress: 0.1,
        });
        return { cancel: jest.fn() };
      },
    );
    render(<RagChat />);

    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Build a timeline" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Query" }));

    expect(await screen.findByRole("status")).toHaveTextContent(
      "Searching indexed medical records…",
    );
    expect(screen.queryByText(/thinking process/i)).not.toBeInTheDocument();
  });

  test("exposes Gradio-equivalent stop and clear controls", async () => {
    const cancel = jest.fn();
    jest.mocked(triggerRagChatSSE).mockReturnValue({ cancel });
    render(<RagChat />);

    const stop = screen.getByRole("button", { name: "Stop generating" });
    expect(stop).toBeVisible();
    expect(stop).toBeDisabled();

    const input = screen.getByPlaceholderText(/Ask a medicolegal question or request an audit.../i);
    fireEvent.change(input, { target: { value: "Build a timeline" } });
    fireEvent.click(screen.getByRole("button", { name: "Send Query" }));

    await waitFor(() => expect(stop).toBeEnabled());
    fireEvent.click(stop);
    expect(cancel).toHaveBeenCalledWith("RAG generation stopped by user");
    expect(screen.getByText(/stopped by user/i)).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: /Clear Chat/i })[0]);
    expect(screen.queryByText("Build a timeline")).not.toBeInTheDocument();
  });
});
