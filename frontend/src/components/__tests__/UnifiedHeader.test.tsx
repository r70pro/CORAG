import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { UnifiedHeader } from "../UnifiedHeader";
import { shutdownApp } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchSystemHealth: jest.fn().mockResolvedValue({
    status: "healthy",
    all_healthy: true,
    services: [
      { name: "postgres", is_up: true, latency_ms: 1.2 },
      { name: "redis", is_up: true, latency_ms: 0.8 },
      { name: "minio", is_up: true, latency_ms: 2.1 },
      { name: "qdrant", is_up: true, latency_ms: 3.4 },
      { name: "vllm", is_up: true, latency_ms: 12.5 },
    ],
    failed_services: [],
    vllm_model: "allenai/olmOCR-2-7B-1025-FP8",
    vllm_progress: null,
    gpu: {
      name: "NVIDIA GeForce RTX 4090",
      vram_used: 4.2,
      vram_total: 24.0,
      vram_pct: 17.5,
      cuda_available: true,
    },
  }),
  fetchCaseSummary: jest.fn().mockResolvedValue({
    indexed_cases: [
      { run_id: "souki_enclosures", display_name: "Souki Enclosures" },
      { run_id: "test_case_02", display_name: "Test Case 02" },
    ],
    stats: { total_chunks: 54, unique_authors: 4 },
  }),
  fetchDocumentRuns: jest.fn().mockResolvedValue(["souki_enclosures", "test_case_02"]),
  shutdownApp: jest.fn().mockResolvedValue({ success: true, message: "Shutdown accepted." }),
}));

describe("UnifiedHeader Component", () => {
  const mockOnSelectView = jest.fn();
  const mockOnSelectCase = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders brand logo, view nav buttons, system health badge, and VRAM gauge", async () => {
    render(
      <UnifiedHeader
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeCaseId="souki_enclosures"
        onSelectCase={mockOnSelectCase}
      />
    );

    expect(screen.getByText("KIRAG")).toBeInTheDocument();
    expect(screen.getByText("Quick Jump:")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("✓ System Healthy")).toBeInTheDocument();
      expect(screen.getByText("allenai/olmOCR-2-7B-1025-FP8")).toBeInTheDocument();
      expect(screen.getByText("● Best suited for PDF conversion")).toBeInTheDocument();
    });
  });

  test("handles view navigation clicks correctly", async () => {
    render(
      <UnifiedHeader
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeCaseId="souki_enclosures"
        onSelectCase={mockOnSelectCase}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Dashboard")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Dashboard"));
    expect(mockOnSelectView).toHaveBeenCalledWith("dashboard");

    fireEvent.click(screen.getByText("RAG Chat"));
    expect(mockOnSelectView).toHaveBeenCalledWith("chat");
  });

  test("handles quick jump case switcher selection", async () => {
    render(
      <UnifiedHeader
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeCaseId="souki_enclosures"
        onSelectCase={mockOnSelectCase}
      />
    );

    await waitFor(() => {
      const select = screen.getByTitle("Switch Active Case Context") as HTMLSelectElement;
      expect(select).toBeInTheDocument();
    });

    const select = screen.getByTitle("Switch Active Case Context");
    fireEvent.change(select, { target: { value: "test_case_02" } });
    expect(mockOnSelectCase).toHaveBeenCalledWith("test_case_02");
  });

  test("opens health breakdown popover when health badge is clicked", async () => {
    render(
      <UnifiedHeader
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeCaseId="souki_enclosures"
        onSelectCase={mockOnSelectCase}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("✓ System Healthy")).toBeInTheDocument();
    });

    const healthBtn = screen.getByText("✓ System Healthy");
    fireEvent.click(healthBtn);

    await waitFor(() => {
      expect(screen.getByText("Services Health Matrix")).toBeInTheDocument();
      expect(screen.getByText("postgres")).toBeInTheDocument();
      expect(screen.getByText("qdrant")).toBeInTheDocument();
    });
  });

  test("requires typed confirmation before requesting host shutdown", async () => {
    render(
      <UnifiedHeader
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeCaseId="souki_enclosures"
        onSelectCase={mockOnSelectCase}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Shut down KIRAG" }));
    const confirmButton = screen.getByRole("button", { name: "Stop KIRAG" });
    expect(confirmButton).toBeDisabled();

    fireEvent.change(screen.getByLabelText("Type SHUTDOWN to confirm"), { target: { value: "SHUTDOWN" } });
    fireEvent.click(confirmButton);

    await waitFor(() => expect(shutdownApp).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("Shutdown accepted.")).toBeInTheDocument();
  });
});
