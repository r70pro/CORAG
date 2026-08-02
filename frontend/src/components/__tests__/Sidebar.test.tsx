import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Sidebar } from "../Sidebar";
import { startAnalysisModelSwitch } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchDockerStatus: jest.fn().mockResolvedValue({ status: "running", is_ready: true }),
  fetchDockerModels: jest.fn().mockResolvedValue({ models: ["allenai/olmOCR-2-7B-1025-FP8", "nvidia/Phi-4-reasoning-plus-NVFP4"] }),
  fetchAnalysisModelStatus: jest.fn().mockResolvedValue({
    configured_model: "Qwen/Qwen3.6-35B-A3B",
    served_model: "Qwen/Qwen3.6-35B-A3B",
    profiles: [
      { model: "Qwen/Qwen3.6-35B-A3B", display_name: "Qwen 3.6 35B A3B", revision: "q", context_length: 262144, dtype: "bfloat16", quantization: "none", estimated_load_seconds: 300, cache_complete: true, cache_error: "", snapshot: "/qwen" },
      { model: "google/gemma-4-31B-it", display_name: "Gemma 4 31B IT", revision: "g", context_length: 262144, dtype: "bfloat16", quantization: "none", estimated_load_seconds: 420, cache_complete: true, cache_error: "", snapshot: "/gemma" },
    ],
    operation: null,
  }),
  fetchAnalysisSwitchOperation: jest.fn(),
  startAnalysisModelSwitch: jest.fn().mockResolvedValue({ id: "operation-1", state: "queued", message: "queued", progress: 0 }),
  setVllmRoleRunning: jest.fn().mockResolvedValue({ success: true }),
  setStartupMode: jest.fn().mockResolvedValue({ success: true }),
  fetchSettings: jest.fn().mockResolvedValue({ hf_token: "test_token" }),
  startDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  stopDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  createDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  shutdownDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  updateSettings: jest.fn().mockResolvedValue({ success: true }),
}));


describe("Sidebar Component", () => {
  const mockOnSelectView = jest.fn();
  const mockOnRoleChange = jest.fn();
  const mockOnDensityChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders KIRAG branding and navigation items", async () => {
    render(
      <Sidebar
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeRole="Admin"
        onRoleChange={mockOnRoleChange}
        density="comfortable"
        onDensityChange={mockOnDensityChange}
      />
    );

    expect(screen.getByText(/IQ-RAG Client/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/Ingestion Pipeline/i)).toBeInTheDocument();
      expect(screen.getByText(/Case Dashboard/i)).toBeInTheDocument();
      expect(screen.getByText(/RAG Processing/i)).toBeInTheDocument();
    });
  });

  test("triggers onSelectView callback when a navigation item is clicked", async () => {
    render(
      <Sidebar
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeRole="Admin"
        onRoleChange={mockOnRoleChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Case Dashboard/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Case Dashboard/i));
    expect(mockOnSelectView).toHaveBeenCalledWith("dashboard");

    fireEvent.click(screen.getByText(/RAG Processing/i));
    expect(mockOnSelectView).toHaveBeenCalledWith("chat");
  });

  test("keeps OCR role fixed and starts only the guarded analysis switch", async () => {
    jest.spyOn(window, "confirm").mockReturnValue(true);
    render(
      <Sidebar
        currentView="ingestion"
        onSelectView={mockOnSelectView}
        activeRole="Admin"
        onRoleChange={mockOnRoleChange}
      />,
    );
    expect(await screen.findByText("allenai/olmOCR-2-7B-1025-FP8")).toBeInTheDocument();
    const profile = await screen.findByLabelText("Verified target profile");
    fireEvent.change(profile, { target: { value: "google/gemma-4-31B-it" } });
    fireEvent.click(screen.getByRole("button", { name: "Switch Analysis Model" }));
    await waitFor(() => expect(startAnalysisModelSwitch).toHaveBeenCalledWith("google/gemma-4-31B-it"));
    expect(screen.queryByRole("option", { name: "google/gemma-4-31B-it" })).not.toBeInTheDocument();
  });
});
