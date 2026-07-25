import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { SystemDiagnostics } from "../SystemDiagnostics";
import { fetchSystemHealth, executeCleanup } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchSystemHealth: jest.fn().mockResolvedValue({
    status: "healthy",
    services: [
      { name: "PostgreSQL 16", extra_info: "port: 5432", is_up: true, latency_ms: 2.4 }
    ],
    gpu: { vram_pct: 42, vram_used_mb: 10240 }
  }),
  fetchDockerStatus: jest.fn().mockResolvedValue({ status: "running", is_ready: true }),
  startDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  stopDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  createDockerContainer: jest.fn().mockResolvedValue({ success: true }),
  executeCleanup: jest.fn().mockResolvedValue({ success: true }),
  fetchInstalledModels: jest.fn().mockResolvedValue({ models: [], total_count: 0, total_size_bytes: 0, total_human_size: "0 B" }),
  deleteInstalledModels: jest.fn().mockResolvedValue({ success: true, message: "Deleted" }),
  API_BASE_URL: "http://127.0.0.1:8001",
}));

const mockFetchSystemHealth = fetchSystemHealth as jest.Mock;
const mockExecuteCleanup = executeCleanup as jest.Mock;

describe("SystemDiagnostics Component", () => {
  test("renders system health & diagnostics dashboard", async () => {
    render(<SystemDiagnostics />);

    await waitFor(() => {
      expect(screen.getByText("System Diagnostics")).toBeInTheDocument();
      expect(screen.getByText(/Infrastructure health telemetry/i)).toBeInTheDocument();
    });
  });

  test("handles object/dictionary health.services gracefully without map exception", async () => {
    mockFetchSystemHealth.mockResolvedValueOnce({
      status: "healthy",
      services: {
        postgres: { is_up: true, latency: 2.4, extra_info: "port: 5432" }
      },
      gpu: { vram_pct: 50, vram_used_mb: 12000 }
    });

    render(<SystemDiagnostics />);

    await waitFor(() => {
      expect(screen.getByText(/POSTGRES/i)).toBeInTheDocument();
    });
  });

  test("handles cleanup network errors gracefully", async () => {
    mockExecuteCleanup.mockResolvedValueOnce({
      success: false,
      message: "Network error connecting to API server at http://127.0.0.1:8001: TypeError: NetworkError",
    });

    render(<SystemDiagnostics />);

    await waitFor(() => {
      expect(screen.getByText("System Diagnostics")).toBeInTheDocument();
    });

    const cleanButton = screen.getByRole("button", { name: /Clean Selected Components/i });
    fireEvent.click(cleanButton);

    await waitFor(() => {
      expect(screen.getAllByText(/Network error connecting to API server/i).length).toBeGreaterThan(0);
    });
  });
});
