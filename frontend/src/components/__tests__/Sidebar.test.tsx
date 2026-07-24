import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Sidebar } from "../Sidebar";

jest.mock("@/lib/api", () => ({
  fetchDockerStatus: jest.fn().mockResolvedValue({ status: "running", is_ready: true }),
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
});
