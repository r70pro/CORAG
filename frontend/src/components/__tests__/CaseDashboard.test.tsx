import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { CaseDashboard } from "../CaseDashboard";
import { fetchCaseSummary, deleteCases } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchCaseSummary: jest.fn().mockResolvedValue({
    stats: { indexed_runs: 2, total_documents: 5, total_chunks: 120, unique_authors: 8 },
    indexed_cases: [
      { run_id: "run_001", display_name: "Souki, Issa", created_at: "2026-07-22" }
    ],
    vector_store: { points_count: 120, status: "green" }
  }),
  deleteCases: jest.fn().mockResolvedValue({ success: true }),
}));

const mockFetchCaseSummary = fetchCaseSummary as jest.Mock;
const mockDeleteCases = deleteCases as jest.Mock;

describe("CaseDashboard Component", () => {
  test("renders dashboard header and case list", async () => {
    render(<CaseDashboard />);

    expect(screen.getByText("Case Dashboard")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Souki, Issa")).toBeInTheDocument();
    });
  });

  test("allows searching timeline events by physician or finding", async () => {
    render(<CaseDashboard />);

    await waitFor(() => {
      expect(screen.getByText("Souki, Issa")).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText("Search timeline events, physicians, or findings...");
    fireEvent.change(searchInput, { target: { value: "Dr. Eugene" } });

    expect(searchInput).toHaveValue("Dr. Eugene");
  });
  test("handles Delete All Cases and displays empty state", async () => {
    window.confirm = jest.fn().mockReturnValue(true);

    render(<CaseDashboard />);

    await waitFor(() => {
      expect(screen.getByText("Souki, Issa")).toBeInTheDocument();
    });

    mockFetchCaseSummary.mockResolvedValueOnce({
      stats: { indexed_runs: 0, total_documents: 0, total_chunks: 0, unique_authors: 0 },
      indexed_cases: [],
      vector_store: { points_count: 0, status: "green" }
    });

    const deleteAllBtn = screen.getByText("Delete All Cases");
    fireEvent.click(deleteAllBtn);

    await waitFor(() => {
      expect(mockDeleteCases).toHaveBeenCalledWith([], true);
      expect(screen.getByText("No Indexed Cases Found")).toBeInTheDocument();
    });
  });
});
