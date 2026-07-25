import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

os.environ["TESTING"] = "true"

from api.server import app


class TestAPIServer(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("system_diagnostics.check_backing_services_data")
    @patch("system_diagnostics.get_gpu_metrics_data")
    def test_health_endpoint(self, mock_gpu, mock_backing):
        mock_backing.return_value = {"postgres": "healthy", "redis": "healthy"}
        mock_gpu.return_value = {"utilization": "10%"}

        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        postgres_service = next((s for s in data["services"] if s["name"] == "postgres"), None)
        self.assertIsNotNone(postgres_service)
        self.assertTrue(postgres_service["is_up"])

    @patch("rag.db.get_corpus_stats")
    @patch("rag.db.get_runs_with_stats")
    @patch("rag.db.get_indexed_runs")
    @patch("rag.embedding.get_collection_info")
    def test_case_summary_endpoint(self, mock_qdrant, mock_runs, mock_runs_with_stats, mock_stats):
        mock_stats.return_value = {"indexed_runs": 2, "total_chunks": 150}
        mock_runs.return_value = [{"run_id": "run_01", "display_name": "Run 01", "created_at": "2026-07-22"}]
        mock_runs_with_stats.return_value = [{"run_id": "run_01", "display_name": "Run 01", "created_at": "2026-07-22"}]
        mock_qdrant.return_value = {"points_count": 150, "status": "green"}

        response = self.client.get("/api/case-summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stats"]["indexed_runs"], 2)
        self.assertEqual(len(data["indexed_cases"]), 1)
        self.assertEqual(data["vector_store"]["points_count"], 150)

    @patch("rag.analyzer.analyze")
    def test_chat_endpoint(self, mock_analyze):

        mock_analyze.return_value = ["Hello ", "world"]

        response = self.client.post(
            "/api/chat",
            json={
                "query": "Test query",
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "Hello world")

    @patch("cleanup_manager.perform_reset_cleanup")
    def test_diagnostics_cleanup_endpoint(self, mock_cleanup):
        mock_cleanup.return_value = "### Cleanup Summary\n\nSpace freed: 10 KB"
        response = self.client.post("/api/diagnostics/cleanup", json={"components": ["pycache"]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("Cleanup Summary", data["message"])
