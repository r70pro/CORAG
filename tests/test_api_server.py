import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

os.environ["TESTING"] = "true"

from api.server import app


class TestAPIServer(unittest.TestCase):

    def setUp(self):
        self.auth_environment = patch.dict(
            os.environ,
            {
                "KIRAG_API_KEY": "test-api-key",
                "KIRAG_ADMIN_API_KEY": "test-admin-key",
            },
        )
        self.auth_environment.start()
        self.client = TestClient(
            app,
            headers={
                "X-API-Key": "test-api-key",
                "X-Admin-API-Key": "test-admin-key",
            },
        )

    def tearDown(self):
        self.client.close()
        self.auth_environment.stop()

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
    @patch("rag.embedding.get_collection_info")
    @patch("rag.metadata_helper.get_all_cases_metadata")
    @patch("rag.metadata_helper.get_case_timeline")
    def test_case_summary_endpoint(
        self, mock_timeline, mock_metadata, mock_qdrant, mock_runs, mock_stats
    ):
        mock_stats.return_value = {"indexed_runs": 2, "total_chunks": 150}
        mock_runs.return_value = [
            {
                "run_id": "run_01",
                "created_at": "2026-07-22",
                "total_documents": 4,
                "total_chunks": 150,
            }
        ]
        mock_qdrant.return_value = {"points_count": 150, "status": "green"}
        mock_metadata.return_value = {"run_01": {"names": ["Run 01"]}}
        mock_timeline.return_value = []

        response = self.client.get("/api/case-summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stats"]["indexed_runs"], 2)
        self.assertEqual(len(data["indexed_cases"]), 1)
        self.assertEqual(data["vector_store"]["points_count"], 150)

    @patch("rag.embedding.get_collection_info", side_effect=RuntimeError("qdrant unavailable"))
    @patch("rag.metadata_helper.get_all_cases_metadata", return_value={})
    @patch("rag.db.get_runs_with_stats", return_value=[])
    @patch("rag.db.get_corpus_stats", return_value={})
    def test_case_summary_service_failure_is_non_200(self, *_mocks):
        response = self.client.get("/api/case-summary")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")

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
