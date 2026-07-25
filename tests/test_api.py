import os
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient

# Disable docker commands and atexit hooks in testing
os.environ["TESTING"] = "true"

from api.main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "KIRAG API")
        self.assertIn("version", data)

    # ── Settings ──────────────────────────────────────────────────────────────

    @patch("api.routes.settings.load_settings")
    def test_get_settings(self, mock_load):
        mock_load.return_value = {"server_url": "http://localhost", "hf_token": "some-token"}
        response = self.client.get("/api/settings/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["server_url"], "http://localhost")
        self.assertEqual(data["hf_token"], "********")

    @patch("api.routes.settings.save_settings")
    @patch("api.routes.settings.load_settings")
    def test_update_settings(self, mock_load, mock_save):
        mock_load.return_value = {"server_url": "http://localhost"}
        mock_save.return_value = "Settings saved successfully."

        # Case 1: successful update
        response = self.client.put("/api/settings/", json={"server_url": "http://new-url"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        mock_save.assert_called_once()

        # Case 2: empty payload
        response = self.client.put("/api/settings/", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @patch("system_diagnostics.check_backing_services_data")
    @patch("system_diagnostics.get_gpu_metrics_data")
    @patch("settings_manager.load_settings")
    def test_diagnostics_health(self, mock_settings, mock_gpu, mock_backing):
        mock_settings.return_value = {"docker_port": 8000}
        mock_gpu.return_value = {
            "cuda_available": True,
            "gpu_name": "Tesla T4",
            "vram_used": 1000.0,
            "vram_total": 15000.0,
            "vram_pct": 6.6,
            "vram_free": 14000.0,
            "vram_reclaimable": 0.0,
            "processes": []
        }
        mock_backing.return_value = {
            "all_healthy": True,
            "services": {
                "postgres": {"is_up": True, "latency": 15.0},
                "redis": {"is_up": True, "latency": 2.0}
            }
        }

        response = self.client.get("/api/diagnostics/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["overall"], "healthy")
        self.assertEqual(data["gpu"]["gpu_name"], "Tesla T4")
        self.assertEqual(len(data["services"]), 2)

    @patch("system_diagnostics.get_gpu_metrics_data")
    def test_diagnostics_gpu(self, mock_gpu):
        mock_gpu.return_value = {"cuda_available": False, "gpu_name": "N/A"}
        response = self.client.get("/api/diagnostics/gpu")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["cuda_available"])
        self.assertEqual(data["gpu_name"], "N/A")

    @patch("system_diagnostics.check_backing_services_data")
    @patch("settings_manager.load_settings")
    def test_diagnostics_services(self, mock_settings, mock_backing):
        mock_settings.return_value = {}
        mock_backing.return_value = {"all_healthy": False}
        response = self.client.get("/api/diagnostics/services")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["all_healthy"])

    @patch("system_diagnostics.generate_diagnostic_report_file")
    @patch("settings_manager.load_settings")
    def test_diagnostics_report(self, mock_settings, mock_gen):
        mock_settings.return_value = {}
        
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as tmp:
            tmp.write("# Diagnostic Report")
            tmp_path = tmp.name

        mock_gen.return_value = tmp_path
        try:
            response = self.client.get("/api/diagnostics/report")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "# Diagnostic Report")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch("system_diagnostics.get_installed_models_data")
    def test_diagnostics_models(self, mock_models):
        mock_models.return_value = {
            "models": [
                {
                    "id": "allenai/olmOCR-2-7B-1025-FP8",
                    "name": "olmOCR-2-7B-1025-FP8",
                    "folder": "models--allenai--olmOCR-2-7B-1025-FP8",
                    "path": "/cache/models--allenai--olmOCR-2-7B-1025-FP8",
                    "size_bytes": 1000000,
                    "human_size": "1.00 MB",
                    "context_length": 15360,
                    "model_type": "Vision LLM",
                    "is_active": True,
                    "modified_at": "2026-07-25 12:00",
                }
            ],
            "total_count": 1,
            "total_size_bytes": 1000000,
            "total_human_size": "1.00 MB",
        }
        response = self.client.get("/api/diagnostics/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["models"][0]["id"], "allenai/olmOCR-2-7B-1025-FP8")

    @patch("system_diagnostics.delete_installed_models")
    def test_delete_diagnostics_models(self, mock_delete):
        mock_delete.return_value = (True, "Successfully deleted 1 model(s).", ["old/model"], 5000)
        response = self.client.request("DELETE", "/api/diagnostics/models", json={"model_ids": ["old/model"]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["deleted_models"], ["old/model"])



    # ── Docker ────────────────────────────────────────────────────────────────

    @patch("docker_manager.get_docker_status_str")
    @patch("settings_manager.load_settings")
    def test_docker_status(self, mock_settings, mock_status_str):
        mock_settings.return_value = {}
        mock_status_str.return_value = ("vLLM Running", "<span class='badge-success'>Ready</span>")

        response = self.client.get("/api/docker/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["message"], "vLLM Running")

    @patch("docker_manager.start_docker_container")
    def test_docker_start(self, mock_start):
        mock_start.return_value = (True, "Container started")
        response = self.client.post("/api/docker/start")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["message"], "Container started")

    @patch("docker_manager.stop_docker_container")
    def test_docker_stop(self, mock_stop):
        mock_stop.return_value = (True, "Container stopped")
        response = self.client.post("/api/docker/stop")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    @patch("docker_manager.create_docker_container")
    @patch("settings_manager.save_settings")
    @patch("settings_manager.load_settings")
    def test_docker_create(self, mock_load, mock_save, mock_create):
        mock_load.return_value = {}
        mock_create.return_value = (True, "Container created")

        payload = {
            "hf_token": "test-token",
            "port": 8000,
            "model": "test-model",
            "gpu_mem": 0.8,
            "max_model_len": 4096
        }
        response = self.client.post("/api/docker/create", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        mock_save.assert_called_once()

    @patch("docker_manager.shutdown_docker_container")
    def test_docker_shutdown(self, mock_shutdown):
        mock_shutdown.return_value = (True, "Container removed")
        response = self.client.post("/api/docker/shutdown")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    # ── Documents ─────────────────────────────────────────────────────────────

    @patch("settings_manager.get_available_runs")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_documents_runs(self, mock_listdir, mock_exists, mock_runs):
        mock_runs.return_value = [("run1 (3 files)", "/path/to/run1")]
        mock_exists.return_value = True
        mock_listdir.return_value = ["file1.md", "file2.md"]

        response = self.client.get("/api/documents/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["run_name"], "run1")
        self.assertEqual(data[0]["file_count"], 2)

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_list_run_files(self, mock_listdir, mock_exists, mock_isdir):
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_listdir.return_value = ["a.md", "b.md"]

        response = self.client.get("/api/documents/runs/run1/files")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, ["a.md", "b.md"])

        # Run not found
        mock_isdir.return_value = False
        response = self.client.get("/api/documents/runs/invalid-run/files")
        self.assertEqual(response.status_code, 404)

    @patch("os.path.isfile")
    def test_get_markdown(self, mock_isfile):
        mock_isfile.return_value = True
        with patch("builtins.open", mock_open(read_data="Markdown content")):
            response = self.client.get("/api/documents/runs/run1/markdown/file.md")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "Markdown content")

        # Bad path (traversal) - '..' inside the parameter
        response = self.client.get("/api/documents/runs/run1/markdown/..file.md")
        self.assertEqual(response.status_code, 400)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    @patch("settings_manager.get_available_runs")
    def test_pipeline_runs(self, mock_runs):
        mock_runs.return_value = [("run_abc (5 files)", "/path/abc")]
        response = self.client.get("/api/pipeline/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["file_count"], 5)

    @patch("pipeline_manager.stop_processing")
    def test_pipeline_stop(self, mock_stop):
        mock_stop.return_value = "Stop request sent for run1"
        response = self.client.post("/api/pipeline/stop/run1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    @patch("process_state.active_runs_lock")
    @patch("process_state.active_runs")
    def test_pipeline_status(self, mock_active_runs, mock_lock):
        # Case 1: Run not found
        mock_active_runs.get.return_value = None
        response = self.client.get("/api/pipeline/status/run1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run1")
        self.assertEqual(response.json()["status"], "unknown")

        # Case 2: Running
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_active_runs.get.return_value = {"proc": mock_proc, "log_tail": "running logs"}
        response = self.client.get("/api/pipeline/status/run1")
        self.assertEqual(response.json()["status"], "running")

    @patch("pipeline_manager.process_pdfs")
    @patch("os.path.isfile")
    def test_pipeline_start(self, mock_isfile, mock_process):
        mock_isfile.return_value = True
        mock_process.return_value = [("log", "badge", "progress", None, None, None, None, None, None, "run_id", "status", "manifest", "stop")]

        payload = {
            "file_paths": ["/path/doc.pdf"],
            "server_url": "http://localhost",
            "model_name": "test-model"
        }
        response = self.client.post("/api/pipeline/start", json=payload)
        self.assertEqual(response.status_code, 200)
        
        # Read the event stream
        lines = [line for line in response.iter_lines() if line]
        self.assertTrue(any("run_id" in line for line in lines))

    def test_pipeline_upload(self):
        files = [("files", ("test.pdf", b"%PDF-1.4 test content", "application/pdf"))]
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["file_paths"]) == 1)

    def test_pipeline_upload_with_spaces(self):
        files = [("files", ("test file with spaces.pdf", b"%PDF-1.4 test content", "application/pdf"))]
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["file_paths"][0].endswith("test file with spaces.pdf"))

    def test_pipeline_upload_single_file_param(self):
        files = {"files": ("test_single.pdf", b"%PDF-1.4 test content", "application/pdf")}
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["file_paths"][0].endswith("test_single.pdf"))

    def test_pipeline_upload_multiple_files(self):
        files = [
            ("files", ("file1.pdf", b"%PDF-1.4 content 1", "application/pdf")),
            ("files", ("file2.pdf", b"%PDF-1.4 content 2", "application/pdf")),
        ]
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["file_paths"]), 2)

    def test_pipeline_upload_empty(self):
        response = self.client.post("/api/pipeline/upload")
        self.assertEqual(response.status_code, 400)

    # ── RAG ───────────────────────────────────────────────────────────────────

    @patch("rag_infra_manager.start_and_init_rag")
    def test_rag_infra_start(self, mock_start):
        mock_start.return_value = (True, "Infra started")
        response = self.client.post("/api/rag/infra/start")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("rag_infra_manager.stop_rag_infrastructure")
    def test_rag_infra_stop(self, mock_stop):
        mock_stop.return_value = (True, "Infra stopped")
        response = self.client.post("/api/rag/infra/stop")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("rag_infra_manager.get_rag_service_status")
    def test_rag_infra_status(self, mock_status):
        mock_status.return_value = {"postgres": "up", "redis": "up", "qdrant": "up", "minio": "up"}
        response = self.client.get("/api/rag/infra/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["postgres"], "up")

    @patch("rag.db.get_corpus_stats")
    @patch("rag.embedding.get_collection_info")
    def test_rag_corpus_stats(self, mock_coll, mock_stats):
        mock_stats.return_value = {"indexed_runs": 2, "total_chunks": 100}
        mock_coll.return_value = {"points_count": 500}
        response = self.client.get("/api/rag/corpus/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["indexed_runs"], 2)
        self.assertEqual(data["vectors_count"], 500)

    @patch("rag.db.get_indexed_runs")
    def test_rag_corpus_cases(self, mock_runs):
        mock_runs.return_value = [{"run_id": "case1", "display_name": "Case One"}]
        response = self.client.get("/api/rag/corpus/cases")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["run_id"], "case1")

    @patch("indexing_service.CorpusIndexingService.index_run")
    def test_rag_index_run(self, mock_index):
        mock_index.return_value = ["Scanning...", "✅ Done"]
        response = self.client.post("/api/rag/index", json={"run_dir": "/path/run"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("indexing_service.CorpusIndexingService.index_all_runs")
    def test_rag_index_all_runs(self, mock_index):
        mock_index.return_value = ["Scanning...", "✅ Done"]
        response = self.client.post("/api/rag/index-all")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("indexing_service.CorpusIndexingService.add_markdown_to_case")
    def test_rag_upload_markdown(self, mock_add):
        mock_add.return_value = ["✅ Done"]
        
        # Send a mock file upload
        files = {"files": ("file.md", b"# Markdown content", "text/markdown")}
        data = {"case_option": "new", "new_case_name": "TestCase"}
        response = self.client.post("/api/rag/upload-markdown", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("rag.analyzer.analyze")
    def test_rag_query(self, mock_analyze):
        mock_analyze.return_value = ["Answer ", "chunk"]
        
        # Case 1: Streaming
        payload = {"query": "test query", "stream": True}
        response = self.client.post("/api/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        lines = [line for line in response.iter_lines() if line]
        self.assertTrue(any("Answer" in line for line in lines))

        # Case 2: Non-streaming
        payload = {"query": "test query", "stream": False}
        response = self.client.post("/api/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Answer chunk")

    # ── Authentication & Security ─────────────────────────────────────────────

    def test_api_key_authentication(self):
        with patch.dict(os.environ, {"KIRAG_API_KEY": "test-secret-key"}):
            # Test without API key header -> should fail with 401
            res = self.client.get("/")
            self.assertEqual(res.status_code, 401)

            # Test with invalid API key header -> 401
            res = self.client.get("/", headers={"X-API-Key": "wrong-key"})
            self.assertEqual(res.status_code, 401)

            # Test with valid X-API-Key header -> 200
            res = self.client.get("/", headers={"X-API-Key": "test-secret-key"})
            self.assertEqual(res.status_code, 200)

            # Test with valid Bearer token -> 200
            res = self.client.get("/", headers={"Authorization": "Bearer test-secret-key"})
            self.assertEqual(res.status_code, 200)

    def test_cors_headers(self):
        # OPTIONS preflight request from allowed origin
        response = self.client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    # ── Consolidated Phase 1 Core Endpoints ─────────────────────────────────────

    @patch("system_diagnostics.check_backing_services_data")
    @patch("system_diagnostics.get_gpu_metrics_data")
    def test_consolidated_health(self, mock_gpu, mock_backing):
        mock_backing.return_value = {"postgres": "healthy"}
        mock_gpu.return_value = {"utilization": "5%"}
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    @patch("rag.db.get_corpus_stats")
    @patch("rag.db.get_indexed_runs")
    @patch("rag.embedding.get_collection_info")
    def test_consolidated_case_summary(self, mock_qdrant, mock_runs, mock_stats):
        mock_stats.return_value = {"indexed_runs": 1, "total_chunks": 10}
        mock_runs.return_value = [{"run_id": "r1", "display_name": "Run 1", "created_at": "2026-07-22"}]
        mock_qdrant.return_value = {"points_count": 10, "status": "green"}

        res = self.client.get("/api/case-summary")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["stats"]["indexed_runs"], 1)


if __name__ == "__main__":
    unittest.main()

