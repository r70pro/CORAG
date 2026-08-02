import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter

# Disable docker commands and atexit hooks in testing
os.environ["TESTING"] = "true"

from api.main import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.workspace = os.path.join(self.temp_directory.name, "workspace")
        self.run_dir = os.path.join(self.workspace, "run_case")
        self.markdown_dir = os.path.join(self.run_dir, "markdown", "inputs")
        self.upload_dir = os.path.join(self.workspace, "uploads")
        os.makedirs(self.markdown_dir)
        os.makedirs(self.upload_dir)

        pdf_buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf_buffer)
        self.pdf_bytes = pdf_buffer.getvalue()

        self.workspace_patchers = [
            patch("settings_manager.WORKSPACE_DIR", self.workspace),
            patch("api.routes.documents.WORKSPACE_DIR", self.workspace),
        ]
        for patcher in self.workspace_patchers:
            patcher.start()
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
        for patcher in reversed(self.workspace_patchers):
            patcher.stop()
        self.temp_directory.cleanup()

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "KIRAG API")
        self.assertIn("version", data)

    def test_host_shutdown_requires_confirmation_and_creates_trigger(self):
        marker = os.path.join(self.temp_directory.name, "shutdown-request")
        with patch("api.routes.system.SHUTDOWN_REQUEST_PATH", marker):
            rejected = self.client.post(
                "/api/system/shutdown", json={"confirmation": "shutdown"}
            )
            self.assertEqual(rejected.status_code, 422)
            self.assertFalse(os.path.exists(marker))

            accepted = self.client.post(
                "/api/system/shutdown", json={"confirmation": "SHUTDOWN"}
            )
            self.assertEqual(accepted.status_code, 202)
            self.assertTrue(accepted.json()["success"])
            with open(marker, encoding="ascii") as request_file:
                self.assertEqual(request_file.read(), "shutdown\n")

    # ── Settings ──────────────────────────────────────────────────────────────

    @patch("api.routes.settings.load_settings")
    def test_get_settings(self, mock_load):
        mock_load.return_value = {"server_url": "http://localhost", "hf_token": "some-token"}
        response = self.client.get("/api/settings/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["server_url"], "http://localhost")
        self.assertEqual(data["hf_token"], "********")
        mock_load.assert_called_once_with(include_env_secrets=False)

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
        mock_load.assert_called_with(include_env_secrets=False)

        # Case 2: empty payload
        response = self.client.put("/api/settings/", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "bad_request")

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
            "processes": [],
        }
        mock_backing.return_value = {
            "all_healthy": True,
            "services": {
                "postgres": {"is_up": True, "latency": 15.0},
                "redis": {"is_up": True, "latency": 2.0},
            },
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

        export_dir = os.path.join(self.workspace, "exports")
        os.makedirs(export_dir)
        tmp_path = os.path.join(export_dir, "diagnostic_report.md")
        with open(tmp_path, "w", encoding="utf-8") as report:
            report.write("# Diagnostic Report")

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
        response = self.client.request(
            "DELETE", "/api/diagnostics/models", json={"model_ids": ["old/model"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["deleted_models"], ["old/model"])

        mock_delete.return_value = (
            False,
            "Successfully deleted 1 model(s). Failed to delete 1 model(s): denied.",
            ["old/model"],
            5000,
        )
        response = self.client.request(
            "DELETE", "/api/diagnostics/models", json={"model_ids": ["old/model", "bad/model"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Failed to delete", data["message"])
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

    @patch("docker_manager.get_docker_status_str")
    @patch("settings_manager.load_settings")
    def test_docker_status_reports_foreign_container(self, mock_settings, mock_status_str):
        mock_settings.return_value = {}
        mock_status_str.return_value = (
            "foreign",
            "<span class='badge-failed'>Docker: Foreign Container</span>",
        )

        response = self.client.get("/api/docker/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "foreign")

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
            "model": "allenai/olmOCR-2-7B-1025-FP8",
            "gpu_mem": 0.8,
            "max_model_len": 4096,
        }
        response = self.client.post("/api/docker/create", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        mock_save.assert_called_once()

    @patch("docker_manager.create_docker_container")
    @patch("settings_manager.save_settings")
    @patch("settings_manager.load_settings")
    def test_docker_create_does_not_persist_environment_token(
        self, mock_load, mock_save, mock_create
    ):
        mock_load.return_value = {"hf_token": ""}
        mock_create.return_value = (True, "Container created")

        with patch.dict(os.environ, {"HF_TOKEN": "environment-only-token"}):
            response = self.client.post(
                "/api/docker/create",
                json={"model": "allenai/olmOCR-2-7B-1025-FP8", "port": 8000, "max_model_len": 4096},
            )

        self.assertEqual(response.status_code, 200)
        mock_load.assert_called_once_with(include_env_secrets=False)
        self.assertEqual(mock_create.call_args.args[0], "environment-only-token")
        saved_settings = mock_save.call_args.args[0]
        self.assertEqual(saved_settings.get("hf_token", ""), "")

    @patch("docker_manager.create_docker_container")
    @patch("settings_manager.save_settings")
    @patch("settings_manager.load_settings")
    def test_docker_create_rejects_analysis_models_for_ocr_role(
        self, mock_load, mock_save, mock_create
    ):
        mock_load.return_value = {
            "model_name": "allenai/olmOCR-2-7B-1025-FP8",
            "server_url": "http://localhost:8000/v1",
        }
        mock_create.return_value = (True, "Container created")

        response = self.client.post(
            "/api/docker/create",
            json={"model": "Qwen/Qwen3.6-35B-A3B", "port": 8000},
        )

        self.assertEqual(response.status_code, 422)
        mock_create.assert_not_called()
        mock_save.assert_not_called()

    @patch("analysis_profiles.analysis_status")
    def test_analysis_status_endpoint(self, mock_status):
        mock_status.return_value = {
            "configured_model": "Qwen/Qwen3.6-35B-A3B",
            "served_model": "Qwen/Qwen3.6-35B-A3B",
            "configuration_matches_runtime": True,
            "profiles": [],
            "operation": None,
            "runtime_state": None,
        }
        response = self.client.get("/api/docker/analysis/status")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["configuration_matches_runtime"])

    @patch("analysis_profiles.start_switch")
    def test_analysis_switch_returns_operation(self, mock_switch):
        mock_switch.return_value = {"id": "a" * 32, "state": "queued"}
        response = self.client.post(
            "/api/docker/analysis/switch",
            json={"target_model": "google/gemma-4-31B-it", "confirmation": "SWITCH"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["state"], "queued")

    @patch("docker_manager.shutdown_docker_container")
    def test_docker_shutdown(self, mock_shutdown):
        mock_shutdown.return_value = (True, "Container removed")
        response = self.client.post("/api/docker/shutdown")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    # ── Documents ─────────────────────────────────────────────────────────────

    @patch("settings_manager.get_available_runs")
    def test_documents_runs(self, mock_runs):
        for filename in ("file1.md", "file2.md"):
            with open(os.path.join(self.markdown_dir, filename), "w", encoding="utf-8"):
                pass
        mock_runs.return_value = [("run_case (2 files)", self.run_dir)]

        response = self.client.get("/api/documents/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["run_name"], "run_case")
        self.assertEqual(data[0]["file_count"], 2)
        self.assertFalse(data[0]["has_pdf"])

    def test_list_run_files(self):
        for filename in ("a.md", "b.md"):
            with open(os.path.join(self.markdown_dir, filename), "w", encoding="utf-8"):
                pass

        response = self.client.get("/api/documents/runs/run_case/files")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, ["a.md", "b.md"])

        # Run not found
        response = self.client.get("/api/documents/runs/run_missing/files")
        self.assertEqual(response.status_code, 404)

    def test_get_markdown(self):
        with open(os.path.join(self.markdown_dir, "file.md"), "w", encoding="utf-8") as markdown:
            markdown.write("Markdown content")
        response = self.client.get("/api/documents/runs/run_case/markdown/file.md")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Markdown content")

        # A non-Markdown filename is rejected at the boundary.
        response = self.client.get("/api/documents/runs/run_case/markdown/file.pdf")
        self.assertEqual(response.status_code, 400)

    def test_download_run_markdown_zip(self):
        for filename, content in (("a.md", "Alpha"), ("b.md", "Beta")):
            with open(os.path.join(self.markdown_dir, filename), "w", encoding="utf-8") as markdown:
                markdown.write(content)

        response = self.client.get("/api/documents/runs/run_case/markdown.zip")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(archive.namelist(), ["a.md", "b.md"])
            self.assertEqual(archive.read("a.md"), b"Alpha")

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
    def test_pipeline_start(self, mock_process):
        stored_pdf = os.path.join(self.upload_dir, "stored.pdf")
        with open(stored_pdf, "wb") as pdf:
            pdf.write(self.pdf_bytes)
        with open(f"{stored_pdf}.metadata.json", "w", encoding="utf-8") as metadata:
            json.dump({"original_name": "original report.pdf"}, metadata)
        mock_process.return_value = [
            (
                "log",
                "badge",
                "progress",
                None,
                None,
                None,
                None,
                None,
                None,
                "run_id",
                "status",
                "manifest",
                "stop",
            )
        ]

        payload = {
            "file_paths": ["stored.pdf"],
            "server_url": "http://localhost",
            "model_name": "test-model",
        }
        response = self.client.post("/api/pipeline/start", json=payload)
        self.assertEqual(response.status_code, 200)

        # Read the event stream
        lines = [line for line in response.iter_lines() if line]
        self.assertTrue(any("run_id" in line for line in lines))
        submitted_file = mock_process.call_args.kwargs["files"][0]
        self.assertEqual(submitted_file.name, stored_pdf)
        self.assertEqual(submitted_file.original_filename, "original report.pdf")

    def test_pipeline_upload(self):
        files = [("files", ("test.pdf", self.pdf_bytes, "application/pdf"))]
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["file_paths"]) == 1)

    def test_pipeline_upload_with_spaces(self):
        files = [("files", ("test file with spaces.pdf", self.pdf_bytes, "application/pdf"))]
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["files"][0]["original_name"], "test file with spaces.pdf")

    def test_pipeline_upload_single_file_param(self):
        files = {"files": ("test_single.pdf", self.pdf_bytes, "application/pdf")}
        response = self.client.post("/api/pipeline/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["files"][0]["original_name"], "test_single.pdf")

    def test_pipeline_upload_multiple_files(self):
        files = [
            ("files", ("file1.pdf", self.pdf_bytes, "application/pdf")),
            ("files", ("file2.pdf", self.pdf_bytes, "application/pdf")),
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
        response = self.client.post("/api/rag/index", json={"run_dir": "run_case"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("indexing_service.CorpusIndexingService.index_run")
    def test_rag_index_run_streams_progress(self, mock_index):
        mock_index.return_value = ["Scanning...", "✅ Done"]
        response = self.client.post("/api/rag/index/stream", json={"run_dir": "run_case"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn('data: {"message": "Scanning..."}', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    @patch("indexing_service.CorpusIndexingService.index_all_runs")
    def test_rag_index_all_runs(self, mock_index):
        mock_index.return_value = ["Scanning...", "✅ Done"]
        response = self.client.post("/api/rag/index-all")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("indexing_service.CorpusIndexingService.index_all_runs")
    def test_rag_index_all_runs_streams_progress(self, mock_index):
        mock_index.return_value = ["Scanning...", "✅ Done"]
        response = self.client.post("/api/rag/index-all/stream")
        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"message": "Scanning..."}', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

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
        self.assertTrue(any('"type": "status"' in line for line in lines))
        self.assertTrue(any('"type": "content"' in line for line in lines))
        self.assertEqual(
            response.headers["cache-control"],
            "no-cache, no-transform",
        )

        # Case 2: Non-streaming
        payload = {"query": "test query", "stream": False}
        response = self.client.post("/api/rag/query", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Answer chunk")

    # ── Authentication & Security ─────────────────────────────────────────────

    def test_api_key_authentication(self):
        with patch.dict(os.environ, {"KIRAG_API_KEY": "test-secret-key"}):
            client = TestClient(app)
            # Test without API key header -> should fail with 401
            res = client.get("/")
            self.assertEqual(res.status_code, 401)

            # Test with invalid API key header -> 401
            res = client.get("/", headers={"X-API-Key": "wrong-key"})
            self.assertEqual(res.status_code, 401)

            # Test with valid X-API-Key header -> 200
            res = client.get("/", headers={"X-API-Key": "test-secret-key"})
            self.assertEqual(res.status_code, 200)

            # Test with valid Bearer token -> 200
            res = client.get("/", headers={"Authorization": "Bearer test-secret-key"})
            self.assertEqual(res.status_code, 200)
            client.close()

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
        self.assertEqual(
            response.headers.get("access-control-allow-origin"), "http://localhost:3000"
        )

    # ── Consolidated Phase 1 Core Endpoints ─────────────────────────────────────

    @patch("system_diagnostics.check_backing_services_data")
    @patch("system_diagnostics.get_gpu_metrics_data")
    def test_consolidated_health(self, mock_gpu, mock_backing):
        mock_backing.return_value = {"postgres": "healthy"}
        mock_gpu.return_value = {"utilization": "5%"}
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    @patch("system_diagnostics.check_backing_services_data")
    def test_readiness_reflects_core_dependencies_only(self, mock_backing):
        mock_backing.return_value = {"all_healthy": True, "failed_services": []}
        self.assertEqual(self.client.get("/readyz").status_code, 200)

        mock_backing.return_value = {
            "all_healthy": False,
            "failed_services": ["postgres"],
        }
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "service_unavailable")

        mock_backing.return_value = {"all_healthy": False, "failed_services": ["vllm_ocr"]}
        self.assertEqual(self.client.get("/readyz").status_code, 200)

    @patch("api.main._inference_endpoint_ready", side_effect=[False, True])
    def test_inference_readiness_is_role_specific(self, _mock_ready):
        response = self.client.get("/inference/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["roles"], {"ocr": False, "analysis": True})

    @patch("rag.db.get_corpus_stats")
    @patch("rag.db.get_runs_with_stats")
    @patch("rag.embedding.get_collection_info")
    @patch("rag.metadata_helper.get_all_cases_metadata")
    @patch("rag.metadata_helper.get_case_timeline")
    def test_consolidated_case_summary(
        self, mock_timeline, mock_metadata, mock_qdrant, mock_runs, mock_stats
    ):
        mock_stats.return_value = {"indexed_runs": 1, "total_chunks": 10}
        mock_runs.return_value = [
            {
                "run_id": "r1",
                "created_at": "2026-07-22",
                "total_documents": 2,
                "total_chunks": 10,
            }
        ]
        mock_qdrant.return_value = {"points_count": 10, "status": "green"}
        mock_metadata.return_value = {
            "r1": {"names": ["Case One"], "dob": "1980-01-02", "injuries": ["Shoulder"]}
        }
        mock_timeline.return_value = [{"date": "2026-07-22", "event": "Indexed"}]

        res = self.client.get("/api/case-summary")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["stats"]["indexed_runs"], 1)
        self.assertEqual(res.json()["indexed_cases"][0]["client_name"], "Case One")
        mock_runs.assert_called_once_with()

    @patch("rag.db.get_corpus_stats", side_effect=RuntimeError("database unavailable"))
    def test_consolidated_case_summary_service_failure_is_non_200(self, _mock_stats):
        res = self.client.get("/api/case-summary")

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json()["error"]["code"], "internal_error")
        self.assertNotIn("database unavailable", res.text)


if __name__ == "__main__":
    unittest.main()
