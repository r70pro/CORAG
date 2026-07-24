"""
Unit tests targeting remaining inner callback functions in app.py to achieve 100% statement coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

import app


class TestAppCallbacks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Extract callbacks from Gradio app demo fns
        cls.callbacks = {}
        for block_fn in app.demo.fns.values():
            fn = block_fn.fn
            if fn:
                cls.callbacks[getattr(fn, "__name__", "")] = fn

    def test_go_prev_page(self):
        go_prev_page = self.callbacks.get("go_prev_page")
        self.assertIsNotNone(go_prev_page)
        self.assertEqual(go_prev_page(5), 4)
        self.assertEqual(go_prev_page(1), 1)

    def test_go_next_page(self):
        go_next_page = self.callbacks.get("go_next_page")
        self.assertIsNotNone(go_next_page)
        self.assertEqual(go_next_page(2, 5), 3)
        self.assertEqual(go_next_page(5, 5), 5)

    @patch("app.save_settings")
    def test_trigger_save_settings(self, mock_save):
        trigger_save_settings = self.callbacks.get("trigger_save_settings")
        self.assertIsNotNone(trigger_save_settings)
        
        mock_save.return_value = "Settings saved"
        res = trigger_save_settings(
            "url", "model", "4", "20", "1288", "8", True, "8000", "0.9", "2048", "token"
        )
        self.assertEqual(res, "Settings saved")
        mock_save.assert_called_once()

    @patch("app.start_docker_container")
    @patch("app.get_docker_status_str")
    def test_ui_start_container(self, mock_status_str, mock_start):
        ui_start = self.callbacks.get("ui_start_container")
        self.assertIsNotNone(ui_start)

        mock_start.return_value = (True, "Started")
        mock_status_str.return_value = (True, "Running badge")
        
        msg, badge = ui_start(8000)
        self.assertEqual(msg, "Started")
        self.assertEqual(badge, "Running badge")

    @patch("app.stop_docker_container")
    @patch("app.get_docker_status_str")
    def test_ui_stop_container(self, mock_status_str, mock_stop):
        ui_stop = self.callbacks.get("ui_stop_container")
        self.assertIsNotNone(ui_stop)

        mock_stop.return_value = (True, "Stopped")
        mock_status_str.return_value = (True, "Stopped badge")
        
        msg, badge = ui_stop(8000)
        self.assertEqual(msg, "Stopped")
        self.assertEqual(badge, "Stopped badge")

    @patch("app.shutdown_docker_container")
    @patch("rag_infra_manager.destroy_rag_infrastructure")
    @patch("app.get_docker_status_str")
    def test_ui_shutdown_all_containers(self, mock_status_str, mock_destroy, mock_shutdown):
        ui_shutdown = self.callbacks.get("ui_shutdown_all_containers")
        self.assertIsNotNone(ui_shutdown)

        mock_shutdown.return_value = (True, "Container shutdown successfully.")
        mock_destroy.return_value = (True, "RAG infrastructure destroyed.")
        mock_status_str.return_value = (True, "Shutdown badge")
        
        msg, badge = ui_shutdown(8000)
        self.assertEqual(msg, "Container shutdown successfully. RAG: RAG infrastructure destroyed.")
        self.assertEqual(badge, "Shutdown badge")

        # Case when destroy_rag_infrastructure raises an exception
        mock_destroy.side_effect = Exception("failed to down")
        msg, badge = ui_shutdown(8000)
        self.assertIn("Error destroying infrastructure", msg)

    @patch("app.create_docker_container")
    @patch("app.get_docker_status_str")
    @patch("app.save_settings")
    def test_ui_recreate_container(self, mock_save, mock_status_str, mock_create):
        ui_recreate = self.callbacks.get("ui_recreate_container")
        self.assertIsNotNone(ui_recreate)

        mock_create.return_value = (True, "Recreated")
        mock_status_str.return_value = (True, "Recreated badge")
        
        msg, badge, url = ui_recreate("token", 8000, "model", 0.9, 2048)
        self.assertEqual(msg, "Recreated")
        self.assertEqual(badge, "Recreated badge")
        self.assertEqual(url, "http://localhost:8000/v1")

    @patch("app.start_docker_container")
    @patch("app.get_docker_status_str")
    def test_ui_header_start(self, mock_status_str, mock_start):
        ui_header_start = self.callbacks.get("ui_header_start")
        self.assertIsNotNone(ui_header_start)

        mock_status_str.return_value = (True, "Running badge")
        badge = ui_header_start(8000)
        self.assertEqual(badge, "Running badge")

    @patch("app.stop_docker_container")
    @patch("app.get_docker_status_str")
    def test_ui_header_stop(self, mock_status_str, mock_stop):
        ui_header_stop = self.callbacks.get("ui_header_stop")
        self.assertIsNotNone(ui_header_stop)

        mock_status_str.return_value = (True, "Stopped badge")
        badge = ui_header_stop(8000)
        self.assertEqual(badge, "Stopped badge")

    @patch("app.get_docker_status_str")
    def test_periodic_status_check(self, mock_status_str):
        periodic_check = self.callbacks.get("periodic_status_check")
        self.assertIsNotNone(periodic_check)

        mock_status_str.return_value = (True, "Status badge")
        
        # 1. Port value is None fallback
        badge1 = periodic_check(None)
        self.assertEqual(badge1, "Status badge")

        # 2. Port value is valid
        badge2 = periodic_check(8080)
        self.assertEqual(badge2, "Status badge")

    @patch("psycopg2.connect")
    @patch("redis.Redis")
    @patch("requests.get")
    @patch("socket.socket")
    def test_get_service_latency(self, mock_socket, mock_get, mock_redis, mock_postgres):
        # 1. postgres
        mock_conn = mock_postgres.return_value
        ok, lat, extra = app.get_service_latency("postgres")
        self.assertTrue(ok)
        self.assertIsNone(extra)
        mock_conn.close.assert_called()

        # 2. redis
        mock_r = mock_redis.return_value
        ok, lat, extra = app.get_service_latency("redis")
        self.assertTrue(ok)
        self.assertIsNone(extra)
        mock_r.ping.assert_called()

        # 3. minio (healthy)
        mock_get.return_value = MagicMock(status_code=200)
        ok, lat, extra = app.get_service_latency("minio")
        self.assertTrue(ok)
        self.assertIsNone(extra)

        # minio (unhealthy)
        mock_get.return_value = MagicMock(status_code=500)
        ok, lat, extra = app.get_service_latency("minio")
        self.assertFalse(ok)
        self.assertIsNone(extra)

        # 4. qdrant (healthy)
        mock_get.return_value = MagicMock(status_code=200)
        ok, lat, extra = app.get_service_latency("qdrant")
        self.assertTrue(ok)
        self.assertIsNone(extra)

        # 5. vllm (healthy)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"data": [{"id": "allenai/olmOCR-2-7B-1025-FP8"}]}
        mock_get.return_value = mock_response
        ok, lat, extra = app.get_service_latency("vllm")
        self.assertTrue(ok)
        self.assertEqual(extra, "allenai/olmOCR-2-7B-1025-FP8")

        # 6. other (socket check)
        mock_s = mock_socket.return_value
        ok, lat, extra = app.get_service_latency("other_service", port=1234)
        self.assertTrue(ok)
        self.assertIsNone(extra)
        mock_s.connect.assert_called_with(("127.0.0.1", 1234))

        # 7. exception flow
        mock_get.side_effect = Exception("HTTP error")
        ok, lat, extra = app.get_service_latency("minio")
        self.assertFalse(ok)
        self.assertIsNone(extra)

    def test_get_simulated_sparkline(self):
        # 1. is_up=False
        res = app.get_simulated_sparkline(is_up=False)
        self.assertTrue("sparkline-red" in res)

        # 2. is_up=True, latency_history=None
        res = app.get_simulated_sparkline(is_up=True, latency_history=None)
        self.assertTrue("sparkline-svg" in res)
        self.assertFalse("sparkline-red" in res)

        # 3. is_up=True, latency_history has values
        res = app.get_simulated_sparkline(is_up=True, latency_history=[10.0, 20.0, 15.0])
        self.assertTrue("sparkline-svg" in res)

    @patch("system_diagnostics.get_service_latency")
    def test_check_backing_services(self, mock_latency):
        # All healthy - general model (suited for RAG)
        mock_latency.return_value = (True, 5.0, "nvidia/Llama-3.3-70B-Instruct-NVFP4")
        html, badge = app.check_backing_services()
        self.assertTrue("✓ System Healthy" in badge)
        self.assertTrue("nvidia/Llama-3.3-70B-Instruct-NVFP4" in badge)
        self.assertTrue("Best suited for RAG processing" in badge)
        self.assertTrue("PostgreSQL" in html)
        self.assertTrue("nvidia/Llama-3.3-70B-Instruct-NVFP4" in html)

        # All healthy - OCR model (suited for PDF conversion)
        mock_latency.return_value = (True, 5.0, "allenai/olmOCR-2-7B-1025-FP8")
        html, badge = app.check_backing_services()
        self.assertTrue("✓ System Healthy" in badge)
        self.assertTrue("allenai/olmOCR-2-7B-1025-FP8" in badge)
        self.assertTrue("Best suited for PDF conversion" in badge)

        # Some unhealthy
        mock_latency.return_value = (False, 0.0, None)
        html, badge = app.check_backing_services()
        self.assertTrue("✗ System Degraded" in badge)
        self.assertTrue("PostgreSQL" in badge)
        self.assertTrue("Start PostgreSQL service/container." in badge)

    @patch("subprocess.run")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    def test_get_gpu_metrics(self, mock_mem, mock_prop, mock_name, mock_avail, mock_run):
        # Mock MagicMock
        from unittest.mock import MagicMock

        # Case 1: CUDA available via torch
        mock_avail.return_value = True
        mock_name.return_value = "Test GPU 3080"
        mock_prop.return_value = MagicMock(total_memory=16 * 1024 * 1024 * 1024)
        mock_mem.return_value = 4 * 1024 * 1024 * 1024
        mock_run.side_effect = Exception("nvidia-smi not found")

        html = app.get_gpu_metrics()
        self.assertTrue("CUDA Available" in html)
        self.assertTrue("Test GPU 3080" in html)

        # Case 2: CUDA available via nvidia-smi
        mock_avail.return_value = False
        mock_run.side_effect = None
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 4000, 24000\n"
        )
        html = app.get_gpu_metrics()
        self.assertTrue("CUDA Available" in html)
        self.assertTrue("RTX 4090" in html)

        # Case 3: CUDA unavailable completely
        mock_run.side_effect = Exception("No GPU")
        html = app.get_gpu_metrics()
        self.assertTrue("CUDA Unavailable" in html)
        self.assertTrue("Running on Host CPU" in html)

    @patch("subprocess.run")
    def test_get_vllm_loading_progress(self, mock_run):
        from unittest.mock import MagicMock
        
        # Case 1: Docker logs show loading progress with ETA
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Loading safetensors checkpoint shards:  18% Completed | 3/17 [01:21<06:22, 27.31s/it]\n"
        )
        progress = app.get_vllm_loading_progress()
        self.assertIsNotNone(progress)
        self.assertEqual(progress["pct"], 18)
        self.assertEqual(progress["shards_loaded"], 3)
        self.assertEqual(progress["shards_total"], 17)
        self.assertEqual(progress["eta"], "6m 22s")
        
        # Case 2: Docker logs show initial state with "?" ETA
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Loading safetensors checkpoint shards:   0% Completed | 0/17 [00:00<?, ?it/s]\n"
        )
        progress = app.get_vllm_loading_progress()
        self.assertIsNotNone(progress)
        self.assertEqual(progress["pct"], 0)
        self.assertEqual(progress["eta"], "Calculating...")

        # Case 3: Docker logs don't have progress
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Starting model runner...\n"
        )
        progress = app.get_vllm_loading_progress()
        self.assertIsNone(progress)

    def test_select_view(self):
        self.assertTrue(hasattr(app, "select_view"))
        res = app.select_view(0)
        self.assertEqual(len(res), 13)
        self.assertIsNotNone(res[0])

    @patch("app.check_backing_services")
    @patch("app.get_gpu_metrics")
    def test_periodic_diagnostics_check(self, mock_gpu, mock_backing):
        periodic_diag = self.callbacks.get("periodic_diagnostics_check")
        self.assertIsNotNone(periodic_diag)

        mock_backing.return_value = ("backing_html", "health_badge")
        mock_gpu.return_value = "gpu_html"

        backing, gpu, badge = periodic_diag(8000)
        self.assertEqual(backing, "backing_html")
        
        # Test default fallback value (None)
        backing2, gpu2, badge2 = periodic_diag(None)
        self.assertEqual(backing2, "backing_html")

    @patch("subprocess.run")
    def test_get_vllm_loading_progress_extended(self, mock_run):
        from unittest.mock import MagicMock
        
        # Case 1: 3-part ETA (hours, minutes, seconds)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Loading safetensors checkpoint shards:  50% Completed | 5/10 [01:00:00<02:30:15, 27s/it]\n"
        )
        progress = app.get_vllm_loading_progress()
        self.assertIsNotNone(progress)
        self.assertEqual(progress["eta"], "2h 30m 15s")

        # Case 2: No parts (just raw string without colon)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Loading safetensors checkpoint shards:  50% Completed | 5/10 [01:00:00<10s, 27s/it]\n"
        )
        progress = app.get_vllm_loading_progress()
        self.assertIsNotNone(progress)
        self.assertEqual(progress["eta"], "10s")

        # Case 3: Empty stdout / returncode non-zero
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=""
        )
        self.assertIsNone(app.get_vllm_loading_progress())

        # Case 4: Exception raised
        mock_run.side_effect = Exception("Docker logs failed")
        self.assertIsNone(app.get_vllm_loading_progress())

    @patch("system_diagnostics.get_service_latency")
    @patch("system_diagnostics.get_vllm_loading_progress")
    def test_check_backing_services_loading(self, mock_progress, mock_latency):
        # Scenario: vllm is the only failed service and is loading
        def latency_side_effect(service, **kwargs):
            if service == "vllm":
                return False, 0.0, None
            return True, 5.0, None
        mock_latency.side_effect = latency_side_effect

        mock_progress.return_value = {
            "pct": 45,
            "shards_loaded": 9,
            "shards_total": 20,
            "eta": "1m 30s"
        }

        html, badge = app.check_backing_services()
        self.assertTrue("⚡ Model Loading" in badge)
        self.assertTrue("Progress: <span style='font-weight:600; color:#e2e8f0;'>45%</span>" in badge)
        self.assertTrue("ETA: 1m 30s" in badge)
        self.assertTrue("Progress: 45%" in html)

        # Scenario: All healthy but model is None Loaded or Unknown
        mock_latency.side_effect = None
        mock_latency.return_value = (True, 5.0, "None Loaded")
        html, badge = app.check_backing_services()
        self.assertTrue("No model loaded" in badge)

    @patch("subprocess.run")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    @patch("builtins.open", create=True)
    def test_get_gpu_metrics_detailed(self, mock_file_open, mock_mem, mock_prop, mock_name, mock_avail, mock_run):
        from unittest.mock import MagicMock, mock_open
        
        mock_avail.return_value = True
        mock_name.return_value = "Test GPU 3080"
        mock_prop.return_value = MagicMock(total_memory=16 * 1024 * 1024 * 1024)
        mock_mem.return_value = 4 * 1024 * 1024 * 1024

        def run_side_effect(args, **kwargs):
            if "nvidia-smi" in args and "--query-gpu=name,memory.used,memory.total" in args:
                return MagicMock(returncode=0, stdout="Test GPU 3080, 4000, 16000\n")
            elif "docker" in args and "ps" in args:
                return MagicMock(returncode=0, stdout="container123|my-vllm-container\n")
            elif "nvidia-smi" in args:
                processes_stdout = (
                    "| Processes:                                                                            |\n"
                    "|  GPU   GI   CI        PID   Type   Process name                             vram usage     |\n"
                    "|============================================================================================|\n"
                    "|    0   N/A  N/A     10001      C   /usr/bin/python3                              15000MiB |\n"
                    "|    0   N/A  N/A     10002      G   /usr/bin/xorg                                   100MiB |\n"
                    "|    0   N/A  N/A     10003      C   /usr/bin/python3                               5000MiB |\n"
                    "+--------------------------------------------------------------------------------------------+\n"
                )
                return MagicMock(returncode=0, stdout=processes_stdout)
            return MagicMock(returncode=1)
        mock_run.side_effect = run_side_effect

        # Helper to simulate file open side effect
        def open_side_effect(file, *args, **kwargs):
            filename = str(file)
            if "/proc/10001/cmdline" in filename:
                return mock_open(read_data="python\x00my_script.py\x00")()
            elif "/proc/10001/cgroup" in filename:
                return mock_open(read_data="12:cpu:/docker-container123.scope\n")()
            elif "/proc/10002/cmdline" in filename:
                return mock_open(read_data="/usr/bin/xorg\x00")()
            elif "/proc/10002/cgroup" in filename:
                return mock_open(read_data="12:cpu:/\n")()
            elif "/proc/10003/cmdline" in filename:
                return mock_open(read_data="python\x00other_script.py\x00")()
            elif "/proc/10003/cgroup" in filename:
                return mock_open(read_data="12:cpu:/docker/container123/some_subpath\n")()
            raise FileNotFoundError(f"Mocked file not found: {filename}")
        mock_file_open.side_effect = open_side_effect

        html = app.get_gpu_metrics()
        self.assertTrue("CUDA Available" in html)
        self.assertTrue("python: my_script.py" in html)
        self.assertTrue("Docker: my-vllm-container" in html)
        self.assertTrue("System Graphics" in html)
        self.assertTrue("python: other_script.py" in html)

    def test_app_handlers_invalid_port_and_download_report(self):
        import app_handlers
        mock_create = MagicMock(return_value=(True, "Created"))
        mock_status = MagicMock(return_value=("ready", "Badge"))
        def mock_get_app_fn(name, fallback):
            if name == "create_docker_container":
                return mock_create
            elif name == "get_docker_status_str":
                return mock_status
            return fallback

        with patch("app_handlers.get_app_fn", side_effect=mock_get_app_fn), \
             patch("app_handlers.load_settings", return_value={}), \
             patch("app_handlers.save_settings"):
            msg, badge, new_url = app_handlers.ui_recreate_container("tok", "invalid_port", "model", 0.8, 15000)
            self.assertTrue("created" in msg.lower())
            self.assertTrue(isinstance(badge, str))

        with patch("system_diagnostics.generate_diagnostic_report_file", return_value="/tmp/diag.txt"):
            res1 = app_handlers.trigger_download_report(None)
            self.assertEqual(res1.get("value"), "/tmp/diag.txt")
            res2 = app_handlers.trigger_download_report(8000)
            self.assertEqual(res2.get("value"), "/tmp/diag.txt")


if __name__ == "__main__":
    from unittest.mock import MagicMock
    unittest.main()


