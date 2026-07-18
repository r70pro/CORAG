"""
Unit tests for system_diagnostics.py targeting 100% statement and branch coverage.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Prevent system operations during import
os.environ["TESTING"] = "true"

import system_diagnostics


class TestSystemDiagnostics(unittest.TestCase):

    @patch("psycopg2.connect")
    @patch("redis.Redis")
    @patch("requests.get")
    @patch("socket.socket")
    def test_get_service_latency(self, mock_socket, mock_http, mock_redis, mock_pg):
        # 1. postgres success
        mock_conn = MagicMock()
        mock_pg.return_value = mock_conn
        success, latency, info = system_diagnostics.get_service_latency("postgres")
        self.assertTrue(success)
        mock_conn.close.assert_called_once()

        # 2. redis success
        mock_r = MagicMock()
        mock_redis.return_value = mock_r
        success, latency, info = system_diagnostics.get_service_latency("redis")
        self.assertTrue(success)
        mock_r.ping.assert_called_once()

        # 3. minio success
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_http.return_value = mock_resp
        success, latency, info = system_diagnostics.get_service_latency("minio")
        self.assertTrue(success)

        # 4. minio failure
        mock_resp.status_code = 500
        success, latency, info = system_diagnostics.get_service_latency("minio")
        self.assertFalse(success)

        # 5. qdrant success
        mock_resp.status_code = 200
        success, latency, info = system_diagnostics.get_service_latency("qdrant")
        self.assertTrue(success)

        # 6. qdrant failure
        mock_resp.status_code = 404
        success, latency, info = system_diagnostics.get_service_latency("qdrant")
        self.assertFalse(success)

        # 7. vllm success with loaded model
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "my_vllm_model"}]}
        success, latency, info = system_diagnostics.get_service_latency("vllm")
        self.assertTrue(success)
        self.assertEqual(info, "my_vllm_model")

        # 8. vllm success but no models loaded
        mock_resp.json.return_value = {"data": []}
        success, latency, info = system_diagnostics.get_service_latency("vllm")
        self.assertTrue(success)
        self.assertEqual(info, "None Loaded")

        # 9. vllm success but json decoding fails
        mock_resp.json.side_effect = Exception("Invalid JSON")
        success, latency, info = system_diagnostics.get_service_latency("vllm")
        self.assertTrue(success)
        self.assertEqual(info, "Unknown")
        mock_resp.json.side_effect = None

        # 10. vllm failure
        mock_resp.status_code = 500
        success, latency, info = system_diagnostics.get_service_latency("vllm")
        self.assertFalse(success)

        # 11. Custom generic socket success
        success, latency, info = system_diagnostics.get_service_latency("custom_service", host="localhost", port=1234)
        self.assertTrue(success)
        mock_socket.assert_called_once()

    @patch("subprocess.run")
    def test_get_vllm_loading_progress(self, mock_run):
        # 1. returncode != 0
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertIsNone(res)

        # 2. Match with "Calculating..." ETA
        log_calculating = "Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]"
        mock_run.return_value = MagicMock(returncode=0, stdout=log_calculating)
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertEqual(res["pct"], 0)
        self.assertEqual(res["shards_loaded"], 0)
        self.assertEqual(res["shards_total"], 2)
        self.assertEqual(res["eta"], "Calculating...")

        # 3. Match with mm:ss ETA
        log_m_s = "Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:10<00:10,  9.88s/it]"
        mock_run.return_value = MagicMock(returncode=0, stdout=log_m_s)
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertEqual(res["pct"], 50)
        self.assertEqual(res["shards_loaded"], 1)
        self.assertEqual(res["shards_total"], 2)
        self.assertEqual(res["eta"], "0m 10s")

        # 4. Match with hh:mm:ss ETA
        log_h_m_s = "Loading safetensors checkpoint shards:  10% Completed | 1/10 [01:00<01:00:00, 600.0s/it]"
        mock_run.return_value = MagicMock(returncode=0, stdout=log_h_m_s)
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertEqual(res["eta"], "1h 0m 0s")

        # 5. Generic ETA string
        log_generic = "Loading safetensors checkpoint shards:  99% Completed | 99/100 [00:45<some_eta, 0.4s/it]"
        mock_run.return_value = MagicMock(returncode=0, stdout=log_generic)
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertEqual(res["eta"], "some_eta")

        # 6. Exception raised in subprocess.run
        mock_run.side_effect = Exception("Docker command failed")
        res = system_diagnostics.get_vllm_loading_progress()
        self.assertIsNone(res)

    @patch("system_diagnostics.get_service_latency")
    @patch("system_diagnostics.get_vllm_loading_progress")
    def test_check_backing_services_data(self, mock_progress, mock_latency):
        # Service latency return: is_up, latency, extra_info
        mock_latency.side_effect = [
            (True, 1.5, None),   # postgres
            (True, 2.0, None),   # redis
            (True, 3.0, None),   # minio
            (True, 4.0, None),   # qdrant
            (False, 0.0, None),  # vllm (down)
        ]
        mock_progress.return_value = {"pct": 50, "shards_loaded": 1, "shards_total": 2, "eta": "10s"}

        history = {}
        res = system_diagnostics.check_backing_services_data(history)
        self.assertFalse(res["all_healthy"])
        self.assertEqual(res["failed_services"], ["vllm"])
        self.assertEqual(res["vllm_progress"], {"pct": 50, "shards_loaded": 1, "shards_total": 2, "eta": "10s"})

        # History collection check: should populate up to 8 slots
        mock_latency.side_effect = None
        mock_latency.return_value = (True, 5.0, "phi-4")
        mock_progress.return_value = None

        # Call multiple times to check list cap
        for _ in range(12):
            res2 = system_diagnostics.check_backing_services_data(history)
        self.assertEqual(len(history["postgres"]), 8)
        self.assertEqual(res2["vllm_model"], "phi-4")

    @patch("subprocess.run")
    def test_get_docker_containers(self, mock_run):
        # 1. Success
        mock_run.return_value = MagicMock(returncode=0, stdout="c123|postgres\nc456|redis\n")
        containers = system_diagnostics.get_docker_containers()
        self.assertEqual(containers["c123"], "postgres")
        self.assertEqual(containers["c456"], "redis")

        # 2. Exception
        mock_run.side_effect = Exception("Docker down")
        self.assertEqual(system_diagnostics.get_docker_containers(), {})

    def test_resolve_process_details(self):
        # We need to mock opening /proc/{pid}/cmdline and /proc/{pid}/cgroup
        m_cmdline = mock_open(read_data="python\x00my_script.py\x00arg1\x00")
        m_cgroup_111 = mock_open(read_data="13:name=systemd:/docker-c1234567890abcdef.scope")
        m_cgroup_222 = mock_open(read_data="13:name=systemd:/docker/c1234567890abcdef")
        m_cgroup_other = mock_open(read_data="13:name=systemd:/")

        def side_effect(path, *args, **kwargs):
            if "cmdline" in path:
                return m_cmdline()
            if "cgroup" in path:
                if "111" in path:
                    return m_cgroup_111()
                if "222" in path:
                    return m_cgroup_222()
                return m_cgroup_other()
            raise FileNotFoundError()

        with patch("builtins.open", side_effect=side_effect):
            # Case 1: Docker format 1
            cmd, is_docker, cname = system_diagnostics.resolve_process_details(111, "python")
            self.assertEqual(cmd, "python my_script.py arg1")
            self.assertTrue(is_docker)
            self.assertEqual(cname, "c1234567890a")

            # Case 2: Docker format 2
            cmd, is_docker, cname = system_diagnostics.resolve_process_details(222, "python")
            self.assertTrue(is_docker)
            self.assertEqual(cname, "c1234567890a")

            # Case 3: Non-docker process
            cmd, is_docker, cname = system_diagnostics.resolve_process_details(333, "python")
            self.assertFalse(is_docker)

        # Exception path (proc doesn't exist)
        with patch("builtins.open", side_effect=FileNotFoundError()):
            cmd, is_docker, cname = system_diagnostics.resolve_process_details(999, "fallback")
            self.assertEqual(cmd, "fallback")
            self.assertFalse(is_docker)

    def test_get_display_name(self):
        # 1. empty cmdline
        self.assertEqual(system_diagnostics.get_display_name("", "default"), "default")
        
        # 2. python script mapping
        self.assertEqual(system_diagnostics.get_display_name("python /path/to/my_script.py", "python"), "python: my_script.py")
        
        # 3. regular script basename
        self.assertEqual(system_diagnostics.get_display_name("/usr/bin/postgres -D ...", "postgres"), "postgres")

        # 4. empty basename path (e.g. root "/")
        self.assertEqual(system_diagnostics.get_display_name("/", "default"), "default")

    @patch("system_diagnostics.subprocess.run")
    @patch("system_diagnostics.get_docker_containers")
    def test_get_gpu_metrics_data(self, mock_docker, mock_run):
        # 1. CUDA not available and nvidia-smi failed
        mock_run.side_effect = Exception("nvidia-smi not found")
        with patch.dict(sys.modules, {"torch": None}): # bypass torch checks
            res = system_diagnostics.get_gpu_metrics_data()
            self.assertFalse(res["cuda_available"])
            self.assertEqual(res["processes"], [])

        # Reset mock
        mock_run.side_effect = None
        mock_docker.return_value = {"c123": "my_db_container"}

        # 2. CUDA is available via torch, and nvidia-smi succeeds
        # Setup torch mock
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=24000 * 1024 * 1024)
        mock_torch.cuda.memory_allocated.return_value = 4000 * 1024 * 1024

        # nvidia-smi stdout mocks
        smi_query_out = "NVIDIA RTX 4090, 4000, 24000"
        smi_proc_section = """
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
+-----------------------------------------------------------------------------+
|  GPU   GI   CI        PID   Type   Process name                             |
|=============================================================================|
|    0   N/A  N/A       111      C   python                               500MiB |
|    0   N/A  N/A       222      G   /usr/bin/gnome-shell                 100MiB |
|    0   N/A  N/A       333      C   /usr/local/bin/app                    50MiB |
+-----------------------------------------------------------------------------+
"""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=smi_query_out),
            MagicMock(returncode=0, stdout=smi_proc_section),
        ]

        # Setup resolve process details mock
        def mock_resolve(pid, name):
            if pid == 111:
                return "python /app/vllm_server.py", True, "c123"
            elif pid == 222:
                return "/usr/bin/gnome-shell", False, ""
            else:
                return "/usr/local/bin/app", False, ""

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("system_diagnostics.resolve_process_details", mock_resolve):
                res = system_diagnostics.get_gpu_metrics_data()
                self.assertTrue(res["cuda_available"])
                self.assertEqual(res["gpu_name"], "NVIDIA RTX 4090")
                self.assertEqual(res["vram_used"], 4000.0)
                self.assertEqual(res["vram_total"], 24000.0)
                self.assertEqual(len(res["processes"]), 3)
                
                # Check resolved categories
                # Process 111 should map to Docker container
                self.assertEqual(res["processes"][0]["type_text"], "Docker: my_db_container")
                # Process 222 should map to System Graphics (gnome-shell is essential keyword)
                self.assertEqual(res["processes"][1]["type_text"], "System Graphics")
                # Process 333 should map to Application
                self.assertEqual(res["processes"][2]["type_text"], "Application")

    @patch("system_diagnostics.check_backing_services_data")
    @patch("system_diagnostics.get_gpu_metrics_data")
    @patch("settings_manager.load_settings")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_generate_diagnostic_report_file(self, mock_makedirs, mock_file, mock_load_settings, mock_gpu_data, mock_backing_data):
        mock_load_settings.return_value = {"hf_token": "some_token", "other_setting": "value"}
        mock_backing_data.return_value = {
            "all_healthy": True,
            "services": {
                "postgres": {"is_up": True, "latency": 1.5, "extra_info": "Connected"},
            }
        }
        mock_gpu_data.return_value = {
            "cuda_available": True,
            "gpu_name": "RTX 4090",
            "vram_used": 1000.0,
            "vram_free": 15000.0,
            "vram_total": 16000.0,
            "vram_pct": 6.25,
            "vram_reclaimable": 0.0,
            "vram_potential_free": 15000.0,
            "processes": [
                {"pid": 123, "display_name": "python", "vram": 500.0, "type_text": "Application"}
            ]
        }
        
        # Test path when CUDA is available
        path = system_diagnostics.generate_diagnostic_report_file(8000)
        self.assertTrue(path.endswith("diagnostic_report.md"))
        mock_file.assert_called_with(path, "w", encoding="utf-8")
        
        # Test path when CUDA is not available
        mock_gpu_data.return_value["cuda_available"] = False
        path = system_diagnostics.generate_diagnostic_report_file(8000)
        self.assertTrue(path.endswith("diagnostic_report.md"))


if __name__ == "__main__":
    unittest.main()
