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
        ok, lat = app.get_service_latency("postgres")
        self.assertTrue(ok)
        mock_conn.close.assert_called()

        # 2. redis
        mock_r = mock_redis.return_value
        ok, lat = app.get_service_latency("redis")
        self.assertTrue(ok)
        mock_r.ping.assert_called()

        # 3. minio (healthy)
        mock_get.return_value = MagicMock(status_code=200)
        ok, lat = app.get_service_latency("minio")
        self.assertTrue(ok)

        # minio (unhealthy)
        mock_get.return_value = MagicMock(status_code=500)
        ok, lat = app.get_service_latency("minio")
        self.assertFalse(ok)

        # 4. qdrant (healthy)
        mock_get.return_value = MagicMock(status_code=200)
        ok, lat = app.get_service_latency("qdrant")
        self.assertTrue(ok)

        # 5. vllm (healthy)
        mock_get.return_value = MagicMock(status_code=200)
        ok, lat = app.get_service_latency("vllm")
        self.assertTrue(ok)

        # 6. other (socket check)
        mock_s = mock_socket.return_value
        ok, lat = app.get_service_latency("other_service", port=1234)
        self.assertTrue(ok)
        mock_s.connect.assert_called_with(("127.0.0.1", 1234))

        # 7. exception flow
        mock_get.side_effect = Exception("HTTP error")
        ok, lat = app.get_service_latency("minio")
        self.assertFalse(ok)

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

    @patch("app.get_service_latency")
    def test_check_backing_services(self, mock_latency):
        # All healthy
        mock_latency.return_value = (True, 5.0)
        html, badge = app.check_backing_services()
        self.assertTrue("✓ System Healthy" in badge)
        self.assertTrue("PostgreSQL" in html)

        # Some unhealthy
        mock_latency.return_value = (False, 0.0)
        html, badge = app.check_backing_services()
        self.assertTrue("✗ System Degraded" in badge)

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

    def test_select_view(self):
        self.assertTrue(hasattr(app, "select_view"))
        res = app.select_view(0)
        self.assertEqual(len(res), 11)
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
        self.assertEqual(gpu, "gpu_html")
        self.assertEqual(badge, "health_badge")


if __name__ == "__main__":
    from unittest.mock import MagicMock
    unittest.main()

