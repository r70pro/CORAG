"""
Unit tests targeting remaining inner callback functions in app.py to achieve 100% statement coverage.
"""

import os
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
