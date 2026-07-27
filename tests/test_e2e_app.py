"""
End-to-End (E2E) tests for OLMOCR Gradio Dashboard.

Launches the Gradio application inside the test thread to record code coverage on callbacks.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from gradio_client import Client

# Disable actual docker controls during startup
os.environ["TESTING"] = "true"

import app


class TestAppE2E(unittest.TestCase):

    started_infra = False

    @classmethod
    def setUpClass(cls):
        # The callback exercised below persists analysis configuration. Keep
        # that write isolated from the tracked production settings file.
        cls.settings_directory = tempfile.TemporaryDirectory()
        cls.settings_patcher = patch(
            "settings_manager.SETTINGS_FILE",
            os.path.join(cls.settings_directory.name, "settings.json"),
        )
        cls.settings_patcher.start()
        cls.addClassCleanup(cls.settings_directory.cleanup)
        cls.addClassCleanup(cls.settings_patcher.stop)

        # Auto start RAG infra if it's not running
        from rag_infra_manager import is_rag_infrastructure_ready, start_and_init_rag
        if not is_rag_infrastructure_ready():
            print("RAG Infrastructure not ready. Starting it for E2E tests...")
            success, msg = start_and_init_rag()
            if not success:
                raise unittest.SkipTest(f"Skipping E2E tests: Failed to start RAG infra: {msg}")
            cls.started_infra = True

        cls.server_port = 7868
        cls.client_url = f"http://127.0.0.1:{cls.server_port}/"
        
        # Launch demo in a separate thread so it does not block execution
        cls.demo = app.demo
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.server_port,
            prevent_thread_lock=True,
            allowed_paths=["/home/owner"]
        )
        
        try:
            cls.client = Client(cls.client_url)
        except Exception as e:
            cls.demo.close()
            raise unittest.SkipTest(
                f"Skipping E2E tests: Gradio Client connection failed: {e}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "demo"):
            cls.demo.close()
        if cls.started_infra:
            print("Stopping RAG infrastructure started for E2E tests...")
            from rag_infra_manager import stop_rag_infrastructure
            stop_rag_infrastructure()

    def test_01_refresh_corpus_display(self):
        # Test endpoint to refresh statistics
        result = self.client.predict(api_name="/refresh_corpus_display")
        self.assertIsNotNone(result)
        self.assertTrue("Corpus Statistics" in result or "stats" in result.lower())

    def test_02_refresh_runs_dropdown(self):
        # Test endpoint to scan runs in workspace
        result = self.client.predict(api_name="/refresh_runs_dropdown")
        # Gradio dropdown gr.update returns a dict with choices and values
        self.assertTrue(isinstance(result, dict))
        self.assertTrue("choices" in result)

    def test_03_save_analysis_settings(self):
        # Test endpoint to save analysis configurations
        result = self.client.predict(
            url="http://localhost:8000/v1",
            name="nvidia/Phi-4-reasoning-plus-NVFP4",
            top_k=5,
            api_name="/save_analysis_settings",
        )
        self.assertTrue("saved successfully" in result.lower())

    def test_04_periodic_status_check(self):
        # Test pipeline periodic Docker status checks
        result = self.client.predict(
            port_val=8000,
            api_name="/periodic_status_check"
        )
        self.assertIsNotNone(result)
        self.assertTrue("Docker" in result or "Server" in result or "Offline" in result or "Badge" in result or "<span" in result)


if __name__ == "__main__":
    unittest.main()
