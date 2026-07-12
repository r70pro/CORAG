"""
Unit tests for download_models.py.
"""

import os
import unittest
from unittest.mock import patch
import importlib

# Prevent actual model downloads during test imports
os.environ["TESTING"] = "true"


class TestDownloadModels(unittest.TestCase):

    @patch("download_models.snapshot_download")
    @patch("download_models.time.sleep")
    def test_download_success(self, mock_sleep, mock_snapshot):
        mock_snapshot.return_value = "/path/to/downloaded/model"
        
        with patch("builtins.print"):
            import download_models
            download_models.download_all_models()
            
        self.assertEqual(mock_snapshot.call_count, len(download_models.MODELS))
        mock_sleep.assert_not_called()

    @patch("download_models.snapshot_download")
    @patch("download_models.time.sleep")
    def test_download_failure_and_retry(self, mock_sleep, mock_snapshot):
        # Fail first 2 attempts, succeed on 3rd for the first model, and succeed immediately for others
        mock_snapshot.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            "/path/to/downloaded/model",
            "/path/to/downloaded/model",
            "/path/to/downloaded/model",
            "/path/to/downloaded/model",
        ]
        
        with patch("builtins.print"):
            import download_models
            download_models.download_all_models()
            
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_called_with(10)

    @patch("download_models.snapshot_download")
    @patch("download_models.time.sleep")
    def test_download_all_attempts_fail(self, mock_sleep, mock_snapshot):
        mock_snapshot.side_effect = Exception("Auth failed")
        
        with patch("builtins.print"), patch("traceback.print_exc") as mock_traceback:
            import download_models
            download_models.download_all_models()
            
        self.assertEqual(mock_sleep.call_count, 16)  # 4 retries per model * 4 models
        self.assertEqual(mock_traceback.call_count, 4)

    @patch("settings_manager.load_settings")
    @patch("download_models.snapshot_download")
    @patch("download_models.time.sleep")
    def test_token_resolution(self, mock_sleep, mock_snapshot, mock_load_settings):
        # Case 1: Environment token is set
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            with patch("builtins.print"):
                import download_models
                importlib.reload(download_models)
                self.assertEqual(download_models.HF_TOKEN, "env_token")
                
        # Case 2: Settings token is set (env token missing)
        mock_load_settings.return_value = {"hf_token": "settings_token"}
        # Temporarily clear env var
        with patch.dict(os.environ, {}):
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            with patch("builtins.print"):
                importlib.reload(download_models)
                self.assertEqual(download_models.HF_TOKEN, "settings_token")


if __name__ == "__main__":
    unittest.main()

