"""
Unit tests for settings_manager.py targeting 100% statement and branch coverage.
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Prevent system operations during import
os.environ["TESTING"] = "true"

import settings_manager


class TestSettingsManager(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.settings_file = os.path.join(self.tmp_dir, "settings.json")
        self.workspace_dir = os.path.join(self.tmp_dir, "workspace")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_dotenv_loading_variants(self):
        # We want to test manual loading of dotenv file.
        # Since it executes at import-time, we reload the module to trigger dotenv loading logic.
        # Create a mock .env file in a temporary folder
        temp_env_dir = tempfile.mkdtemp()
        env_file_path = os.path.join(temp_env_dir, ".env")
        
        with open(env_file_path, "w", encoding="utf-8") as f:
            f.write("# This is a comment\n")
            f.write("\n") # empty line
            f.write("TEST_KEY_1 = val1 \n")
            f.write("TEST_KEY_2='val2'\n")
            f.write("TEST_KEY_3=\"val3\"\n")
            f.write("TEST_KEY_4\n") # no '='

        # Patch os.path.dirname/os.path.abspath to point to temp_env_dir
        with patch("os.path.dirname") as mock_dir:
            mock_dir.return_value = temp_env_dir
            with patch.dict(os.environ, {}):
                # Ensure keys are not in env beforehand
                for key in ["TEST_KEY_1", "TEST_KEY_2", "TEST_KEY_3", "TEST_KEY_4"]:
                    if key in os.environ:
                        del os.environ[key]

                import importlib
                importlib.reload(settings_manager)

                self.assertEqual(os.environ.get("TEST_KEY_1"), "val1")
                self.assertEqual(os.environ.get("TEST_KEY_2"), "val2")
                self.assertEqual(os.environ.get("TEST_KEY_3"), "val3")
                self.assertNotIn("TEST_KEY_4", os.environ)

        # Cleanup temp env dir
        shutil.rmtree(temp_env_dir)

    def test_dotenv_loading_exception(self):
        # Trigger Exception inside dotenv parsing
        temp_env_dir = tempfile.mkdtemp()
        env_file_path = os.path.join(temp_env_dir, ".env")
        # Create a directory instead of file to trigger IsADirectoryError / PermissionError on open
        os.makedirs(env_file_path, exist_ok=True)

        with patch("os.path.dirname", return_value=temp_env_dir):
            with patch("builtins.print") as mock_print:
                import importlib
                importlib.reload(settings_manager)
                mock_print.assert_called()
                self.assertTrue(any("Error loading .env file:" in call[0][0] for call in mock_print.call_args_list))

        shutil.rmtree(temp_env_dir)

    def test_hf_home_default(self):
        # Clear HF_HOME env var and verify it gets set to workspace/huggingface default
        temp_dir = tempfile.mkdtemp()
        with patch("os.path.dirname", return_value=temp_dir):
            with patch.dict(os.environ, {}):
                if "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]
                
                import importlib
                importlib.reload(settings_manager)
                
                self.assertIn("workspace/huggingface", os.environ["HF_HOME"])
        shutil.rmtree(temp_dir)

    def test_load_settings_success_and_exception(self):
        # 1. Custom settings.json does not exist -> loads defaults
        with patch("settings_manager.SETTINGS_FILE", "/nonexistent/settings.json"):
            settings = settings_manager.load_settings()
            self.assertEqual(settings.get("server_url"), "http://localhost:8000/v1")

        # 2. Custom settings.json exists -> overrides defaults
        with open(self.settings_file, "w") as f:
            json.dump({"server_url": "http://custom-url:9000"}, f)

        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            settings = settings_manager.load_settings()
            self.assertEqual(settings.get("server_url"), "http://custom-url:9000")
            self.assertEqual(settings.get("model_name"), "allenai/olmOCR-2-7B-1025-FP8") # default preserved

        # 3. Exception in json load (invalid file)
        with open(self.settings_file, "w") as f:
            f.write("{invalid json}")

        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            with patch("builtins.print") as mock_print:
                settings = settings_manager.load_settings()
                self.assertEqual(settings.get("server_url"), "http://localhost:8000/v1") # fallback to defaults
                mock_print.assert_called()

    def test_save_settings_success_and_exception(self):
        # 1. Success
        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            res = settings_manager.save_settings({"custom": "value"})
            self.assertEqual(res, "Settings saved successfully.")
            self.assertTrue(os.path.exists(self.settings_file))
            with open(self.settings_file, "r") as f:
                data = json.load(f)
                self.assertEqual(data.get("custom"), "value")

        # 2. Exception (read-only location)
        with patch("settings_manager.SETTINGS_FILE", "/sys/readonly/settings.json"):
            res = settings_manager.save_settings({"custom": "value"})
            self.assertTrue(res.startswith("Error saving settings:"))

    def test_get_available_runs(self):
        # 1. Workspace does not exist
        with patch("settings_manager.WORKSPACE_DIR", "/nonexistent-workspace"):
            runs = settings_manager.get_available_runs()
            self.assertEqual(runs, [])

        # 2. Workspace exists with runs
        os.makedirs(self.workspace_dir)
        
        # Ok run
        run_ok = os.path.join(self.workspace_dir, "run_case_1")
        os.makedirs(os.path.join(run_ok, "markdown", "inputs"))
        with open(os.path.join(run_ok, "markdown", "inputs", "doc1.md"), "w") as f:
            f.write("# doc1")
        with open(os.path.join(run_ok, "markdown", "inputs", "doc2.md"), "w") as f:
            f.write("# doc2")

        # Empty run
        run_empty = os.path.join(self.workspace_dir, "run_case_empty")
        os.makedirs(os.path.join(run_empty, "markdown", "inputs"))

        # Folder not starting with run_
        not_run = os.path.join(self.workspace_dir, "other_case_1")
        os.makedirs(os.path.join(not_run, "markdown", "inputs"))
        with open(os.path.join(not_run, "markdown", "inputs", "doc.md"), "w") as f:
            f.write("# doc")

        # Run that is a file, not a directory
        file_run = os.path.join(self.workspace_dir, "run_is_a_file")
        with open(file_run, "w") as f:
            f.write("")

        with patch("settings_manager.WORKSPACE_DIR", self.workspace_dir):
            runs = settings_manager.get_available_runs()
            self.assertEqual(len(runs), 1)
            # Label format: "run_20260711_092213 (2 files)"
            self.assertEqual(runs[0][0], "run_case_1 (2 files)")
            self.assertEqual(runs[0][1], run_ok)


if __name__ == "__main__":
    unittest.main()
