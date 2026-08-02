"""
Unit tests for settings_manager.py targeting 100% statement and branch coverage.
"""

import json
import os
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
            f.write("\n")  # empty line
            f.write("TEST_KEY_1 = val1 \n")
            f.write("TEST_KEY_2='val2'\n")
            f.write('TEST_KEY_3="val3"\n')
            f.write("TEST_KEY_4\n")  # no '='

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
            with patch("settings_manager.logger.error") as mock_log:
                import importlib

                importlib.reload(settings_manager)
                mock_log.assert_called()
                self.assertTrue(
                    any(
                        "Error loading .env file:" in call[0][0] for call in mock_log.call_args_list
                    )
                )

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
            self.assertEqual(
                settings.get("model_name"), "allenai/olmOCR-2-7B-1025-FP8"
            )  # default preserved

        # 3. Exception in json load (invalid file)
        with open(self.settings_file, "w") as f:
            f.write("{invalid json}")

        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            with patch("settings_manager.logger.error") as mock_log:
                settings = settings_manager.load_settings()
                self.assertEqual(
                    settings.get("server_url"), "http://localhost:8000/v1"
                )  # fallback to defaults
                mock_log.assert_called()

    def test_production_inference_environment_overrides_saved_ui_state(self):
        saved = {
            "server_url": "http://old:8000/v1",
            "model_name": "old-ocr",
            "analysis_server_url": "http://old:8001/v1",
            "analysis_model_name": "old-analysis",
        }
        with open(self.settings_file, "w") as settings_file:
            json.dump(saved, settings_file)

        environment = {
            "TESTING": "false",
            "KIRAG_OCR_SERVER_URL": "http://127.0.0.1:8000/v1",
            "KIRAG_OCR_MODEL": "pinned-ocr",
            "KIRAG_ANALYSIS_SERVER_URL": "http://127.0.0.1:8002/v1",
            "KIRAG_ANALYSIS_MODEL": "pinned-analysis",
        }
        with (
            patch("settings_manager.SETTINGS_FILE", self.settings_file),
            patch.dict(os.environ, environment),
            patch("analysis_profiles.read_runtime_profile", return_value=None),
        ):
            settings = settings_manager.load_settings()

        self.assertEqual(settings["server_url"], environment["KIRAG_OCR_SERVER_URL"])
        self.assertEqual(settings["model_name"], "pinned-ocr")
        self.assertEqual(settings["analysis_server_url"], environment["KIRAG_ANALYSIS_SERVER_URL"])
        self.assertEqual(settings["analysis_model_name"], "pinned-analysis")

    def test_verified_runtime_analysis_profile_overrides_deployment_default(self):
        environment = {
            "TESTING": "false",
            "KIRAG_ANALYSIS_SERVER_URL": "http://127.0.0.1:8002/v1",
            "KIRAG_ANALYSIS_MODEL": "Qwen/Qwen3.6-35B-A3B",
        }
        runtime = {
            "model": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        }
        with (
            patch("settings_manager.SETTINGS_FILE", self.settings_file),
            patch.dict(os.environ, environment),
            patch("analysis_profiles.read_runtime_profile", return_value=runtime),
        ):
            settings = settings_manager.load_settings()
        self.assertEqual(settings["analysis_model_name"], "google/gemma-4-31B-it")
        self.assertEqual(settings["analysis_server_url"], "http://127.0.0.1:8002/v1")

    def test_save_settings_success_and_exception(self):
        # 1. Success
        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            res = settings_manager.save_settings({"custom": "value"})
            self.assertEqual(res, "Settings saved successfully.")
            self.assertTrue(os.path.exists(self.settings_file))
            with open(self.settings_file) as f:
                data = json.load(f)
                self.assertEqual(data.get("custom"), "value")

        # 2. Exception (read-only location)
        with patch("settings_manager.SETTINGS_FILE", "/sys/readonly/settings.json"):
            res = settings_manager.save_settings({"custom": "value"})
            self.assertTrue(res.startswith("Error saving settings:"))

    def test_save_settings_replaces_complete_json_atomically(self):
        with (
            patch("settings_manager.SETTINGS_FILE", self.settings_file),
            patch("settings_manager.os.replace", wraps=os.replace) as replace,
        ):
            res = settings_manager.save_settings({"custom": "complete"})

        self.assertEqual(res, "Settings saved successfully.")
        replace.assert_called_once()
        source, destination = replace.call_args.args
        self.assertEqual(destination, self.settings_file)
        self.assertNotEqual(source, destination)
        with open(self.settings_file, encoding="utf-8") as saved:
            self.assertEqual(json.load(saved), {"custom": "complete"})

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
            self.assertIn("run_case_1 (2 files)", runs[0][0])
            self.assertEqual(runs[0][1], run_ok)

    def test_safe_stream(self):
        class MockOriginal:
            def __init__(self):
                self.mode = "w"

            def write(self, data):
                if data == "errno5":
                    err = OSError()
                    err.errno = 5
                    raise err
                elif data == "errno2":
                    err = OSError()
                    err.errno = 2
                    raise err
                elif data == "error":
                    raise Exception("generic error")
                return len(data)

            def flush(self):
                if getattr(self, "flush_fail", None) == 5:
                    err = OSError()
                    err.errno = 5
                    raise err
                elif getattr(self, "flush_fail", None) == 2:
                    err = OSError()
                    err.errno = 2
                    raise err
                elif getattr(self, "flush_fail", None) == "err":
                    raise Exception("generic error")

            def isatty(self):
                if getattr(self, "isatty_fail", False):
                    raise Exception("isatty error")
                return True

        mock_orig = MockOriginal()
        stream = settings_manager.SafeStream(mock_orig)

        # write tests
        stream.write("ok")
        stream.write("errno5")  # should pass silently
        with self.assertRaises(OSError):
            stream.write("errno2")
        stream.write("error")  # generic Exception caught and passed

        # flush tests
        stream.flush()
        mock_orig.flush_fail = 5
        stream.flush()  # should pass silently
        mock_orig.flush_fail = 2
        with self.assertRaises(OSError):
            stream.flush()
        mock_orig.flush_fail = "err"
        stream.flush()  # generic Exception caught and passed

        # isatty tests
        mock_orig.isatty_fail = False
        self.assertTrue(stream.isatty())
        mock_orig.isatty_fail = True
        self.assertFalse(stream.isatty())

        # getattr test
        self.assertEqual(stream.mode, "w")

        # None stream
        none_stream = settings_manager.SafeStream(None)
        none_stream.write("test")
        none_stream.flush()
        self.assertFalse(none_stream.isatty())

    def test_resolve_hf_home_fallback(self):
        with patch("os.access", return_value=False), patch("os.path.isdir", return_value=False):
            hf_home = settings_manager._resolve_hf_home()
            self.assertTrue(hf_home.endswith(".cache/huggingface") or ".cache" in hf_home)

    def test_load_settings_hf_token_handling(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env_token_val"}):
            with patch("settings_manager.SETTINGS_FILE", self.settings_file):
                with open(self.settings_file, "w") as f:
                    json.dump({"hf_token": ""}, f)
                settings = settings_manager.load_settings()
                self.assertEqual(settings.get("hf_token"), "env_token_val")

    def test_save_settings_analysis_model_copy(self):
        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            res = settings_manager.save_settings({"model_name": "custom/model-fp8"})
            self.assertIn("saved successfully", res)
            loaded = settings_manager.load_settings()
            self.assertEqual(loaded.get("analysis_model_name"), "custom/model-fp8")

    def test_load_settings_missing_analysis_model_fallback(self):
        with patch("settings_manager.SETTINGS_FILE", self.settings_file):
            with open(self.settings_file, "w") as f:
                json.dump({"server_url": "http://localhost:8000/v1", "analysis_model_name": ""}, f)
            s = settings_manager.load_settings()
            self.assertEqual(s.get("analysis_model_name"), "allenai/olmOCR-2-7B-1025-FP8")

    def test_delete_run_directory(self):
        # Empty arg
        self.assertFalse(settings_manager.delete_run_directory(""))

        test_run_dir = os.path.join(self.workspace_dir, "run_direct")
        os.makedirs(test_run_dir, exist_ok=True)
        with patch("settings_manager.WORKSPACE_DIR", self.workspace_dir):
            # Absolute paths are no longer accepted at this boundary.
            self.assertFalse(settings_manager.delete_run_directory(test_run_dir))
            self.assertTrue(settings_manager.delete_run_directory("run_direct"))
        self.assertFalse(os.path.exists(test_run_dir))

        # Candidate workspace search
        test_ws = os.path.join(self.tmp_dir, "ws_cand")
        test_sub = os.path.join(test_ws, "run_cand_123")
        os.makedirs(test_sub, exist_ok=True)
        with patch.object(settings_manager, "WORKSPACE_DIR", test_ws):
            self.assertTrue(settings_manager.delete_run_directory("run_cand_123"))

    def test_get_available_runs_with_indexed_status(self):
        ws_dir = os.path.join(self.tmp_dir, "workspace_test")
        run_dir = os.path.join(ws_dir, "run_20240101_100000")
        md_dir = os.path.join(run_dir, "markdown", "inputs")
        os.makedirs(md_dir, exist_ok=True)
        with open(os.path.join(md_dir, "doc.md"), "w") as f:
            f.write("# Sample MD")

        with patch("rag.db.is_run_indexed", return_value=True):
            runs = settings_manager.get_available_runs(ws_dir)
            self.assertEqual(len(runs), 1)
            self.assertIn("✅", runs[0][0])
            self.assertIn("[INDEXED]", runs[0][0])

        with patch("rag.db.is_run_indexed", return_value=False):
            runs2 = settings_manager.get_available_runs(ws_dir)
            self.assertEqual(len(runs2), 1)
            self.assertIn("📄", runs2[0][0])


if __name__ == "__main__":
    unittest.main()
