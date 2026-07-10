import os
import sys
import json
import unittest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
import io
import tempfile
import shutil

# Prevent atexit hooks from executing Docker/subprocess operations during import/tests
os.environ["TESTING"] = "true"

import app

class TestOLMOCRApp(unittest.TestCase):

    def setUp(self):
        # Reset active runs between tests
        with app.active_runs_lock:
            app.active_runs.clear()

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data='{"server_url": "http://test-server:8000/v1"}')
    def test_load_settings_exists(self, mock_file, mock_exists):
        mock_exists.return_value = True
        settings = app.load_settings()
        self.assertEqual(settings["server_url"], "http://test-server:8000/v1")
        self.assertEqual(settings["workers"], 4) # Default retained

    @patch("os.path.exists")
    def test_load_settings_not_exists(self, mock_exists):
        mock_exists.return_value = False
        settings = app.load_settings()
        self.assertEqual(settings["server_url"], "http://localhost:8000/v1")

    @patch("os.path.exists")
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_load_settings_error(self, mock_file, mock_exists):
        mock_exists.return_value = True
        settings = app.load_settings()
        self.assertEqual(settings["server_url"], "http://localhost:8000/v1") # Returns default on fail

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_settings_success(self, mock_file, mock_makedirs):
        settings_to_save = {"server_url": "http://custom:123/v1"}
        res = app.save_settings(settings_to_save)
        self.assertEqual(res, "Settings saved successfully.")
        mock_makedirs.assert_called_once()
        mock_file().write.assert_called()

    @patch("os.makedirs", side_effect=OSError("Write failure"))
    def test_save_settings_failure(self, mock_makedirs):
        settings_to_save = {"server_url": "http://custom:123/v1"}
        res = app.save_settings(settings_to_save)
        self.assertTrue("Error saving settings" in res)

    @patch("subprocess.run")
    def test_get_docker_status(self, mock_run):
        # Case 1: running
        mock_run.return_value = MagicMock(returncode=0, stdout="running\n")
        self.assertEqual(app.get_docker_status(), "running")

        # Case 2: not found
        mock_run.return_value = MagicMock(returncode=1, stderr="no such inspect object")
        self.assertEqual(app.get_docker_status(), "not_found")

        # Case 3: error
        mock_run.return_value = MagicMock(returncode=1, stderr="other error")
        self.assertEqual(app.get_docker_status(), "error")

        # Case 4: exception
        mock_run.side_effect = Exception("Docker daemon dead")
        self.assertEqual(app.get_docker_status(), "error")

    @patch("httpx.get")
    def test_check_server_ready(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(app.check_server_ready(8000))

        mock_get.side_effect = Exception("Conn refused")
        self.assertFalse(app.check_server_ready(8000))

    @patch("app.check_server_ready")
    @patch("app.get_docker_status")
    def test_get_docker_status_str(self, mock_status, mock_ready):
        mock_status.return_value = "not_found"
        state, html = app.get_docker_status_str(8000)
        self.assertEqual(state, "not_found")
        self.assertTrue("Docker: Not Created" in html)

        mock_status.return_value = "exited"
        state, html = app.get_docker_status_str(8000)
        self.assertEqual(state, "stopped")
        self.assertTrue("Docker: Stopped" in html)

        mock_status.return_value = "running"
        mock_ready.return_value = False
        state, html = app.get_docker_status_str(8000)
        self.assertEqual(state, "starting")
        self.assertTrue("Starting / Loading Model" in html)

        mock_ready.return_value = True
        state, html = app.get_docker_status_str(8000)
        self.assertEqual(state, "ready")
        self.assertTrue("Inference Server: Ready" in html)

    @patch("subprocess.run")
    @patch("app.get_docker_status")
    def test_start_docker_container(self, mock_status, mock_run):
        # Exited -> Success
        mock_status.return_value = "exited"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = app.start_docker_container()
        self.assertTrue(success)
        self.assertEqual(msg, "Container started successfully.")

        # Exited -> Failure
        mock_status.return_value = "exited"
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker start", stderr=b"Docker error")
        success, msg = app.start_docker_container()
        self.assertFalse(success)
        self.assertTrue("Failed to start container" in msg)

        # Running -> Safe Return
        mock_run.side_effect = None
        mock_status.return_value = "running"
        success, msg = app.start_docker_container()
        self.assertTrue(success)
        self.assertEqual(msg, "Container is already running.")

    @patch("subprocess.run")
    @patch("app.get_docker_status")
    def test_stop_docker_container(self, mock_status, mock_run):
        mock_status.return_value = "running"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = app.stop_docker_container()
        self.assertTrue(success)
        self.assertEqual(msg, "Container stopped successfully.")

    @patch("subprocess.run")
    @patch("app.get_docker_status")
    def test_create_docker_container(self, mock_status, mock_run):
        mock_status.return_value = "running"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = app.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertTrue(success)
        self.assertEqual(msg, "Container created and started successfully.")

    def test_html_builders(self):
        progress_html = app.make_progress_bar_html(2, 10, 10.0)
        self.assertTrue("2/10 Pages" in progress_html)
        self.assertTrue("20%" in progress_html)
        self.assertTrue("10s elapsed" in progress_html)

        manifest_html = app.make_upload_manifest_html({0: "doc.pdf"}, {0: 5}, {0: 1024})
        self.assertTrue("doc.pdf" in manifest_html)
        self.assertTrue("1.0 KB" in manifest_html)
        self.assertTrue("Total (1 files)" in manifest_html)

        status_html = app.make_file_status_html({0: "doc.pdf"}, {0: 5}, {0}, {1})
        self.assertTrue("doc.pdf" in status_html)
        self.assertTrue("✓ Done" in status_html)

    @patch("zipfile.ZipFile")
    @patch("os.walk")
    def test_make_zip(self, mock_walk, mock_zip):
        mock_walk.return_value = [("/tmp/dir", [], ["file1.md", "file2.txt"])]
        app.make_zip("/tmp/dir", "/tmp/out.zip")
        mock_zip.assert_called_once_with("/tmp/out.zip", "w", unittest.mock.ANY)
        mock_zip.return_value.__enter__().write.assert_called_once_with("/tmp/dir/file1.md", "file1.md")

    def test_load_markdown_content(self):
        # Run info not found
        content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "invalid_run")
        self.assertEqual(content_r, "Run info not found.")

        # File not found
        with app.active_runs_lock:
            app.active_runs["test_run"] = {"run_dir": "/tmp/nonexistent_dir"}
        content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "test_run")
        self.assertEqual(content_r, "File not found.")

        # File read success
        temp_dir = tempfile.mkdtemp()
        try:
            inputs_dir = os.path.join(temp_dir, "markdown", "inputs")
            os.makedirs(inputs_dir)
            doc_path = os.path.join(inputs_dir, "0_doc.md")
            with open(doc_path, "w") as f:
                f.write("hello markdown")
            
            with app.active_runs_lock:
                app.active_runs["test_run"] = {"run_dir": temp_dir}
            
            content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "test_run")
            self.assertEqual(content_r, "hello markdown")
            self.assertEqual(file_p, doc_path)
        finally:
            shutil.rmtree(temp_dir)

    @patch("app.active_runs", {})
    def test_stop_processing(self):
        # Empty
        self.assertTrue("No active process to stop" in app.stop_processing(""))
        # Missing
        self.assertTrue("Process not found" in app.stop_processing("missing_run"))
        # Active
        mock_proc = MagicMock()
        with app.active_runs_lock:
            app.active_runs["run1"] = {"proc": mock_proc, "stop": False}
        res = app.stop_processing("run1")
        self.assertTrue("Stop request sent" in res)
        mock_proc.terminate.assert_called_once()

    @patch("httpx.get")
    @patch("pypdf.PdfReader")
    @patch("shutil.copy")
    @patch("subprocess.Popen")
    @patch("time.monotonic")
    def test_process_pdfs_generator(self, mock_time, mock_popen, mock_copy, mock_pdf, mock_get):
        # Set up mocks
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf.return_value.pages = [1, 2] # 2 pages
        
        # Mock time generator to avoid StopIteration
        time_counter = [10.0]
        def time_side_effect():
            time_counter[0] += 0.25
            return time_counter[0]
        mock_time.side_effect = time_side_effect
        
        # Mock process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = 0
        
        # Mock stdout lines from subprocess
        mock_stdout = io.StringIO("completed_pages 1\nvllm running req: 1 queue req: 0\ncompleted_pages 2\n")
        mock_proc.stdout = mock_stdout
        mock_popen.return_value = mock_proc

        # Call generator
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = app.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=2,
            max_concurrent=10,
            max_retries=3,
            target_dim=1024,
            guided_decoding=True
        )

        # Step 1: Initial yield
        yield_1 = next(gen)
        self.assertEqual(yield_1[0], "Initializing pipeline...")
        self.assertTrue("Running" in yield_1[1]["value"]) # Status badge gr.update
        
        # Step 2: Loop yield (due to 0.3s elapsed time change, which triggers the 0.2s throttle check)
        yield_2 = next(gen)
        # Verify yields are processing output log
        self.assertTrue("completed_pages 1" in yield_2[0])

        # Step 3: Consume remaining steps until completion
        results = list(gen)
        final_yield = results[-1]
        self.assertTrue("PROCESS EXITED WITH CODE 0" in final_yield[0])
        self.assertTrue("Success" in final_yield[1]["value"])

    @patch("httpx.get")
    @patch("pypdf.PdfReader")
    @patch("shutil.copy")
    @patch("subprocess.Popen")
    @patch("time.monotonic")
    def test_process_pdfs_progress_parsing(self, mock_time, mock_popen, mock_copy, mock_pdf, mock_get):
        # Set up mocks
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf.return_value.pages = [1, 2]
        
        # Mock time generator to avoid StopIteration
        time_counter = [10.0]
        def time_side_effect():
            time_counter[0] += 0.25
            return time_counter[0]
        mock_time.side_effect = time_side_effect
        
        # Mock process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = 0
        
        # Mock stdout lines containing worker table and final summary metrics
        mock_stdout_content = (
            "Worker ID | finished | started\n"
            "----------+----------+---------\n"
            "0         | 15       | 205     \n"
            "1         | 20       | 205     \n"
            "Completed pages: 45\n"
            "Failed pages: 3\n"
        )
        mock_proc.stdout = io.StringIO(mock_stdout_content)
        mock_popen.return_value = mock_proc

        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = app.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=2,
            max_concurrent=10,
            max_retries=3,
            target_dim=1024,
            guided_decoding=True
        )

        # Consume generator and collect yields
        yields = list(gen)
        
        # We check the final yield output values
        final_yield = yields[-1]
        
        # Verify completed pages updated from the worker table sum (15+20=35)
        # and then from the summary "Completed pages: 45"
        self.assertTrue("Completed Pages" in final_yield[3]["value"])
        self.assertTrue("45" in final_yield[3]["value"])
        
        # Verify failed pages updated from "Failed pages: 3"
        self.assertTrue("Failed Pages" in final_yield[4]["value"])
        self.assertTrue("3" in final_yield[4]["value"])

if __name__ == "__main__":
    unittest.main()
