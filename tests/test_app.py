import os
import unittest
import subprocess
from unittest.mock import patch, MagicMock, mock_open
import io
import tempfile
import shutil

# Prevent atexit hooks from executing Docker/subprocess operations during import/tests
os.environ["TESTING"] = "true"

import app
import docker_manager
import process_state

class TestOLMOCRApp(unittest.TestCase):

    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.workspace, "run_case")
        self.upload_dir = os.path.join(self.workspace, "uploads")
        os.makedirs(self.run_dir)
        os.makedirs(self.upload_dir)
        self.pdf_path = os.path.join(self.upload_dir, "test.pdf")
        with open(self.pdf_path, "wb") as pdf:
            pdf.write(b"%PDF-1.4 test")
        self.workspace_patchers = [
            patch("pipeline_manager.WORKSPACE_DIR", self.workspace),
            patch("pdf_manager.WORKSPACE_DIR", self.workspace),
        ]
        for patcher in self.workspace_patchers:
            patcher.start()
        self.docker_env_patcher = patch.dict(
            os.environ,
            {"OLMOCR_VLLM_IMAGE": docker_manager.DEFAULT_VLLM_IMAGE},
        )
        self.docker_env_patcher.start()
        # Reset active runs between tests
        with process_state.active_runs_lock:
            process_state.active_runs.clear()

    def tearDown(self):
        self.docker_env_patcher.stop()
        for patcher in reversed(self.workspace_patchers):
            patcher.stop()
        shutil.rmtree(self.workspace)

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
        mock_run.return_value = MagicMock(returncode=0, stdout="running\ttrue\n")
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

    @patch("docker_manager.check_server_ready")
    @patch("docker_manager.get_docker_status")
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
    @patch("docker_manager.get_docker_status")
    def test_start_docker_container(self, mock_status, mock_run):
        # Exited -> Success
        mock_status.return_value = "exited"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = app.start_docker_container()
        self.assertTrue(success)
        self.assertIn("started successfully", msg)

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
        self.assertIn("already running", msg)

    @patch("subprocess.run")
    @patch("docker_manager.get_docker_status")
    def test_stop_docker_container(self, mock_status, mock_run):
        mock_status.return_value = "running"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = app.stop_docker_container()
        self.assertTrue(success)
        self.assertIn("stopped successfully", msg)

    @patch("subprocess.run")
    @patch("docker_manager.get_docker_status")
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
    def test_make_zip(self, mock_zip):
        markdown_dir = os.path.join(self.workspace, "markdown")
        os.makedirs(markdown_dir)
        markdown_file = os.path.join(markdown_dir, "file1.md")
        with open(markdown_file, "w", encoding="utf-8") as markdown:
            markdown.write("# test")
        zip_path = os.path.join(self.workspace, "out.zip")

        app.make_zip(markdown_dir, zip_path)

        mock_zip.assert_called_once_with(unittest.mock.ANY, "w", unittest.mock.ANY)
        self.assertEqual(str(mock_zip.call_args.args[0]), zip_path)
        mock_zip.return_value.__enter__().write.assert_called_once_with(
            unittest.mock.ANY, "file1.md"
        )

    def test_load_markdown_content(self):
        # Run info not found
        content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "invalid_run")
        self.assertEqual(content_r, "Run info not found.")

        # File not found
        with process_state.active_runs_lock:
            process_state.active_runs["test_run"] = {"run_dir": self.run_dir}
        content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "test_run")
        self.assertEqual(content_r, "File not found.")

        # File read success
        temp_dir = self.run_dir
        try:
            inputs_dir = os.path.join(temp_dir, "markdown", "inputs")
            os.makedirs(inputs_dir)
            doc_path = os.path.join(inputs_dir, "0_doc.md")
            with open(doc_path, "w") as f:
                f.write("hello markdown")
            
            with process_state.active_runs_lock:
                process_state.active_runs["test_run"] = {"run_dir": temp_dir}
            
            content_r, content_h, file_p = app.load_markdown_content("0_doc.md", "test_run")
            self.assertEqual(content_r, "hello markdown")
            self.assertEqual(file_p, doc_path)
        finally:
            shutil.rmtree(os.path.join(temp_dir, "markdown"))

    def test_stop_processing(self):
        process_state.active_runs.clear()
        # Empty
        self.assertTrue("No active process to stop" in app.stop_processing(""))
        # Missing
        self.assertTrue("Process not found" in app.stop_processing("missing_run"))
        # Active
        mock_proc = MagicMock()
        with process_state.active_runs_lock:
            process_state.active_runs["run1"] = {"proc": mock_proc, "stop": False}
        res = app.stop_processing("run1")
        self.assertTrue("Stop request sent" in res)
        mock_proc.terminate.assert_called_once()
        process_state.active_runs.clear()

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
        mock_file.name = self.pdf_path
        
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
        self.assertTrue("Running" in yield_1[1]) # Status badge is now a plain HTML string
        
        # Step 2: Loop yield (due to 0.3s elapsed time change, which triggers the 0.2s throttle check)
        yield_2 = next(gen)
        # Verify yields are processing output log
        self.assertTrue("completed_pages 1" in yield_2[0])

        # Step 3: Consume remaining steps until completion
        results = list(gen)
        final_yield = results[-1]
        self.assertTrue("PROCESS EXITED WITH CODE 0" in final_yield[0])
        self.assertTrue("Success" in final_yield[1])

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("shutil.copy")
    @patch("subprocess.Popen")
    @patch("time.monotonic")
    def test_process_pdfs_progress_parsing(self, mock_time, mock_popen, mock_copy, mock_pdf, mock_get):
        # Set up mocks
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf.return_value.pages = range(50)
        
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
        mock_file.name = self.pdf_path
        
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

    def test_get_dir_size_and_format_size(self):
        self.assertEqual(app.format_size(100), "100 B")
        self.assertEqual(app.format_size(1500), "1.46 KB")
        self.assertEqual(app.format_size(1500000), "1.43 MB")
        self.assertEqual(app.format_size(1500000000), "1.40 GB")

        # Create a temp dir with some files
        temp_dir = tempfile.mkdtemp()
        try:
            f1 = os.path.join(temp_dir, "file1.txt")
            with open(f1, "w") as f:
                f.write("a" * 100) # 100 bytes
            
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            f2 = os.path.join(sub_dir, "file2.txt")
            with open(f2, "w") as f:
                f.write("b" * 200) # 200 bytes

            total_size = app.get_dir_size(temp_dir)
            self.assertEqual(total_size, 300)
        finally:
            shutil.rmtree(temp_dir)

    def test_perform_reset_cleanup(self):
        active_run = os.path.join(self.workspace, "run_active")
        obsolete_run = os.path.join(self.workspace, "run_obsolete")
        os.makedirs(active_run)
        os.makedirs(obsolete_run)
        with open(os.path.join(obsolete_run, "result.md"), "w", encoding="utf-8") as result:
            result.write("obsolete")

        with process_state.active_runs_lock:
            process_state.active_runs["active_id"] = {
                "proc": MagicMock(),
                "completed": False,
                "run_dir": active_run,
            }
            process_state.active_runs["active_id"]["proc"].poll.return_value = None

        res = app.perform_reset_cleanup(
            clean_runs=True,
            clean_gradio=False,
            clean_pycache=False,
            clean_hf=False,
            workspace_dir=self.workspace,
        )

        self.assertTrue(os.path.isdir(active_run))
        self.assertFalse(os.path.exists(obsolete_run))
        self.assertIn("Obsolete run directory: `run_obsolete`", res)

    @patch("app.process_pdfs")
    def test_process_pdfs_ui_wrapper(self, mock_process_pdfs):
        from pipeline_manager import PipelineResult

        mock_result = PipelineResult(
            "log", "badge", "progress", "pages", "fail",
            "selector", "zip", "indiv", "start", "run_id",
            "status_table", "manifest", "stop",
        )
        mock_process_pdfs.return_value = [mock_result]
        
        gen = app.process_pdfs_ui_wrapper("arg1", kwarg1="val1")
        results = list(gen)
        self.assertEqual(len(results), 1)
        # The adapter passes through non-dict values unchanged
        self.assertEqual(results[0][0], "log")       # log_text
        self.assertEqual(results[0][2], "progress")   # progress_bar
        self.assertEqual(results[0][9], "run_id")     # active_run_id
        mock_process_pdfs.assert_called_once_with("arg1", kwarg1="val1")

    @patch("app.gr.update")
    def test_update_max_content_length(self, mock_gr_update):
        app.update_max_content_length("unknown_model", 200000)
        mock_gr_update.assert_called_once_with(maximum=131072, value=131072)

if __name__ == "__main__":
    unittest.main()
