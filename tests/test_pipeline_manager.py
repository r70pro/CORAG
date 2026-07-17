"""
Unit tests for pipeline_manager.py.
"""

import os
import queue
import unittest
import subprocess
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

import pipeline_manager
import process_state


class TestPipelineManager(unittest.TestCase):

    def setUp(self):
        with process_state.active_runs_lock:
            process_state.active_runs.clear()

    def test_enqueue_output_success(self):
        mock_out = MagicMock()
        mock_out.readline.side_effect = ["line1\n", "line2\n", ""]
        q = queue.Queue()
        
        pipeline_manager.enqueue_output(mock_out, q)
        
        self.assertEqual(q.get(), "line1\n")
        self.assertEqual(q.get(), "line2\n")
        self.assertTrue(q.empty())
        mock_out.close.assert_called_once()

    def test_enqueue_output_exception(self):
        mock_out = MagicMock()
        mock_out.readline.side_effect = Exception("Read error")
        q = queue.Queue()
        
        pipeline_manager.enqueue_output(mock_out, q)
        
        self.assertTrue(q.empty())
        mock_out.close.assert_called_once()

    def test_make_empty_yield(self):
        res = pipeline_manager._make_empty_yield("log text", "badge", "progress", start_interactive=True, run_id="run123")
        self.assertEqual(res[0], "log text")
        self.assertEqual(res[9], "run123")

    @patch("httpx.get")
    def test_process_pdfs_no_files(self, mock_get):
        gen = pipeline_manager.process_pdfs(
            files=[],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "No files uploaded.")

    @patch("httpx.get")
    def test_process_pdfs_invalid_files(self, mock_get):
        gen = pipeline_manager.process_pdfs(
            files=[None],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "Invalid file uploads.")

    @patch("httpx.get")
    def test_process_pdfs_preflight_failed(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertEqual(len(res), 1)
        self.assertTrue("Pre-flight check failed" in res[0][0])

    @patch("httpx.get")
    def test_process_pdfs_preflight_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertEqual(len(res), 1)
        self.assertTrue("Pre-flight check failed" in res[0][0])

    def test_stop_processing_empty(self):
        res = pipeline_manager.stop_processing("")
        self.assertTrue("No active process to stop" in res)

    def test_stop_processing_not_found(self):
        res = pipeline_manager.stop_processing("missing_run")
        self.assertTrue("Process not found" in res)

    def test_stop_processing_active(self):
        mock_proc = MagicMock()
        with process_state.active_runs_lock:
            process_state.active_runs["run123"] = {
                "proc": mock_proc,
                "stop": False,
                "run_dir": "/tmp/run"
            }
        res = pipeline_manager.stop_processing("run123")
        self.assertTrue("Stop request sent" in res)
        mock_proc.terminate.assert_called_once()

    def test_cleanup_active_runs(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None # Running
        
        active_runs_dict = process_state.active_runs
        active_runs_dict["run123"] = {
            "proc": mock_proc,
            "stop": False
        }
        
        with patch.dict(os.environ, {"TESTING": "false"}):
            pipeline_manager.cleanup_active_runs()
        
        mock_proc.terminate.assert_called_once()
        # Clean up
        active_runs_dict.clear()

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_full_flow_success(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        # Setup mocks
        mock_get.return_value = MagicMock(status_code=200)
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()] * 2
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        mock_proc_inst = MagicMock()
        mock_proc_inst.returncode = 0
        mock_popen.return_value = mock_proc_inst

        # Simulate logs from stdout
        mock_proc_inst.stdout.readline.side_effect = [
            "Worker ID | finished | started\n",
            "0 | 1 | 0\n",
            "Completed pages: 1\n",
            "vllm running req: 1 queue req: 0\n",
            "[INFO] Page 1 of inputs/test.pdf finished successfully.\n",
            "failed_pages 0\n",
            ""
        ]

        # Use local patches to mock exists and listdir to test dropdown logic at the end
        from unittest.mock import patch as local_patch
        real_exists = os.path.exists
        def exists_side_effect(path):
            if "markdown" in path:
                return True
            return real_exists(path)
        def listdir_side_effect(path):
            if "markdown" in path:
                return ["0_test.md"]
            return []

        with local_patch("os.path.exists", side_effect=exists_side_effect), \
             local_patch("os.listdir", side_effect=listdir_side_effect):
            
            # Pass mixture of inputs (mock_file, string, dict) to cover branch types
            gen = pipeline_manager.process_pdfs(
                files=[mock_file, "string_path.pdf", {"path": "dict_path.pdf"}],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=4,
                max_concurrent=20,
                max_retries=8,
                target_dim=1288,
                guided_decoding=True
            )

            res = list(gen)
        
        self.assertTrue(len(res) > 1)
        # Check final yield has "Success" status badge
        final_yield = res[-1]
        self.assertTrue("Success" in str(final_yield[1]))

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_user_terminated(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        # Setup mocks
        mock_get.return_value = MagicMock(status_code=200)
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()] * 2
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        mock_proc_inst = MagicMock()
        mock_proc_inst.wait.side_effect = subprocess.TimeoutExpired(cmd="vllm", timeout=3)
        mock_popen.return_value = mock_proc_inst

        # Simulate logs from stdout
        mock_proc_inst.stdout.readline.side_effect = [
            "Initializing...\n",
            "vllm running req: 1 queue req: 0\n",
            ""
        ]

        # Use local patches to mock exists and listdir to test dropdown value set in stop block
        from unittest.mock import patch as local_patch
        real_exists = os.path.exists
        run_id_ref = [None]
        def exists_side_effect(path):
            if "markdown" in path:
                return True
            return real_exists(path)
        def listdir_side_effect(path):
            if "markdown" in path:
                rid = run_id_ref[0]
                if rid:
                    with process_state.active_runs_lock:
                        if rid in process_state.active_runs:
                            process_state.active_runs[rid]["stop"] = True
                return ["0_test.md"]
            return []

        with local_patch("os.path.exists", side_effect=exists_side_effect), \
             local_patch("os.listdir", side_effect=listdir_side_effect), \
             patch("time.monotonic", return_value=100.0):
            
            # Start generator
            gen = pipeline_manager.process_pdfs(
                files=[mock_file],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=4,
                max_concurrent=20,
                max_retries=8,
                target_dim=1288,
                guided_decoding=True
            )

            # Consume the first yield to obtain the run_id
            res = []
            step_1 = next(gen)
            res.append(step_1)
            run_id_ref[0] = step_1[9]

            # Consume the remaining steps
            res.extend(list(gen))

        self.assertTrue(len(res) > 0)
        self.assertTrue("PROCESS TERMINATED" in str(res[-1][0]))
        # Verify proc.kill is called due to TimeoutExpired
        mock_proc_inst.kill.assert_called_once()

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_user_terminated_empty_choices(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()] * 2
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        mock_proc_inst = MagicMock()
        mock_proc_inst.wait.side_effect = subprocess.TimeoutExpired(cmd="vllm", timeout=3)
        mock_popen.return_value = mock_proc_inst

        mock_proc_inst.stdout.readline.side_effect = [
            "Initializing...\n",
            ""
        ]

        # No exists/listdir mocks, so choices remains empty
        with patch("time.monotonic", return_value=100.0):
            gen = pipeline_manager.process_pdfs(
                files=[mock_file],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=4,
                max_concurrent=20,
                max_retries=8,
                target_dim=1288,
                guided_decoding=True
            )

            res = []
            step_1 = next(gen)
            res.append(step_1)
            run_id = step_1[9]
            if run_id:
                with process_state.active_runs_lock:
                    if run_id in process_state.active_runs:
                        process_state.active_runs[run_id]["stop"] = True

            res.extend(list(gen))

        self.assertTrue(len(res) > 0)
        self.assertTrue("PROCESS TERMINATED" in str(res[-1][0]))
        mock_proc_inst.kill.assert_called_once()

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_choices_empty(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"

        mock_proc_inst = MagicMock()
        mock_proc_inst.returncode = 0
        mock_proc_inst.stdout.readline.side_effect = [""]
        mock_popen.return_value = mock_proc_inst

        # Mock exists to return True and listdir to return empty list, so choices is empty at the end
        from unittest.mock import patch as local_patch
        real_exists = os.path.exists
        def exists_side_effect(path):
            if "markdown" in path:
                return True
            return real_exists(path)
        def listdir_side_effect(path):
            if "markdown" in path:
                return []
            return []

        with local_patch("os.path.exists", side_effect=exists_side_effect), \
             local_patch("os.listdir", side_effect=listdir_side_effect):
            gen = pipeline_manager.process_pdfs(
                files=[mock_file],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=2,
                max_concurrent=10,
                max_retries=3,
                target_dim=1024,
                guided_decoding=True
            )
            res = list(gen)
            self.assertTrue(len(res) > 0)

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_popen_exception(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()] * 2
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"

        # Force Popen to raise Exception
        mock_popen.side_effect = OSError("Permission denied")

        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertTrue(len(res) > 0)
        self.assertTrue("Failed to start pipeline process" in str(res[-1][0]))

    @patch("httpx.get")
    @patch("pipeline_manager.PdfReader")
    @patch("subprocess.Popen")
    @patch("shutil.copy")
    def test_process_pdfs_edge_scenarios(self, mock_copy, mock_popen, mock_pdf_reader, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf_reader.return_value = mock_pdf

        mock_file = MagicMock()
        mock_file.name = "test.pdf"

        # 1. Test process_pdfs with guided_decoding = False (147->150 branch)
        mock_proc_inst1 = MagicMock()
        mock_proc_inst1.returncode = 0
        mock_proc_inst1.stdout.readline.side_effect = ["", ""]
        mock_popen.return_value = mock_proc_inst1

        gen1 = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=2,
            max_concurrent=10,
            max_retries=3,
            target_dim=1024,
            guided_decoding=False
        )
        res1 = list(gen1)
        self.assertTrue(len(res1) > 0)

        # 2. Test run_id missing in process_state.active_runs when popen starts (160->179 branch)
        def popen_side_effect(*args, **kwargs):
            with process_state.active_runs_lock:
                process_state.active_runs.clear()
            return mock_proc_inst1

        mock_popen.side_effect = popen_side_effect
        gen2 = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="test-model",
            workers=2,
            max_concurrent=10,
            max_retries=3,
            target_dim=1024,
            guided_decoding=True
        )
        res2 = list(gen2)
        self.assertTrue(len(res2) > 0)
        mock_popen.side_effect = None

        # 3. Test log parsing with errored column and standalone queues, failed pages parsing,
        # and queue.Empty waiting block.
        import time
        
        lines_to_yield = [
            "0 | 1 | 2\n", # 277: Triggers 3 columns header fallback (since current_headers is empty)
            "0 | 1 | 2 | 3\n", # 279: Triggers 4 columns header fallback (since len(parts) != len(current_headers))
            "0 | 1 | 2 | 3 | 4\n", # parts length > current_headers (causes continue)
            "Completed pages:\n", # Triggers match_c is False branch (300->307)
            "Failed pages:\n", # Triggers match_f is False branch (309->316)
            "Running: 4 Waiting: 1\n", # standalone queue
            "failed_pages 2.0\n",
            "failed_pages 2.5.5\n", # ValueError in failed pages float conversion
            "failed_pages abc\n", # match_f is False
            "DELAY", # Triggers queue.Empty branch (257-259)
            "DELAY", # Triggers listdir with choice_tuple already in streaming_choices (328->330)
            ""
        ]

        idx = 0
        def readline_mock(*args, **kwargs):
            nonlocal idx
            if idx >= len(lines_to_yield):
                return ""
            val = lines_to_yield[idx]
            idx += 1
            if val == "DELAY":
                time.sleep(0.1)
                return "delayed line\n"
            return val

        mock_proc_inst2 = MagicMock()
        mock_proc_inst2.returncode = 1
        mock_proc_inst2.stdout.readline.side_effect = readline_mock
        mock_popen.return_value = mock_proc_inst2

        # Use local patches for exists and listdir to avoid breaking os.makedirs
        from unittest.mock import patch as local_patch
        real_exists = os.path.exists
        
        def exists_side_effect(path):
            if "markdown" in path:
                return True
            return real_exists(path)

        def listdir_side_effect(path):
            if "markdown" in path:
                # Returns valid name and invalid name to cover 322->320 and 384->382
                return ["0_test.md", "invalid.md"]
            return []

        with local_patch("os.path.exists", side_effect=exists_side_effect), \
             local_patch("os.listdir", side_effect=listdir_side_effect), \
             patch("time.monotonic", side_effect=[100.0 + i * 0.25 for i in range(100)]), \
             patch("pipeline_manager.make_zip", side_effect=Exception("zip crash")): # 396-397: make_zip exception
            
            gen3 = pipeline_manager.process_pdfs(
                files=[mock_file],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=2,
                max_concurrent=10,
                max_retries=3,
                target_dim=1024,
                guided_decoding=True
            )
            res3 = list(gen3)
            self.assertTrue(len(res3) > 0)
            
        process_state.active_runs.clear()

        # 4. Test generic generator exception (432-434)
        def exists_crash_side_effect(path):
            if "markdown" in path:
                raise Exception("Fatal exists check crash")
            return real_exists(path)

        mock_proc_inst3 = MagicMock()
        mock_proc_inst3.returncode = 0
        mock_proc_inst3.stdout.readline.side_effect = ["", ""]
        mock_popen.return_value = mock_proc_inst3

        with local_patch("os.path.exists", side_effect=exists_crash_side_effect):
            gen4 = pipeline_manager.process_pdfs(
                files=[mock_file],
                server_url="http://localhost:8000/v1",
                model_name="test-model",
                workers=2,
                max_concurrent=10,
                max_retries=3,
                target_dim=1024,
                guided_decoding=True
            )
            res4 = list(gen4)
            self.assertTrue(any("Exception during processing" in str(chunk[0]) for chunk in res4))

    def test_cleanup_active_runs_exception(self):
        # Line 476-477: verify exception caught during terminate/wait
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = Exception("Terminate failed")
        
        process_state.active_runs["run_exc"] = {
            "proc": mock_proc,
            "stop": False
        }
        with patch.dict(os.environ, {"TESTING": "false"}):
            pipeline_manager.cleanup_active_runs()
        mock_proc.terminate.assert_called_once()
        process_state.active_runs.clear()

    def test_stop_processing_no_proc(self):
        # 460->462 branch (stop_processing when proc is None)
        process_state.active_runs["run_no_proc"] = {
            "proc": None,
            "stop": False
        }
        res = pipeline_manager.stop_processing("run_no_proc")
        self.assertIn("Stop request sent", res)
        self.assertTrue(process_state.active_runs["run_no_proc"]["stop"])
        process_state.active_runs.clear()

        # stop_processing when run_id is empty
        res2 = pipeline_manager.stop_processing("")
        self.assertIn("No active process to stop", res2)

    def test_cleanup_active_runs_branches(self):
        # 467: TESTING = "true" branch
        with patch.dict(os.environ, {"TESTING": "true"}):
            pipeline_manager.cleanup_active_runs() # should return immediately

        # 471->469: proc is None or proc.poll() is not None (does not terminate)
        mock_proc_finished = MagicMock()
        mock_proc_finished.poll.return_value = 0 # process finished

        process_state.active_runs["run_finished"] = {
            "proc": mock_proc_finished,
            "stop": False
        }
        process_state.active_runs["run_none_proc"] = {
            "proc": None,
            "stop": False
        }
        with patch.dict(os.environ, {"TESTING": "false"}):
            pipeline_manager.cleanup_active_runs()
        
        mock_proc_finished.terminate.assert_not_called()
        process_state.active_runs.clear()

    @patch("httpx.get")
    def test_process_pdfs_preflight_model_mismatch(self, mock_get):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "data": [
                {"id": "nvidia/Phi-4-reasoning-plus-NVFP4"}
            ]
        }
        mock_get.return_value = mock_response
        
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="allenai/olmOCR-2-7B-1025-FP8",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertEqual(len(res), 1)
        self.assertTrue("Model Mismatch" in res[0][1]["value"])
        self.assertTrue("The requested model 'allenai/olmOCR-2-7B-1025-FP8' is not loaded" in res[0][0])

    def test_pipeline_result_properties(self):
        res = pipeline_manager.PipelineResult(
            "log", "badge", "progress", "comp", "fail", "selector", "zip", "indiv", "start", "run_id", "status_table", "manifest"
        )
        self.assertEqual(res.log_text, "log")
        self.assertEqual(res.status_badge, "badge")
        self.assertEqual(res.progress_bar, "progress")
        self.assertEqual(res.completed_pages, "comp")
        self.assertEqual(res.failed_pages, "fail")
        self.assertEqual(res.file_selector, "selector")
        self.assertEqual(res.download_zip, "zip")
        self.assertEqual(res.download_individual, "indiv")
        self.assertEqual(res.start_btn, "start")
        self.assertEqual(res.active_run_id, "run_id")
        self.assertEqual(res.file_status_table, "status_table")
        self.assertEqual(res.upload_manifest_display, "manifest")

    @patch("subprocess.Popen")
    @patch("shutil.copy")
    @patch("httpx.get")
    def test_process_pdfs_preflight_invalid_models_format(self, mock_get, mock_copy, mock_popen):
        mock_popen.side_effect = Exception("abort pipeline execution")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "data": "not a list"
        }
        mock_get.return_value = mock_response
        
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="allenai/olmOCR-2-7B-1025-FP8",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertTrue(len(res) > 0)
        self.assertFalse(any("is not loaded on the server" in item[0] for item in res))

    @patch("subprocess.Popen")
    @patch("shutil.copy")
    @patch("httpx.get")
    def test_process_pdfs_preflight_json_decode_error(self, mock_get, mock_copy, mock_popen):
        mock_popen.side_effect = Exception("abort pipeline execution")
        mock_response = MagicMock(status_code=200)
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        
        gen = pipeline_manager.process_pdfs(
            files=[mock_file],
            server_url="http://localhost:8000/v1",
            model_name="allenai/olmOCR-2-7B-1025-FP8",
            workers=4,
            max_concurrent=20,
            max_retries=8,
            target_dim=1288,
            guided_decoding=True
        )
        res = list(gen)
        self.assertTrue(len(res) > 0)
        self.assertFalse(any("is not loaded on the server" in item[0] for item in res))


if __name__ == "__main__":
    unittest.main()
