"""
Unit tests for cleanup_manager.py.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

import cleanup_manager
import process_state


class TestCleanupManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        process_state.active_runs.clear()

    def tearDown(self):
        process_state.active_runs.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_dir_size_nonexistent(self):
        self.assertEqual(cleanup_manager.get_dir_size("/mock/nonexistent/path"), 0)

    def test_get_dir_size_file(self):
        with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=123):
            self.assertEqual(cleanup_manager.get_dir_size("/mock/file"), 123)

    def test_get_dir_size_oserror(self):
        # Trigger OSError inside os.walk loop
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os.walk", return_value=[("/mock/dir", [], ["file.txt"])]), \
             patch("os.path.islink", return_value=False), \
             patch("os.path.getsize", side_effect=OSError("Access denied")):
            
            size = cleanup_manager.get_dir_size("/mock/dir")
            self.assertEqual(size, 0) # Ignored and returns 0

    def test_format_size_bytes(self):
        self.assertEqual(cleanup_manager.format_size(500), "500 B")

    def test_perform_reset_cleanup_none_selected(self):
        res = cleanup_manager.perform_reset_cleanup(False, False, False, False)
        self.assertEqual(res, "### No files selected or found to clean up.")

    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_rmtree_exception(self, mock_rmtree):
        workspace = os.path.join(self.temp_dir, "workspace")
        os.makedirs(os.path.join(workspace, "run_123"))
        mock_rmtree.side_effect = Exception("Rmtree failed")

        res = cleanup_manager.perform_reset_cleanup(
            clean_runs=True,
            clean_gradio=False,
            clean_pycache=False,
            clean_hf=False,
            workspace_dir=workspace,
        )
        self.assertTrue("Failed to delete run directory" in res)

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_active_runs(self, mock_rmtree, mock_listdir, mock_isdir, mock_exists):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ["run_active", "run_incomplete"]
        
        from settings_manager import WORKSPACE_DIR
        workspace_dir = WORKSPACE_DIR

        # Configure process_state.active_runs
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None # Process running
        process_state.active_runs["run_active"] = {
            "run_dir": os.path.join(workspace_dir, "run_active"),
            "proc": mock_proc,
            "completed": False
        }
        process_state.active_runs["run_incomplete"] = {
            "run_dir": os.path.join(workspace_dir, "run_incomplete"),
            "proc": None,
            "completed": False
        }

        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=False, clean_pycache=False, clean_hf=False, workspace_dir=workspace_dir)
        # Verify rmtree was not called since they are active/incomplete
        mock_rmtree.assert_not_called()
        self.assertEqual(res, "### No files selected or found to clean up.")

        process_state.active_runs.clear()

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_gradio_error(self, mock_rmtree, mock_listdir, mock_exists):
        # Configure path exists for gradio
        def exists_side_effect(path):
            return path == "/tmp/gradio"
        mock_exists.side_effect = exists_side_effect
        mock_listdir.return_value = ["upload_dir"]
        mock_rmtree.side_effect = Exception("Delete failed")

        res = cleanup_manager.perform_reset_cleanup(clean_runs=False, clean_gradio=True, clean_pycache=False, clean_hf=False)
        self.assertTrue("Failed to clean Gradio temp files" in res)

    @patch("os.walk")
    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_pycache_error(self, mock_rmtree, mock_walk):
        # An os.walk result outside the repository is ignored by the boundary.
        mock_walk.return_value = [
            ("/tmp/repo", ["__pycache__"], []),
            ("/tmp/repo/__pycache__", [], [])
        ]
        mock_rmtree.side_effect = Exception("Pycache delete failed")

        res = cleanup_manager.perform_reset_cleanup(clean_runs=False, clean_gradio=False, clean_pycache=True, clean_hf=False)
        self.assertEqual(res, "### No files selected or found to clean up.")
        mock_rmtree.assert_not_called()

    def test_perform_reset_cleanup_deletes_project_pycache_only(self):
        repo = Path(self.temp_dir) / "repo"
        project_cache = repo / "package" / "__pycache__"
        dependency_cache = repo / ".venv" / "package" / "__pycache__"
        project_cache.mkdir(parents=True)
        dependency_cache.mkdir(parents=True)
        (project_cache / "module.pyc").write_bytes(b"project bytecode")
        (dependency_cache / "module.pyc").write_bytes(b"dependency bytecode")

        res = cleanup_manager.perform_reset_cleanup(
            clean_runs=False,
            clean_gradio=False,
            clean_pycache=True,
            clean_hf=False,
            repo_dir=repo,
        )

        self.assertFalse(project_cache.exists())
        self.assertTrue(dependency_cache.exists())
        self.assertIn("Bytecode cache: `package/__pycache__`", res)
        self.assertNotIn("Warnings / Errors", res)

    def test_perform_reset_cleanup_reports_pycache_path_and_error(self):
        repo = Path(self.temp_dir) / "repo"
        pycache = repo / "package" / "__pycache__"
        pycache.mkdir(parents=True)

        with patch("shutil.rmtree", side_effect=PermissionError("permission denied")):
            res = cleanup_manager.perform_reset_cleanup(
                clean_runs=False,
                clean_gradio=False,
                clean_pycache=True,
                clean_hf=False,
                repo_dir=repo,
            )

        self.assertIn("`package/__pycache__`: permission denied", res)

    def test_perform_reset_cleanup_hf_error(self):
        home = Path(self.temp_dir) / "home"
        hf_cache = home / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True)
        (hf_cache / "file1.bin").write_bytes(b"model")

        with (
            patch("pathlib.Path.home", return_value=home),
            patch("pathlib.Path.unlink", side_effect=OSError("HF delete failed")),
        ):
            res = cleanup_manager.perform_reset_cleanup(
                clean_runs=False,
                clean_gradio=False,
                clean_pycache=False,
                clean_hf=True,
            )
        self.assertTrue("Failed to clean Hugging Face cache" in res)

    def test_perform_reset_cleanup_missing_dirs(self):
        missing_workspace = os.path.join(self.temp_dir, "missing")
        missing_home = Path(self.temp_dir) / "missing-home"
        with patch("pathlib.Path.home", return_value=missing_home):
            res = cleanup_manager.perform_reset_cleanup(
                clean_runs=True,
                clean_gradio=False,
                clean_pycache=False,
                clean_hf=True,
                workspace_dir=missing_workspace,
            )
        self.assertEqual(res, "### No files selected or found to clean up.")

    def test_perform_reset_cleanup_completed_runs(self):
        workspace_dir = os.path.join(self.temp_dir, "workspace")
        completed_dir = os.path.join(workspace_dir, "run_completed")
        os.makedirs(completed_dir)
        process_state.active_runs["run_completed"] = {
            "run_dir": completed_dir,
            "proc": None,
            "completed": True
        }
        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=False, clean_pycache=False, clean_hf=False, workspace_dir=workspace_dir)
        self.assertTrue("Obsolete run directory: `run_completed`" in res)
        self.assertFalse(os.path.exists(completed_dir))


if __name__ == "__main__":
    unittest.main()
