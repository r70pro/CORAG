"""
Unit tests for cleanup_manager.py.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

import cleanup_manager
import state


class TestCleanupManager(unittest.TestCase):

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

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_rmtree_exception(self, mock_rmtree, mock_listdir, mock_isdir, mock_exists):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ["run_123"]
        mock_rmtree.side_effect = Exception("Rmtree failed")
        
        # Test runs clean up error
        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=False, clean_pycache=False, clean_hf=False)
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
        workspace_dir = state.get_val('WORKSPACE_DIR', WORKSPACE_DIR)

        # Configure state.active_runs
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None # Process running
        state.active_runs["run_active"] = {
            "run_dir": os.path.join(workspace_dir, "run_active"),
            "proc": mock_proc,
            "completed": False
        }
        state.active_runs["run_incomplete"] = {
            "run_dir": os.path.join(workspace_dir, "run_incomplete"),
            "proc": None,
            "completed": False
        }

        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=False, clean_pycache=False, clean_hf=False)
        # Verify rmtree was not called since they are active/incomplete
        mock_rmtree.assert_not_called()
        self.assertEqual(res, "### No files selected or found to clean up.")

        state.active_runs.clear()

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
        # Configure walk to yield a __pycache__ directory
        mock_walk.return_value = [
            ("/tmp/repo", ["__pycache__"], []),
            ("/tmp/repo/__pycache__", [], [])
        ]
        mock_rmtree.side_effect = Exception("Pycache delete failed")

        res = cleanup_manager.perform_reset_cleanup(clean_runs=False, clean_gradio=False, clean_pycache=True, clean_hf=False)
        self.assertTrue("Failed to delete" in res)

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.remove")
    def test_perform_reset_cleanup_hf_error(self, mock_remove, mock_listdir, mock_exists):
        # Configure exists for HF cache
        def exists_side_effect(path):
            return "huggingface" in path
        mock_exists.side_effect = exists_side_effect
        mock_listdir.return_value = ["file1.bin"]
        mock_remove.side_effect = Exception("HF delete failed")

        res = cleanup_manager.perform_reset_cleanup(clean_runs=False, clean_gradio=False, clean_pycache=False, clean_hf=True)
        self.assertTrue("Failed to clean Hugging Face cache" in res)

    @patch("os.path.exists", return_value=False)
    def test_perform_reset_cleanup_missing_dirs(self, mock_exists):
        # Trigger missing workspace_dir, gradio_temp_dir, and hf_cache_dir (42->68, 70->85, 103->118 branches)
        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=True, clean_pycache=False, clean_hf=True)
        self.assertEqual(res, "### No files selected or found to clean up.")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.listdir")
    @patch("shutil.rmtree")
    def test_perform_reset_cleanup_completed_runs(self, mock_rmtree, mock_listdir, mock_isdir, mock_exists):
        # Normal runs cleanup where run is completed = True, so loop continues (line 55->49 branch)
        mock_listdir.return_value = ["run_completed"]
        from settings_manager import WORKSPACE_DIR
        workspace_dir = state.get_val('WORKSPACE_DIR', WORKSPACE_DIR)
        state.active_runs["run_completed"] = {
            "run_dir": os.path.join(workspace_dir, "run_completed"),
            "proc": None,
            "completed": True
        }
        res = cleanup_manager.perform_reset_cleanup(clean_runs=True, clean_gradio=False, clean_pycache=False, clean_hf=False)
        self.assertTrue("Obsolete run directory: `run_completed`" in res)
        mock_rmtree.assert_called_once()
        state.active_runs.clear()


if __name__ == "__main__":
    unittest.main()
