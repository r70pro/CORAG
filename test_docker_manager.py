"""
Unit tests for docker_manager.py.
"""

import os
import subprocess
import unittest
from unittest.mock import patch, MagicMock
import httpx

import sys

# Prevent system operations during import
os.environ["TESTING"] = "true"

import docker_manager


class TestDockerManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Hide app module to let state.get_fn resolve to locally imported versions
        cls.saved_app = sys.modules.get('app')
        if 'app' in sys.modules:
            del sys.modules['app']

    @classmethod
    def tearDownClass(cls):
        if cls.saved_app:
            sys.modules['app'] = cls.saved_app

    @patch("subprocess.run")
    def test_get_docker_status_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="running\n")
        status = docker_manager.get_docker_status()
        self.assertEqual(status, "running")

    @patch("subprocess.run")
    def test_get_docker_status_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="no such inspect object")
        status = docker_manager.get_docker_status()
        self.assertEqual(status, "not_found")

    @patch("subprocess.run")
    def test_get_docker_status_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="permission denied")
        status = docker_manager.get_docker_status()
        self.assertEqual(status, "error")

    @patch("subprocess.run")
    def test_get_docker_status_exception(self, mock_run):
        mock_run.side_effect = Exception("Docker dead")
        status = docker_manager.get_docker_status()
        self.assertEqual(status, "error")

    @patch("httpx.get")
    def test_check_server_ready_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(docker_manager.check_server_ready(8000))

    @patch("httpx.get")
    def test_check_server_ready_failed(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        self.assertFalse(docker_manager.check_server_ready(8000))

    @patch("httpx.get")
    def test_check_server_ready_exception(self, mock_get):
        mock_get.side_effect = Exception("refused")
        self.assertFalse(docker_manager.check_server_ready(8000))

    @patch("docker_manager.get_docker_status")
    @patch("docker_manager.check_server_ready")
    def test_get_docker_status_str(self, mock_ready, mock_status):
        # Case: not_found
        mock_status.return_value = "not_found"
        self.assertEqual(docker_manager.get_docker_status_str(8000)[0], "not_found")

        # Case: exited
        mock_status.return_value = "exited"
        self.assertEqual(docker_manager.get_docker_status_str(8000)[0], "stopped")

        # Case: running and ready
        mock_status.return_value = "running"
        mock_ready.return_value = True
        self.assertEqual(docker_manager.get_docker_status_str(8000)[0], "ready")

        # Case: running and starting
        mock_ready.return_value = False
        self.assertEqual(docker_manager.get_docker_status_str(8000)[0], "starting")

        # Case: error
        mock_status.return_value = "error"
        self.assertEqual(docker_manager.get_docker_status_str(8000)[0], "error")

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    def test_start_docker_container(self, mock_run, mock_status):
        # Case: running
        mock_status.return_value = "running"
        success, msg = docker_manager.start_docker_container()
        self.assertTrue(success)
        self.assertTrue("already running" in msg)

        # Case: not_found
        mock_status.return_value = "not_found"
        success, msg = docker_manager.start_docker_container()
        self.assertFalse(success)
        self.assertTrue("not found" in msg)

        # Case: exited (success)
        mock_status.return_value = "exited"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.start_docker_container()
        self.assertTrue(success)

        # Case: exited (failure)
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker start", stderr=b"daemon down")
        success, msg = docker_manager.start_docker_container()
        self.assertFalse(success)
        self.assertTrue("Failed to start container" in msg)

        # Case: error status fallback
        mock_run.side_effect = None
        mock_status.return_value = "error"
        success, msg = docker_manager.start_docker_container()
        self.assertFalse(success)

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    def test_stop_docker_container(self, mock_run, mock_status):
        # Case: not running
        mock_status.return_value = "exited"
        success, msg = docker_manager.stop_docker_container()
        self.assertTrue(success)

        # Case: running (success)
        mock_status.return_value = "running"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.stop_docker_container()
        self.assertTrue(success)

        # Case: running (failure)
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker stop", stderr=b"cannot kill")
        success, msg = docker_manager.stop_docker_container()
        self.assertFalse(success)

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    @patch("os.makedirs")
    def test_create_docker_container(self, mock_makedirs, mock_run, mock_status):
        # Case: running -> remove -> create success
        mock_status.return_value = "running"
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertTrue(success)

        # Case: remove failed
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker stop", stderr=b"permission error")
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertFalse(success)
        self.assertTrue("Failed to remove existing container" in msg)

        # Case: create command failed
        mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=0), subprocess.CalledProcessError(1, "docker run", stderr=b"bad flag")]
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertFalse(success)

        # Case: exited status (skips docker stop, goes straight to rm, covers 72->74 branch)
        mock_status.return_value = "exited"
        mock_run.side_effect = None
        mock_run.reset_mock()
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertTrue(success)
        self.assertEqual(mock_run.call_count, 2)

        # Case: not_found status (skips both stop and rm, covers 70->78 branch)
        mock_status.return_value = "not_found"
        mock_run.reset_mock()
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertTrue(success)
        self.assertEqual(mock_run.call_count, 1)

    @patch("subprocess.run")
    def test_cleanup_docker(self, mock_run):
        # Default testing skips
        docker_manager.cleanup_docker()
        mock_run.assert_not_called()

        # Toggle environment
        with patch.dict(os.environ, {"TESTING": "false"}):
            docker_manager.cleanup_docker()
            mock_run.assert_called_once()

        # Toggle environment exception
        mock_run.reset_mock()
        mock_run.side_effect = Exception("Docker shutdown failed")
        with patch.dict(os.environ, {"TESTING": "false"}):
            docker_manager.cleanup_docker()
            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
