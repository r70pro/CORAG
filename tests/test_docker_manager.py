"""
Unit tests for docker_manager.py.
"""

import os
import subprocess
import unittest
from unittest.mock import patch, MagicMock


# Prevent system operations during import
os.environ["TESTING"] = "true"

import docker_manager


class TestDockerManager(unittest.TestCase):



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

    def test_get_cached_models(self):
        models = docker_manager.get_cached_models()
        self.assertIsInstance(models, list)
        self.assertIn("allenai/olmOCR-2-7B-1025-FP8", models)
        self.assertIn("nvidia/Phi-4-reasoning-plus-NVFP4", models)


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
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        # Case: running
        mock_status.return_value = "running"
        success, msg = docker_manager.start_docker_container()
        self.assertTrue(success)
        self.assertTrue("already running" in msg)

        # Case: not_found (attempts provisioning)
        mock_status.return_value = "not_found"
        success, msg = docker_manager.start_docker_container()
        self.assertTrue(success)
        self.assertIn("Provisioned", msg)

        # Case: exited (success)
        mock_status.return_value = "exited"
        success, msg = docker_manager.start_docker_container()
        self.assertTrue(success)

        # Case: exited (failure on docker start)
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker start", stderr=b"daemon down")
        success, msg = docker_manager.start_docker_container()
        self.assertFalse(success)
        self.assertTrue("Failed to start container" in msg)

        # Case: error status fallback
        mock_run.side_effect = None
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_status.return_value = "error"
        success, msg = docker_manager.start_docker_container()
        self.assertFalse(success)

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    def test_stop_docker_container(self, mock_run, mock_status):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        # Case: not running
        mock_status.return_value = "exited"
        success, msg = docker_manager.stop_docker_container()
        self.assertTrue(success)

        # Case: running (success)
        mock_status.return_value = "running"
        success, msg = docker_manager.stop_docker_container()
        self.assertTrue(success)

        # Case: running (failure)
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker stop", stderr=b"cannot kill")
        success, msg = docker_manager.stop_docker_container()
        self.assertFalse(success)

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    def test_shutdown_docker_container(self, mock_run, mock_status):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        # Case: not running / not created
        mock_status.return_value = "not_found"
        success, msg = docker_manager.shutdown_docker_container()
        self.assertTrue(success)
        self.assertIn("not running", msg)

        # Case: exited (success)
        mock_status.return_value = "exited"
        success, msg = docker_manager.shutdown_docker_container()
        self.assertTrue(success)
        self.assertIn("shutdown successfully", msg)

        # Case: running (success)
        mock_status.return_value = "running"
        mock_run.reset_mock()
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.shutdown_docker_container()
        self.assertTrue(success)
        self.assertIn("shutdown successfully", msg)
        self.assertEqual(mock_run.call_count, 3)

        # Case: running (failure on stop)
        mock_status.return_value = "running"
        mock_run.reset_mock()
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker stop", stderr=b"failed to stop")
        success, msg = docker_manager.shutdown_docker_container()
        self.assertFalse(success)
        self.assertIn("failed to stop", msg.lower())

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
        mock_run.side_effect = [MagicMock(returncode=0), subprocess.CalledProcessError(1, "docker run", stderr=b"bad flag")]
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
        success, msg = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000, tensor_parallel_size=4)
        self.assertTrue(success)
        self.assertEqual(mock_run.call_count, 1)
        executed_cmd = mock_run.call_args[0][0]
        self.assertIn(docker_manager.resolve_vllm_image(), executed_cmd)
        self.assertIn("--gpu-memory-utilization", executed_cmd)
        self.assertIn("--max-model-len", executed_cmd)
        self.assertIn("--tensor-parallel-size", executed_cmd)
        tp_idx = executed_cmd.index("--tensor-parallel-size")
        self.assertEqual(executed_cmd[tp_idx + 1], "4")

    @patch("subprocess.run")
    def test_cleanup_docker(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        # Default testing skips
        docker_manager.cleanup_docker()
        mock_run.assert_not_called()

        # Toggle environment
        with patch.dict(os.environ, {"TESTING": "false"}):
            docker_manager.cleanup_docker()
            self.assertTrue(mock_run.called)

        # Toggle environment exception
        mock_run.reset_mock()
        mock_run.side_effect = Exception("Docker shutdown failed")
        with patch.dict(os.environ, {"TESTING": "false"}):
            docker_manager.cleanup_docker()
            self.assertTrue(mock_run.called)

    @patch("docker_manager.get_docker_status")
    @patch("subprocess.run")
    @patch("os.remove")
    def test_docker_manager_edge_cases(self, mock_remove, mock_run, mock_status):
        # 1. Non-numeric port in get_docker_status_str (line 36-37)
        mock_status.return_value = "not_found"
        res_str = docker_manager.get_docker_status_str("invalid_port")
        self.assertEqual(res_str[0], "not_found")

        # 2. Non-numeric port in create_docker_container (line 81-82)
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = docker_manager.create_docker_container("token", "invalid_port", "model", 0.8, 16000)
        self.assertTrue(success)

        # 3. Toggle KEEP_CONTAINERS_ON_EXIT env var (line 159)
        mock_run.reset_mock()
        with patch.dict(os.environ, {"TESTING": "false", "KEEP_CONTAINERS_ON_EXIT": "true"}):
            docker_manager.cleanup_docker()
            mock_run.assert_not_called()

        # 4. OSError on temp file removal inside create_docker_container (lines 131-132, 136-139)
        mock_remove.side_effect = OSError("mock access denied")
        success2, msg2 = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
        self.assertTrue(success2)

    @patch("subprocess.run")
    def test_get_docker_logs(self, mock_run):
        # Case: success with stdout and stderr
        mock_run.return_value = MagicMock(returncode=0, stdout="info log", stderr="warn log")
        logs = docker_manager.get_docker_logs(50)
        self.assertIn("info log", logs)
        self.assertIn("warn log", logs)

        # Case: container not found
        mock_run.return_value = MagicMock(returncode=1, stderr="no such container: olmocr")
        logs = docker_manager.get_docker_logs(50)
        self.assertIn("not found", logs.lower())

        # Case: command exception
        mock_run.side_effect = Exception("docker error")
        logs = docker_manager.get_docker_logs(50)
        self.assertIn("failed", logs.lower())

    @patch("subprocess.run")
    def test_resolve_vllm_image(self, mock_run):
        # 1. Custom image in env when TESTING is true
        with patch.dict(os.environ, {"OLMOCR_VLLM_IMAGE": "custom/image:tag", "TESTING": "true"}):
            self.assertEqual(docker_manager.resolve_vllm_image(), "custom/image:tag")

        # 2. Custom image in env when TESTING is false (inspect success)
        mock_run.return_value = MagicMock(returncode=0)
        with patch.dict(os.environ, {"OLMOCR_VLLM_IMAGE": "custom/image:tag", "TESTING": "false"}):
            self.assertEqual(docker_manager.resolve_vllm_image(), "custom/image:tag")

        # 3. Fallback when inspect fails
        mock_run.return_value = MagicMock(returncode=1)
        with patch.dict(os.environ, {}, clear=True):
            img = docker_manager.resolve_vllm_image()
            self.assertIn("vllm", img)

    @patch("subprocess.run")
    def test_free_host_port(self, mock_run):
        # When TESTING is true -> skips
        with patch.dict(os.environ, {"TESTING": "true"}):
            docker_manager.free_host_port(8000)
            mock_run.assert_not_called()

        # When TESTING is false -> inspects and removes bound containers
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="c123\tolmocr\t0.0.0.0:8000->8000/tcp\n"),
            MagicMock(returncode=0)
        ]
        with patch.dict(os.environ, {"TESTING": "false"}):
            docker_manager.free_host_port(8000)
            self.assertEqual(mock_run.call_count, 2)

    @patch("socket.socket")
    def test_wait_for_port_free(self, mock_sock):
        # When TESTING is true -> return True
        with patch.dict(os.environ, {"TESTING": "true"}):
            self.assertTrue(docker_manager.wait_for_port_free(8000))

        # When TESTING is false -> checks socket
        mock_s_inst = MagicMock()
        mock_sock.return_value.__enter__.return_value = mock_s_inst
        mock_s_inst.connect_ex.return_value = 1  # 1 means port is free
        with patch.dict(os.environ, {"TESTING": "false"}):
            self.assertTrue(docker_manager.wait_for_port_free(8000))

    @patch("subprocess.run")
    def test_create_docker_container_dotenv_token_and_port_conflict(self, mock_run):
        # 1. Test dotenv token fallback (lines 359-371)
        with patch("docker_manager.get_docker_status", return_value="not_found"):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", unittest.mock.mock_open(read_data="HF_TOKEN=dotenv_token_123\n")):
                    with patch.dict(os.environ, {}, clear=True):
                        mock_run.return_value = MagicMock(returncode=0)
                        ok, _ = docker_manager.create_docker_container("********", 8000, "model", 0.8, 16000)
                        self.assertTrue(ok)
                        self.assertEqual(os.environ.get("HF_TOKEN"), "dotenv_token_123")

        # 2. Test port conflict retry branch (lines 438-447)
        with patch("docker_manager.get_docker_status", return_value="not_found"):
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, "docker run", stderr=b"port is already allocated"),
                MagicMock(returncode=0), # rm -f
                MagicMock(returncode=0), # retry run
            ]
            ok2, msg2 = docker_manager.create_docker_container("token", 8000, "model", 0.8, 16000)
            self.assertTrue(ok2)
            self.assertIn("retry", msg2)

    @patch("rag_infra_manager.stop_rag_infrastructure")
    def test_shutdown_and_cleanup_rag_infra_exception(self, mock_stop_infra):
        mock_stop_infra.side_effect = Exception("Infra stop error")

        with patch("docker_manager.get_docker_status", return_value="not_found"):
            ok, msg = docker_manager.shutdown_docker_container()
            self.assertIn("RAG Infra shutdown error", msg)

        with patch.dict(os.environ, {"TESTING": "false"}):
            with patch("subprocess.run"):
                docker_manager.cleanup_docker()

    def test_create_docker_container_invalid_param_coercions(self):
        # Test non-numeric / negative parameters to hit lines 316-317, 323-325, 331-333
        with patch("docker_manager.get_docker_status", return_value="not_found"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                ok, msg = docker_manager.create_docker_container(
                    hf_token="token",
                    port="invalid",
                    model="model",
                    gpu_mem="invalid_gpu",
                    max_model_len="-500",
                    tensor_parallel_size="invalid_tp"
                )
                self.assertTrue(ok)

    @patch("subprocess.run")
    def test_resolve_vllm_image_exception_fallback(self, mock_run):
        # Hit lines 239-240 and 249-250 (exception during docker inspect)
        mock_run.side_effect = Exception("Docker inspect exception")
        with patch.dict(os.environ, {"OLMOCR_VLLM_IMAGE": "custom_img", "TESTING": "false"}):
            img = docker_manager.resolve_vllm_image()
            self.assertEqual(img, "vllm/vllm-openai:v0.20.0")

    def test_get_cached_models_info(self):
        # Hits lines 134-141
        models, max_lens = docker_manager.get_cached_models_info()
        self.assertIsInstance(models, list)
        self.assertIsInstance(max_lens, dict)

    @patch("rag_infra_manager.start_rag_infrastructure")
    def test_start_docker_container_infra_exception(self, mock_start_infra):
        # Hits lines 156-158
        mock_start_infra.side_effect = Exception("Infra start failed")
        with patch("docker_manager.get_docker_status", return_value="running"):
            ok, msg = docker_manager.start_docker_container()
            self.assertTrue(ok)
            self.assertIn("RAG Infra error", msg)

    @patch("docker_manager.get_docker_status")
    def test_get_docker_status_str_unknown(self, mock_status):
        # Hits line 78
        mock_status.return_value = "unknown_status"
        res = docker_manager.get_docker_status_str(8000)
        self.assertEqual(res[0], "error")

    @patch("subprocess.run")
    def test_free_host_port_exception(self, mock_run):
        # Hits lines 277-278
        mock_run.side_effect = Exception("Port check failed")
        with patch.dict(os.environ, {"TESTING": "false"}):
            # Should handle exception silently
            docker_manager.free_host_port(8000)


if __name__ == "__main__":
    unittest.main()




