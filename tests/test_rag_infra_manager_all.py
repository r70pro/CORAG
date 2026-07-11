"""
Comprehensive unit tests for rag_infra_manager.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import subprocess

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag_infra_manager as rim


class TestRAGInfraManagerAll(unittest.TestCase):

    @patch("subprocess.run")
    def test_run_compose_success(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "OK stdout"
        mock_run.return_value = mock_res
        
        success, msg = rim._run_compose(["args"])
        self.assertTrue(success)
        self.assertEqual(msg, "OK stdout")

    @patch("subprocess.run")
    def test_run_compose_non_zero(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = "Error stderr"
        mock_run.return_value = mock_res
        
        success, msg = rim._run_compose(["args"])
        self.assertFalse(success)
        self.assertEqual(msg, "Error stderr")

    @patch("subprocess.run")
    def test_run_compose_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="compose", timeout=10)
        success, msg = rim._run_compose(["args"])
        self.assertFalse(success)
        self.assertIn("timed out", msg)

    @patch("subprocess.run")
    def test_run_compose_exception(self, mock_run):
        mock_run.side_effect = Exception("Subprocess spawn failed")
        success, msg = rim._run_compose(["args"])
        self.assertFalse(success)
        self.assertIn("Subprocess spawn failed", msg)

    @patch("rag_infra_manager._run_compose")
    def test_start_stop_destroy(self, mock_compose):
        # 1. Start success/fail
        mock_compose.return_value = (True, "OK")
        self.assertEqual(rim.start_rag_infrastructure(), (True, "RAG infrastructure started successfully."))
        mock_compose.return_value = (False, "fail")
        self.assertEqual(rim.start_rag_infrastructure(), (False, "Failed to start RAG infrastructure: fail"))

        # 2. Stop success/fail
        mock_compose.return_value = (True, "OK")
        self.assertEqual(rim.stop_rag_infrastructure(), (True, "RAG infrastructure stopped."))
        mock_compose.return_value = (False, "fail")
        self.assertEqual(rim.stop_rag_infrastructure(), (False, "Failed to stop: fail"))

        # 3. Destroy success/fail
        mock_compose.return_value = (True, "OK")
        self.assertEqual(rim.destroy_rag_infrastructure(remove_volumes=True), (True, "RAG infrastructure destroyed."))
        mock_compose.return_value = (False, "fail")
        self.assertEqual(rim.destroy_rag_infrastructure(remove_volumes=False), (False, "Failed to destroy: fail"))

    @patch("subprocess.run")
    def test_get_rag_service_status_parsing(self, mock_run):
        # 1. Success with complex lines
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = (
            '{"Service": "postgres", "State": "exited"}\n' # exited state to trigger line 132
            '{"Service": "redis", "State": "running"}\n'
            '   \n' # whitespace line in the middle to trigger line 115
            '{"Service": "minio", "State": "running", "Health": "unhealthy"}\n'
            '{"Service": "qdrant", "State": "paused"}\n' # paused state to trigger line 134
            '{"Service": "unrecognized-service", "State": "running"}\n' # unrecognized service name to cover branch 122->112
            '{invalid json}\n'
        )
        mock_run.return_value = mock_res
        
        statuses = rim.get_rag_service_status()
        self.assertEqual(statuses["postgres"], "stopped")
        self.assertEqual(statuses["redis"], "running")
        self.assertEqual(statuses["minio"], "unhealthy")
        self.assertEqual(statuses["qdrant"], "paused")

        # 2. Non-zero return code / Exception path
        mock_res.returncode = 1
        statuses2 = rim.get_rag_service_status()
        self.assertEqual(statuses2["postgres"], "unknown")

        # 3. Subprocess run raises Exception to trigger line 138-139
        mock_run.side_effect = OSError("daemon down")
        statuses3 = rim.get_rag_service_status()
        self.assertEqual(statuses3["postgres"], "unknown")
        mock_run.side_effect = None

    def test_get_rag_status_html(self):
        with patch("rag_infra_manager.get_rag_service_status") as mock_status:
            mock_status.return_value = {
                "postgres": "healthy",
                "redis": "running",
                "minio": "unhealthy",
                "qdrant": "unknown"
            }
            html = rim.get_rag_status_html()
            self.assertIn("badge-success", html)
            self.assertIn("badge-running", html)
            self.assertIn("badge-failed", html)
            self.assertIn("badge-idle", html)

    def test_is_rag_infrastructure_ready(self):
        with patch("rag_infra_manager.get_rag_service_status") as mock_status:
            mock_status.return_value = {"postgres": "healthy", "redis": "running"}
            self.assertTrue(rim.is_rag_infrastructure_ready())
            
            mock_status.return_value = {"postgres": "healthy", "redis": "stopped"}
            self.assertFalse(rim.is_rag_infrastructure_ready())

    @patch("time.sleep")
    @patch("rag.db.is_healthy")
    @patch("rag.db.init_schema")
    def test_init_rag_database(self, mock_init, mock_healthy, mock_sleep):
        # 1. Success path
        mock_healthy.side_effect = [False, True]
        success, msg = rim.init_rag_database()
        self.assertTrue(success)
        mock_init.assert_called_once()

        # 2. Timeout path
        mock_healthy.side_effect = [False] * 12
        success2, msg2 = rim.init_rag_database()
        self.assertFalse(success2)
        self.assertIn("PostgreSQL not ready", msg2)

        # 3. Exception path
        mock_healthy.side_effect = Exception("DB driver error")
        success3, msg3 = rim.init_rag_database()
        self.assertFalse(success3)
        self.assertIn("Failed to initialize database", msg3)

    @patch("time.sleep")
    @patch("rag.storage.is_healthy")
    @patch("rag.storage.init_buckets")
    def test_init_rag_storage(self, mock_init, mock_healthy, mock_sleep):
        # 1. Success path
        mock_healthy.side_effect = [False, True]
        success, msg = rim.init_rag_storage()
        self.assertTrue(success)
        mock_init.assert_called_once()

        # 2. Timeout path
        mock_healthy.side_effect = [False] * 12
        success2, msg2 = rim.init_rag_storage()
        self.assertFalse(success2)
        self.assertIn("MinIO not ready", msg2)

        # 3. Exception path
        mock_healthy.side_effect = Exception("Minio driver error")
        success3, msg3 = rim.init_rag_storage()
        self.assertFalse(success3)
        self.assertIn("Failed to initialize storage", msg3)

    @patch("time.sleep")
    @patch("rag.embedding.is_healthy")
    @patch("rag.embedding.init_collection")
    def test_init_rag_vector_store(self, mock_init, mock_healthy, mock_sleep):
        # 1. Success path
        mock_healthy.side_effect = [False, True]
        success, msg = rim.init_rag_vector_store()
        self.assertTrue(success)
        mock_init.assert_called_once()

        # 2. Timeout path
        mock_healthy.side_effect = [False] * 12
        success2, msg2 = rim.init_rag_vector_store()
        self.assertFalse(success2)
        self.assertIn("Qdrant not ready", msg2)

        # 3. Exception path
        mock_healthy.side_effect = Exception("Qdrant driver error")
        success3, msg3 = rim.init_rag_vector_store()
        self.assertFalse(success3)
        self.assertIn("Failed to initialize vector store", msg3)

    @patch("time.sleep")
    @patch("rag_infra_manager.start_rag_infrastructure")
    @patch("rag_infra_manager.init_rag_database")
    @patch("rag_infra_manager.init_rag_storage")
    @patch("rag_infra_manager.init_rag_vector_store")
    def test_start_and_init_rag(self, mock_init_vect, mock_init_stor, mock_init_db, mock_start, mock_sleep):
        # 1. All success
        mock_start.return_value = (True, "Started")
        mock_init_db.return_value = (True, "DB OK")
        mock_init_stor.return_value = (True, "Stor OK")
        mock_init_vect.return_value = (True, "Vect OK")
        
        success, msg = rim.start_and_init_rag()
        self.assertTrue(success)
        self.assertIn("DB OK", msg)

        # 2. Start infra fails early exit
        mock_start.return_value = (False, "compose down")
        success2, msg2 = rim.start_and_init_rag()
        self.assertFalse(success2)
        self.assertIn("compose down", msg2)


if __name__ == "__main__":
    unittest.main()
