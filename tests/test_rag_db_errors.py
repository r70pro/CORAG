"""
Unit tests targeting database error and exception handlers in rag/db.py to ensure 100% code coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import psycopg2

# Prevent system operations during import
os.environ["TESTING"] = "true"

from rag import db as rag_db


class TestRAGDBErrors(unittest.TestCase):

    @patch("psycopg2.connect")
    def test_db_is_healthy_success(self, mock_connect):
        self.assertTrue(rag_db.is_healthy())

    @patch("psycopg2.connect")
    def test_db_is_healthy_exception(self, mock_connect):
        # Configure connection to raise OperationalError
        mock_connect.side_effect = psycopg2.OperationalError("Database is down")
        self.assertFalse(rag_db.is_healthy())

    @patch("psycopg2.connect")
    def test_db_get_connection_exception(self, mock_connect):
        # Test exception raised *inside* the connection context manager block
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        with self.assertRaises(ValueError):
            with rag_db.get_connection():
                raise ValueError("Force rollback")
        mock_conn.rollback.assert_called_once()

    @patch("psycopg2.connect")
    def test_db_get_connection_with_config(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        config = {"host": "custom-host", "port": 5432}
        with rag_db.get_connection(config=config):
            pass
        mock_connect.assert_called_once_with(**config)

    @patch("rag.db.get_connection")
    def test_register_run_exception(self, mock_get_conn):
        mock_conn = mock_get_conn.return_value.__enter__.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.execute.side_effect = psycopg2.DatabaseError("Insert failed")

        with self.assertRaises(psycopg2.DatabaseError):
            rag_db.register_run("run_err", "/path")

    @patch("rag.db.get_connection")
    @patch("psycopg2.extras.execute_values")
    def test_insert_chunks_exception(self, mock_execute_values, mock_get_conn):
        mock_execute_values.side_effect = psycopg2.DatabaseError("Insert failed")

        mock_chunk = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "run_id": "r1",
            "chunk_index": 0,
            "text": "text",
            "char_start": 0,
            "char_end": 4
        }
        with self.assertRaises(psycopg2.DatabaseError):
            rag_db.insert_chunks([mock_chunk])

    def test_insert_chunks_empty(self):
        # Line 265 early return
        self.assertIsNone(rag_db.insert_chunks([]))

    @patch("rag.db.get_connection")
    def test_get_corpus_stats_exception(self, mock_get_conn):
        mock_conn = mock_get_conn.return_value.__enter__.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.execute.side_effect = psycopg2.DatabaseError("Query failed")

        with self.assertRaises(psycopg2.DatabaseError):
            rag_db.get_corpus_stats()

    @patch("rag.db.get_connection")
    def test_delete_run_data_exception(self, mock_get_conn):
        mock_conn = mock_get_conn.return_value.__enter__.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.execute.side_effect = psycopg2.DatabaseError("Delete failed")

        with self.assertRaises(psycopg2.DatabaseError):
            rag_db.delete_run_data("run_err")

    def test_get_chunks_by_qdrant_ids_empty(self):
        # Line 331 early return
        self.assertEqual(rag_db.get_chunks_by_qdrant_ids([]), [])

    @patch("rag.db.get_connection")
    def test_get_chunk_by_qdrant_id(self, mock_get_conn):
        mock_conn = mock_get_conn.return_value.__enter__.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.fetchone.return_value = {"chunk_id": "chunk1"}

        res = rag_db.get_chunk_by_qdrant_id("point123")
        self.assertEqual(res["chunk_id"], "chunk1")

    @patch("rag.db.get_connection")
    @patch("rag.embedding.get_qdrant_client")
    @patch("rag.embedding.get_collection_name")
    def test_is_run_indexed_qdrant_check(self, mock_get_col_name, mock_get_qdrant, mock_get_conn):
        mock_conn = mock_get_conn.return_value.__enter__.return_value
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        mock_cur.fetchone.return_value = ("indexed",)

        mock_get_col_name.return_value = "my_collection"
        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client
        
        # Scenario 1: Collection doesn't exist
        mock_c = MagicMock()
        mock_c.name = "other_collection"
        mock_client.get_collections.return_value.collections = [mock_c]
        res = rag_db.is_run_indexed("r1", check_vector_store=True)
        self.assertFalse(res)

        # Scenario 2: Collection exists, count is 0
        mock_c.name = "my_collection"
        mock_client.get_collections.return_value.collections = [mock_c]
        mock_count_res = MagicMock()
        mock_count_res.count = 0
        mock_client.count.return_value = mock_count_res
        res = rag_db.is_run_indexed("r1", check_vector_store=True)
        self.assertFalse(res)

        # Scenario 3: Collection exists, count > 0
        mock_count_res.count = 5
        res = rag_db.is_run_indexed("r1", check_vector_store=True)
        self.assertTrue(res)

        # Scenario 4: Exception raised
        mock_client.get_collections.side_effect = Exception("Qdrant error")
        res = rag_db.is_run_indexed("r1", check_vector_store=True)
        self.assertTrue(res)

    @patch("rag.db.ThreadedConnectionPool")
    def test_connection_pooling(self, mock_pool_cls):
        import sys
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        # Reset global state to ensure pool is recreated
        rag_db._connection_pool = None

        with patch.dict(os.environ, {"TESTING": "false"}):
            with patch.dict("sys.modules"):
                if "pytest" in sys.modules:
                    del sys.modules["pytest"]
                
                # Trigger pooling path
                with rag_db.get_connection():
                    pass
                
                mock_pool.getconn.assert_called_once()
                mock_pool.putconn.assert_called_once_with(mock_conn)

        # Reset global state back to None after testing
        rag_db._connection_pool = None

    @patch("rag.db.ThreadedConnectionPool")
    def test_connection_pooling_rollback_on_exception(self, mock_pool_cls):
        import sys
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        rag_db._connection_pool = None

        with patch.dict(os.environ, {"TESTING": "false"}):
            with patch.dict("sys.modules"):
                if "pytest" in sys.modules:
                    del sys.modules["pytest"]
                
                with self.assertRaises(ValueError):
                    with rag_db.get_connection():
                        raise ValueError("Rollback in pool")
                
                mock_pool.getconn.assert_called_once()
                mock_conn.rollback.assert_called_once()
                mock_pool.putconn.assert_called_once_with(mock_conn)

        rag_db._connection_pool = None


if __name__ == "__main__":
    unittest.main()
