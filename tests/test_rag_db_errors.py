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


if __name__ == "__main__":
    unittest.main()
