"""
Comprehensive unit tests for rag/storage.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from minio import Minio

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag.storage as rag_storage


class TestRAGStorageAll(unittest.TestCase):

    @patch("rag.storage.Minio")
    def test_get_client_defaults(self, mock_minio):
        client = rag_storage.get_client()
        self.assertIsNotNone(client)
        mock_minio.assert_called_once()

    @patch("rag.storage.Minio")
    def test_get_client_with_config(self, mock_minio):
        config = {
            "endpoint": "custom-s3:9000",
            "access_key": "minio_key",
            "secret_key": "minio_secret",
            "secure": False
        }
        client = rag_storage.get_client(config=config)
        self.assertIsNotNone(client)
        mock_minio.assert_called_once_with(
            "custom-s3:9000",
            access_key="minio_key",
            secret_key="minio_secret",
            secure=False
        )

    @patch("rag.storage.get_client")
    def test_init_buckets_creation(self, mock_get_client):
        mock_client = mock_get_client.return_value
        # Force buckets to not exist
        mock_client.bucket_exists.return_value = False

        rag_storage.init_buckets()
        self.assertEqual(mock_client.make_bucket.call_count, 2)

    @patch("rag.storage.get_client")
    def test_is_healthy_exception(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.list_buckets.side_effect = Exception("S3 down")

        self.assertFalse(rag_storage.is_healthy())

    @patch("rag.storage.get_client")
    @patch("rag.storage.init_buckets")
    def test_upload_markdown_file(self, mock_init, mock_get_client):
        mock_client = mock_get_client.return_value
        
        key = rag_storage.upload_markdown("run1", "doc1", "/tmp/report.md")
        self.assertEqual(key, "run1/doc1/report.md")
        mock_client.fput_object.assert_called_once()

    @patch("rag.storage.get_client")
    def test_download_file(self, mock_get_client):
        mock_client = mock_get_client.return_value
        
        rag_storage.download_file("bucket", "key", "/tmp/dest")
        mock_client.fget_object.assert_called_once_with("bucket", "key", "/tmp/dest")

    @patch("rag.storage.get_client")
    def test_delete_run_objects_no_bucket(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.bucket_exists.return_value = False

        rag_storage.delete_run_objects("run1")
        mock_client.list_objects.assert_not_called()

    @patch("rag.storage.get_client")
    def test_get_storage_stats_no_bucket(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.bucket_exists.return_value = False

        stats = rag_storage.get_storage_stats()
        self.assertEqual(stats[rag_storage.BUCKET_PDFS]["objects"], 0)
        self.assertEqual(stats[rag_storage.BUCKET_MARKDOWN]["objects"], 0)


if __name__ == "__main__":
    unittest.main()
