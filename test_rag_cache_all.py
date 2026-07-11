"""
Comprehensive unit tests for rag/cache.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import redis

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag.cache as rag_cache


class TestRAGCacheAll(unittest.TestCase):

    def setUp(self):
        # Reset singleton client before each test
        rag_cache.reset_client()

    def tearDown(self):
        rag_cache.reset_client()

    @patch("redis.Redis")
    def test_cache_normal_operations(self, mock_redis_class):
        mock_redis = mock_redis_class.return_value
        
        # Configure mocked responses
        mock_redis.ping.return_value = True
        
        with patch("rag.cache.get_client", return_value=mock_redis):
            # is_healthy
            self.assertTrue(rag_cache.is_healthy())

            # get_cached_embedding missed return None
            mock_redis.get.return_value = None
            self.assertIsNone(rag_cache.get_cached_embedding("missing", "model"))

            # get_chat_history missed return empty
            mock_redis.get.return_value = None
            self.assertEqual(rag_cache.get_chat_history("missing"), [])

            # clear_chat_history
            rag_cache.clear_chat_history("session")
            mock_redis.delete.assert_called_once()

            # get_all_stats
            mock_redis.keys.return_value = [rag_cache.PREFIX_STATS + "queries"]
            mock_redis.get.return_value = "42"
            stats = rag_cache.get_all_stats()
            self.assertEqual(stats, {"queries": 42})

            # get_cache_info
            mock_redis.keys.side_effect = [["key1"], ["key2"], ["key3"]]
            mock_redis.info.return_value = {"used_memory_human": "5.5MB"}
            info = rag_cache.get_cache_info()
            self.assertEqual(info["cached_queries"], 1)
            self.assertEqual(info["memory_used"], "5.5MB")

    @patch("redis.Redis")
    def test_cache_fallback_on_exceptions(self, mock_redis_class):
        mock_redis = mock_redis_class.return_value
        
        # Configure Redis methods to raise ConnectionError
        mock_redis.ping.side_effect = redis.exceptions.ConnectionError("Redis down")
        mock_redis.get.side_effect = redis.exceptions.ConnectionError("Redis down")
        mock_redis.set.side_effect = redis.exceptions.ConnectionError("Redis down")
        mock_redis.delete.side_effect = redis.exceptions.ConnectionError("Redis down")
        mock_redis.incr.side_effect = redis.exceptions.ConnectionError("Redis down")
        mock_redis.info.side_effect = redis.exceptions.ConnectionError("Redis down")
        
        # Override singleton
        with patch("rag.cache.get_client", return_value=mock_redis):
            # is_healthy should be False
            self.assertFalse(rag_cache.is_healthy())
            
            # get_cached_query should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.get_cached_query("query")
            
            # cache_query_result should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.cache_query_result("query", {"val": 1})
            
            # get_cached_embedding should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.get_cached_embedding("text", "model")
            
            # cache_embedding should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.cache_embedding("text", [0.1], "model")
            
            # save_chat_history should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.save_chat_history("session", [])
            
            # get_chat_history should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.get_chat_history("session")
            
            # increment_stat should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.increment_stat("stat")
            
            # get_stat should raise
            with self.assertRaises(redis.exceptions.ConnectionError):
                rag_cache.get_stat("stat")

    def test_get_client_instantiation(self):
        # Test default instantiation branch
        with patch("redis.Redis"):
            client = rag_cache.get_client()
            self.assertIsNotNone(client)
            
            # Test custom config instantiation branch
            rag_cache.reset_client()
            client2 = rag_cache.get_client({"host": "custom", "port": 1234})
            self.assertIsNotNone(client2)


    def test_reset_client_exception(self):
        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("Close error")
        
        # Inject client
        rag_cache._client = mock_client
        rag_cache.reset_client()
        # Verify client is reset to None and exception caught
        self.assertIsNone(rag_cache._client)


if __name__ == "__main__":
    unittest.main()
