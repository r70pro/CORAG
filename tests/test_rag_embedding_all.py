"""
Comprehensive unit tests for rag/embedding.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag.embedding as rag_emb


class TestRAGEmbeddingAll(unittest.TestCase):

    def test_get_collection_name_branches(self):
        # 1. Over 40 characters limit
        name = rag_emb.get_collection_name("a" * 50)
        self.assertTrue(len(name) <= 60)

        # 2. None model name with exception
        with patch("rag.embedding.load_settings", side_effect=Exception("Load failed")):
            name2 = rag_emb.get_collection_name(None)
            self.assertTrue("sentence-transformers" in name2 or "minilm" in name2 or "baai" in name2 or "bge-large-en" in name2)

    @patch("rag.embedding.QdrantClient")
    def test_get_qdrant_client(self, mock_qdrant):
        # 1. Config is None (default lookup)
        client = rag_emb.get_qdrant_client()
        self.assertIsNotNone(client)
        
        # 2. Config is provided
        client2 = rag_emb.get_qdrant_client({"host": "custom", "port": 6333})
        self.assertIsNotNone(client2)

    @patch("rag.embedding.get_qdrant_client")
    def test_is_healthy_success_and_exception(self, mock_get_client):
        # 1. Success
        mock_client = mock_get_client.return_value
        mock_client.get_collections.return_value = MagicMock()
        self.assertTrue(rag_emb.is_healthy())

        # 2. Exception
        mock_client.get_collections.side_effect = Exception("Qdrant down")
        self.assertFalse(rag_emb.is_healthy())

    @patch("rag.embedding.get_qdrant_client")
    def test_init_collection_exception(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.get_collections.side_effect = Exception("Qdrant down")
        
        # Expect exception to be raised on dim detection or creation fallback
        with self.assertRaises(Exception):
            rag_emb.init_collection(model_name="model_name", dimension=384)

    @patch("rag.embedding.get_qdrant_client")
    @patch("rag.embedding.get_embedding_dimension")
    def test_init_collection_no_dimension(self, mock_dim, mock_get_client):
        mock_dim.return_value = 128
        mock_client = mock_get_client.return_value
        
        # Configure collections list mock to not contain model
        mock_collections_res = MagicMock()
        mock_collections_res.collections = []
        mock_client.get_collections.return_value = mock_collections_res

        rag_emb.init_collection(model_name="auto_dim_model")
        mock_client.create_collection.assert_called_once()

    @patch("rag.embedding.get_qdrant_client")
    @patch("rag.embedding.get_embedding_dimension")
    def test_init_collection_with_dimension(self, mock_dim, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_collections_res = MagicMock()
        mock_collections_res.collections = []
        mock_client.get_collections.return_value = mock_collections_res

        # Call with explicit dimension = 256
        rag_emb.init_collection(model_name="custom_dim_model", dimension=256)
        mock_dim.assert_not_called()
        mock_client.create_collection.assert_called_once()

    @patch("rag.embedding.get_qdrant_client")
    def test_init_collection_already_exists(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_col = MagicMock()
        mock_col.name = "olmocr_documents_sentence-transformers_all-minilm-l6-v2"
        
        mock_collections_res = MagicMock()
        mock_collections_res.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections_res

        # Call init_collection -> early returns print message
        rag_emb.init_collection(model_name="sentence-transformers/all-MiniLM-L6-v2")
        mock_client.create_collection.assert_not_called()

    @patch("rag.embedding.get_qdrant_client")
    def test_get_collection_info_success(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_info = MagicMock()
        mock_info.points_count = 5
        mock_info.vectors_count = 5
        mock_info.indexed_vectors_count = 5
        mock_info.status.value = "green"
        mock_client.get_collection.return_value = mock_info
        
        info = rag_emb.get_collection_info("model_name")
        self.assertEqual(info["status"], "green")
        self.assertEqual(info["points_count"], 5)

    @patch("rag.embedding.get_qdrant_client")
    def test_get_collection_info_not_found(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        info = rag_emb.get_collection_info("model_name")
        self.assertEqual(info["status"], "not_found")
        self.assertEqual(info["points_count"], 0)

    @patch("rag.embedding.get_qdrant_client")
    def test_delete_run_vectors_success_and_exception(self, mock_get_client):
        # 1. Success
        rag_emb.delete_run_vectors("run1", "model")

        # 2. Exception
        mock_client = mock_get_client.return_value
        mock_client.delete.side_effect = Exception("Qdrant error")
        rag_emb.delete_run_vectors("run1", "model")

    @patch("rag.embedding.get_qdrant_client")
    def test_delete_collection_success_and_exception(self, mock_get_client):
        # 1. Success
        rag_emb.delete_collection("model")

        # 2. Exception
        mock_client = mock_get_client.return_value
        mock_client.delete_collection.side_effect = Exception("Qdrant error")
        rag_emb.delete_collection("model")

    @patch("rag.embedding.get_qdrant_client")
    def test_upsert_chunks_empty(self, mock_get_client):
        # Empty list early return
        self.assertEqual(rag_emb.upsert_chunks([]), [])

    @patch("rag.embedding.get_qdrant_client")
    @patch("rag.embedding.get_embedding_dimension")
    @patch("sentence_transformers.SentenceTransformer")
    def test_upsert_chunks_success(self, mock_transformer, mock_dim, mock_get_client):
        mock_dim.return_value = 4
        mock_client = mock_get_client.return_value
        
        # Configure collections list mock
        mock_collections_res = MagicMock()
        mock_collections_res.collections = []
        mock_client.get_collections.return_value = mock_collections_res
        
        mock_model = mock_transformer.return_value
        mock_tolist = MagicMock()
        mock_tolist.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_model.encode.return_value = mock_tolist

        chunks = [{
            "chunk_id": "c1",
            "doc_id": "d1",
            "run_id": "r1",
            "chunk_index": 0,
            "text": "text",
            "page_number": 1,
            "document_type": "type",
            "author": "author",
            "date_extracted": "2026-07-11",
            "section_type": "sec",
            "patient_name": "patient",
            "token_count": 5
        }]

        # 1. Normal success
        rag_emb._embedding_model = None
        res = rag_emb.upsert_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(len(res), 1)

        # 2. None model with settings load exception
        with patch("rag.embedding.load_settings", side_effect=Exception("Disk error")):
            rag_emb._embedding_model = None
            res2 = rag_emb.upsert_chunks(chunks, model_name=None)
            self.assertEqual(len(res2), 1)

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedding_dimension(self, mock_transformer):
        # Reset cached model reference to invoke SentenceTransformer init
        rag_emb._embedding_model = None

        # From model.get_embedding_dimension()
        mock_model = mock_transformer.return_value
        mock_model.get_embedding_dimension.return_value = 128
        self.assertEqual(rag_emb.get_embedding_dimension("model"), 128)

    def test_encode_texts_empty(self):
        self.assertEqual(rag_emb.encode_texts([]), [])

    @patch("rag.embedding.load_settings")
    @patch("sentence_transformers.SentenceTransformer")
    def test_load_embedding_model_branches(self, mock_transformer, mock_load_settings):
        # 1. Trigger Exception inside load_settings lookup
        mock_load_settings.side_effect = Exception("Load settings failed")
        
        # Reset cached reference
        rag_emb._embedding_model = None
        model = rag_emb.load_embedding_model(None)
        self.assertIsNotNone(model)

        # 2. Singleton caching hit path
        model2 = rag_emb.load_embedding_model(None)
        self.assertEqual(model, model2)

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_query(self, mock_transformer):
        # Reset cached reference
        rag_emb._embedding_model = None

        mock_model = mock_transformer.return_value
        mock_tolist = MagicMock()
        mock_tolist.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_tolist

        vec = rag_emb.encode_query("my query", "model_name")
        self.assertEqual(vec, [0.1, 0.2, 0.3])

    def test_double_checked_locks_embedding(self):
        # Test double checked lock inner path for embedding model
        rag_emb._embedding_model = None
        class MockLock:
            def __enter__(self):
                rag_emb._embedding_model = "fake_inner_model"
                rag_emb._embedding_model_name = "my_model"
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        with patch("rag.embedding._embedding_model_lock", MockLock()):
            res = rag_emb.load_embedding_model("my_model")
            self.assertEqual(res, "fake_inner_model")

        # Test double checked lock inner path for reranker model
        rag_emb._reranker_model = None
        class MockLockRerank:
            def __enter__(self):
                rag_emb._reranker_model = "fake_inner_reranker"
                rag_emb._reranker_model_name = "my_reranker"
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        with patch("rag.embedding._reranker_model_lock", MockLockRerank()):
            res = rag_emb.load_reranker_model("my_reranker")
            self.assertEqual(res, "fake_inner_reranker")

    @patch("rag.cache.is_healthy")
    @patch("rag.cache.get_cached_embedding")
    @patch("rag.cache.cache_embedding")
    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_texts_caching_and_exceptions(self, mock_transformer, mock_cache_write, mock_cache_read, mock_cache_healthy):
        # Setup mock model
        mock_model = mock_transformer.return_value
        mock_tolist = MagicMock()
        mock_tolist.tolist.return_value = [[0.9, 0.9]]
        mock_model.encode.return_value = mock_tolist

        # Baseline: Redis is down
        mock_cache_healthy.return_value = False
        mock_cache_read.return_value = None

        # 1. cache.is_healthy raises Exception (lines 160-161)
        mock_cache_healthy.side_effect = Exception("Redis crash")
        rag_emb._embedding_model = None
        res = rag_emb.encode_texts(["text1"], "my_model")
        self.assertEqual(res, [[0.9, 0.9]])

        # 2. redis_healthy is True, but cache read returns None (miss) (lines 174-175)
        mock_cache_healthy.side_effect = None
        mock_cache_healthy.return_value = True
        mock_cache_read.return_value = None
        rag_emb._embedding_model = None
        res = rag_emb.encode_texts(["text1"], "my_model")
        mock_cache_write.assert_called_with("text1", [0.9, 0.9], "my_model")

        # 3. redis_healthy is True, but cache read raises Exception (lines 176-178)
        mock_cache_read.side_effect = Exception("Read failed")
        rag_emb._embedding_model = None
        res = rag_emb.encode_texts(["text1"], "my_model")
        self.assertEqual(res, [[0.9, 0.9]])

        # 4. redis_healthy is True, but cache write raises Exception (lines 199-200)
        mock_cache_read.side_effect = None
        mock_cache_read.return_value = None
        mock_cache_write.side_effect = Exception("Write failed")
        rag_emb._embedding_model = None
        res = rag_emb.encode_texts(["text1"], "my_model")
        self.assertEqual(res, [[0.9, 0.9]])

    @patch("rag.cache.is_healthy")
    @patch("rag.cache.get_cached_embedding")
    @patch("rag.cache.cache_embedding")
    @patch("sentence_transformers.SentenceTransformer")
    @patch("rag.embedding.load_settings")
    def test_encode_query_caching_and_exceptions(self, mock_load_settings, mock_transformer, mock_cache_write, mock_cache_read, mock_cache_healthy):
        mock_model = mock_transformer.return_value
        mock_tolist = MagicMock()
        mock_tolist.tolist.return_value = [0.9, 0.9]
        mock_model.encode.return_value = mock_tolist

        # Baseline: Redis is down
        mock_cache_healthy.return_value = False
        mock_cache_read.return_value = None

        # 1. load_settings raises Exception (lines 218-219)
        mock_load_settings.side_effect = Exception("disk error")
        rag_emb._embedding_model = None
        res = rag_emb.encode_query("query1", None)
        self.assertEqual(res, [0.9, 0.9])

        # 2. cache.is_healthy raises Exception (lines 227-228)
        mock_load_settings.side_effect = None
        mock_load_settings.return_value = {}
        mock_cache_healthy.side_effect = Exception("Redis error")
        rag_emb._embedding_model = None
        res = rag_emb.encode_query("query1", None)
        self.assertEqual(res, [0.9, 0.9])

        # 3. redis_healthy is True, cache read raises Exception (lines 235-236)
        mock_cache_healthy.side_effect = None
        mock_cache_healthy.return_value = True
        mock_cache_read.side_effect = Exception("read error")
        rag_emb._embedding_model = None
        res = rag_emb.encode_query("query1", "my_model")
        self.assertEqual(res, [0.9, 0.9])

        # 4. redis_healthy is True, cache write raises Exception (lines 248-249)
        mock_cache_read.side_effect = None
        mock_cache_read.return_value = None
        mock_cache_write.side_effect = Exception("write error")
        rag_emb._embedding_model = None
        res = rag_emb.encode_query("query1", "my_model")
        self.assertEqual(res, [0.9, 0.9])

        # 5. redis_healthy is True, cache hits (line 233-234)
        mock_cache_read.return_value = [0.8, 0.8]
        rag_emb._embedding_model = None
        res = rag_emb.encode_query("query1", "my_model")
        self.assertEqual(res, [0.8, 0.8])

    @patch("sentence_transformers.CrossEncoder")
    @patch("rag.embedding.load_settings")
    def test_load_reranker_model_branches(self, mock_load_settings, mock_cross_encoder):
        # Reset cached reference
        rag_emb._reranker_model = None
        rag_emb._reranker_model_name = None

        # 1. load_settings exception path for model and device
        mock_load_settings.side_effect = Exception("disk error")
        res = rag_emb.load_reranker_model(None, None)
        self.assertIsNotNone(res)

        # 2. Singleton caching hit path
        res2 = rag_emb.load_reranker_model(None, None)
        self.assertEqual(res, res2)


if __name__ == "__main__":
    unittest.main()
