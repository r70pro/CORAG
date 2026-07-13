"""
Integration tests for the RAG subsystem.

Ensures real interactions with local Docker services:
- PostgreSQL (database tables schema and CRUD operations)
- Redis (cache get/set query results)
- MinIO (S3 buckets upload and download verification)
- Qdrant (cosine vector similarity query verification)
"""

import os
import unittest
import time
from datetime import date

# Activate live settings
os.environ["TESTING"] = "false"

from rag import db as rag_db
from rag import storage as rag_storage
from rag import cache as rag_cache
from rag import embedding as rag_emb
from rag import retriever as rag_ret
from rag_infra_manager import start_and_init_rag, is_rag_infrastructure_ready


class TestRAGIntegration(unittest.TestCase):

    started_infra = False

    @classmethod
    def setUpClass(cls):
        # Auto start RAG infra if it's not running
        if not is_rag_infrastructure_ready():
            print("RAG Infrastructure not ready. Starting it for integration tests...")
            success, msg = start_and_init_rag()
            if not success:
                raise unittest.SkipTest(f"Skipping integration tests: Failed to start RAG infra: {msg}")
            cls.started_infra = True

        # Ensure all services are initialized
        from rag_infra_manager import init_rag_database, init_rag_storage, init_rag_vector_store
        init_rag_database()
        init_rag_storage()
        init_rag_vector_store()

    @classmethod
    def tearDownClass(cls):
        if cls.started_infra:
            print("Stopping RAG infrastructure started for integration tests...")
            from rag_infra_manager import stop_rag_infrastructure
            stop_rag_infrastructure()

    def test_01_postgres_integration(self):
        self.assertTrue(rag_db.is_healthy(), "PostgreSQL should be healthy")

        run_id = "test_run_int_999"
        doc_id = "test_doc_int_999"
        chunk_id = "test_chunk_int_999"

        # Register run
        rag_db.register_run(run_id, "/mock/run/dir", total_documents=1)
        self.assertFalse(rag_db.is_run_indexed(run_id, check_vector_store=False))

        # Register document
        rag_db.register_document(
            doc_id=doc_id,
            run_id=run_id,
            original_filename="integration_test.pdf",
            pdf_total_pages=3
        )

        # Insert chunks
        chunks = [
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "run_id": run_id,
                "chunk_index": 0,
                "text": "This is a live integration test for PostgreSQL schema.",
                "char_start": 0,
                "char_end": 50,
                "page_number": 1,
                "document_type": "specialist_letter",
                "author": "Dr Integration",
                "date_extracted": date(2026, 7, 11),
                "date_raw": "11/07/2026",
                "section_type": "clinical_findings",
                "patient_name": "Int Patient",
                "token_count": 10,
                "embedding_model": "test-model",
                "qdrant_point_id": "mock-qdrant-point-id-999"
            }
        ]
        rag_db.insert_chunks(chunks)

        # Mark document and run as indexed
        rag_db.mark_document_indexed(doc_id)
        rag_db.mark_run_indexed(run_id, total_chunks=1)

        self.assertTrue(rag_db.is_run_indexed(run_id, check_vector_store=False))

        # Query stats
        stats = rag_db.get_corpus_stats()
        self.assertTrue(stats["total_chunks"] >= 1)

        # Clean up database records
        rag_db.delete_run_data(run_id)

    def test_02_redis_integration(self):
        self.assertTrue(rag_cache.is_healthy(), "Redis should be healthy")

        test_query = "integration query key 123"
        test_result = {"answer": "this is a cached integration answer", "score": 0.99}

        # Query cache test
        rag_cache.cache_query_result(test_query, test_result, ttl=10)
        cached = rag_cache.get_cached_query(test_query)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], test_result["answer"])

        # Chat history test
        session_id = "test_session_int_123"
        messages = [{"role": "user", "content": "hello integration"}]
        rag_cache.save_chat_history(session_id, messages, ttl=10)
        history = rag_cache.get_chat_history(session_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "hello integration")

        # Invalidate
        rag_cache.invalidate_query_cache()
        self.assertIsNone(rag_cache.get_cached_query(test_query))

    def test_03_minio_integration(self):
        self.assertTrue(rag_storage.is_healthy(), "MinIO S3 should be healthy")

        run_id = "int_run_minio"
        doc_id = "int_doc_minio"
        filename = "test_md.md"
        text = "# Markdown integration test\nThis is a sample document for MinIO upload verification."

        # Upload
        key = rag_storage.upload_markdown_text(run_id, doc_id, filename, text)
        self.assertEqual(key, f"{run_id}/{doc_id}/{filename}")

        # Download & verify
        downloaded = rag_storage.get_markdown_text(key)
        self.assertEqual(downloaded, text)

        # Storage stats
        stats = rag_storage.get_storage_stats()
        self.assertTrue(stats[rag_storage.BUCKET_MARKDOWN]["objects"] >= 1)

        # Cleanup MinIO
        rag_storage.delete_run_objects(run_id)

    def test_04_qdrant_integration(self):
        self.assertTrue(rag_emb.is_healthy(), "Qdrant should be healthy")

        run_id = "int_run_qdrant"
        doc_id = "int_doc_qdrant"
        model_name = "BAAI/bge-large-en-v1.5"

        # Check collection info
        col_info = rag_emb.get_collection_info(model_name)
        self.assertNotEqual(col_info["status"], "not_found")

        # Insert points
        chunks = [
            {
                "chunk_id": "chunk_qdrant_1",
                "doc_id": doc_id,
                "run_id": run_id,
                "chunk_index": 0,
                "text": "The patient experienced acute shoulder pain after a motorcycle crash in 2014.",
                "char_start": 0,
                "char_end": 100,
                "page_number": 2,
                "document_type": "specialist_letter",
                "author": "Dr Eugene Ek",
                "date_extracted": "2020-08-27",
                "section_type": "history",
                "patient_name": "Francis Van Rossum",
                "token_count": 20
            }
        ]

        # Upsert
        updated_chunks = rag_emb.upsert_chunks(chunks, model_name=model_name)
        self.assertIsNotNone(updated_chunks[0].get("qdrant_point_id"))

        # Verify Qdrant points count updated
        time.sleep(1) # Let Qdrant index it
        col_info2 = rag_emb.get_collection_info(model_name)
        self.assertTrue(col_info2["points_count"] >= 1)

        # Search similar via retriever
        results = rag_ret.search_similar(
            query="Tell me about the motorcycle accident and shoulder pain",
            top_k=1,
            score_threshold=0.1
        )
        self.assertTrue(len(results) >= 1)
        self.assertTrue("motorcycle" in results[0]["text"].lower())

        # Cleanup run vectors
        rag_emb.delete_run_vectors(run_id, model_name=model_name)


if __name__ == "__main__":
    unittest.main()
