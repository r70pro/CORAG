"""
Unit tests for the OLMOCR RAG subsystem.

Tests:
- Chunker (medicolegal metadata extraction, paragraph splitting)
- DB layer (PostgreSQL registry operations, chunks registry)
- Storage layer (MinIO buckets upload/retrieve)
- Cache layer (Redis query and embedding caching)
- Embedding & Vector Store (dense representation, Qdrant upserts)
- Retriever & Analyzer (metadata-filtered search, LLM prompt formatting)
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

from rag import chunker, db as rag_db, storage as rag_storage, cache as rag_cache, embedding as rag_emb, retriever as rag_ret, analyzer as rag_anal


class TestRAGSubsystem(unittest.TestCase):

    def setUp(self):
        pass

    # ── Chunker tests ──────────────────────────────────────────

    def test_parse_date(self):
        # DD/MM/YYYY
        self.assertEqual(chunker._parse_date("Consulted on 12/02/2018"), "2018-02-12")
        # DD.MM.YYYY
        self.assertEqual(chunker._parse_date("Dated 20.11.2018"), "2018-11-20")
        # Verbal named month
        self.assertEqual(chunker._parse_date("Dictated 12 February 2018"), "2018-02-12")
        self.assertEqual(chunker._parse_date("Seen on August 27, 2020"), "2020-08-27")
        # No date
        self.assertIsNone(chunker._parse_date("No date here"))

    def test_extract_author(self):
        self.assertEqual(chunker._extract_author("Dr Eugene Ek, Orthopaedic Surgeon"), "Eugene Ek")
        self.assertEqual(chunker._extract_author("Kind regards\n\nDr Paul Borbas, MD"), "Paul Borbas")
        self.assertEqual(chunker._extract_author("Yours sincerely\n\n(Dictated but not sighted)\n\nA/Prof. Eugene T. Ek"), "Eugene T. Ek")

    def test_classify_document_type(self):
        self.assertEqual(chunker._classify_document_type("Dear Dr Sybille, thanks for referring Francis..."), "specialist_letter")
        self.assertEqual(chunker._classify_document_type("Date | Clinical Notes | Signature\n12.10.09 | strained abdomen"), "clinical_notes")

    def test_extract_patient_name(self):
        self.assertEqual(chunker._extract_patient_name("Re: Francis VAN ROSSUM DOB: 28.11.1971"), "Francis VAN ROSSUM")
        self.assertEqual(chunker._extract_patient_name("Re: Mr. Francis Van Rossum"), "Francis Van Rossum")

    def test_chunk_document_basic(self):
        test_markdown = """# Clinical Report
Patient Name: Francis VAN ROSSUM
DOB: 28.11.1971

Dear Dr. Sybille Japhary-Dobber,
I reviewed Francis's recent MRI scans. The recent MRI scans do confirm that his superior labral repair has failed and he has a persistent SLAP tear.

Yours sincerely,
A/Prof. Eugene T. Ek
"""
        chunks = chunker.chunk_document(
            markdown_text=test_markdown,
            doc_id="doc_1",
            run_id="run_1",
            max_chunk_size=500,
            chunk_overlap=50
        )
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0]["patient_name"], "Francis VAN ROSSUM")
        self.assertEqual(chunks[0]["author"], "Eugene T. Ek")
        self.assertEqual(chunks[0]["document_type"], "specialist_letter")

    # ── DB layers mock tests ────────────────────────────────────

    @patch("rag.db.get_connection")
    def test_register_run(self, mock_conn):
        mock_cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        rag_db.register_run("run_123", "/path/to/run", total_documents=2)
        mock_cur.execute.assert_called_once()

    @patch("rag.db.get_connection")
    def test_register_document(self, mock_conn):
        mock_cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        rag_db.register_document(
            doc_id="doc_1",
            run_id="run_1",
            original_filename="report.pdf",
            pdf_total_pages=5
        )
        mock_cur.execute.assert_called_once()

    # ── Storage layers mock tests ───────────────────────────────

    @patch("rag.storage.Minio")
    def test_storage_upload_pdf(self, mock_minio):
        mock_client = mock_minio.return_value
        mock_client.bucket_exists.return_value = True
        
        # Test fput_object trigger
        with patch("rag.storage.get_client", return_value=mock_client):
            key = rag_storage.upload_pdf("run_1", "doc_1", "/tmp/nonexistent.pdf")
            self.assertEqual(key, "run_1/doc_1/nonexistent.pdf")
            mock_client.fput_object.assert_called_once()

    # ── Cache layer tests ───────────────────────────────────────

    @patch("redis.Redis")
    def test_cache_query(self, mock_redis_class):
        mock_redis = mock_redis_class.return_value
        mock_redis.get.return_value = '{"answer": "test result"}'
        
        with patch("rag.cache.get_client", return_value=mock_redis):
            res = rag_cache.get_cached_query("who is the patient?")
            self.assertEqual(res["answer"], "test result")
            mock_redis.get.assert_called_once()

            rag_cache.cache_query_result("who is the patient?", {"answer": "test result"})
            mock_redis.set.assert_called_once()

    # ── Retriever & prompt context tests ────────────────────────

    def test_format_context_for_llm(self):
        test_results = [
            {
                "original_filename": "report.pdf",
                "page_number": 3,
                "author": "Dr Ek",
                "date_extracted": "2020-08-27",
                "document_type": "specialist_letter",
                "text": "The superior labral repair has failed."
            }
        ]
        context = rag_ret.format_context_for_llm(test_results)
        self.assertTrue("[Source 1]" in context)
        self.assertTrue("File: report.pdf" in context)
        self.assertTrue("Page: 3" in context)
        self.assertTrue("Author: Dr Ek" in context)
        self.assertTrue("The superior labral repair has failed." in context)

    def test_build_prompt(self):
        query = "What failed?"
        context = "Source 1: The repair failed."
        messages = rag_anal.build_prompt(query, context, mode="free_qa")
        
        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue("medicolegal document analyst" in messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertTrue("DOCUMENT EXCERPTS:" in messages[1]["content"])
        self.assertTrue("What failed?" in messages[1]["content"])

    def test_get_analysis_modes(self):
        modes = rag_anal.get_analysis_modes()
        self.assertTrue("free_qa" in modes)
        self.assertTrue("timeline" in modes)

    @patch("httpx.post")
    def test_query_llm_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "The repair of the SLAP tear failed."}}]
        }
        mock_post.return_value = mock_response

        res = rag_anal.query_llm([], "http://local:8000/v1", "phi-4")
        self.assertEqual(res, "The repair of the SLAP tear failed.")

    @patch("httpx.post")
    def test_query_llm_error_status(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        res = rag_anal.query_llm([], "http://local:8000/v1", "phi-4")
        self.assertTrue("Error: LLM server returned HTTP 500" in res)

    @patch("httpx.post")
    def test_query_llm_exception(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        res = rag_anal.query_llm([], "http://local:8000/v1", "phi-4")
        self.assertTrue("Error: Cannot connect to LLM server" in res)

    @patch("httpx.stream")
    def test_query_llm_streaming_success(self, mock_stream):
        # Create a mock response with a stream iterator
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: [DONE]'
        ]
        
        # Configure the context manager of httpx.stream
        mock_stream.return_value.__enter__.return_value = mock_response

        chunks = list(rag_anal.query_llm_streaming([], "http://local:8000/v1", "phi-4"))
        self.assertEqual(chunks, ["Hello", " world"])

    @patch("httpx.stream")
    def test_query_llm_streaming_exception(self, mock_stream):
        import httpx
        mock_stream.side_effect = httpx.ConnectError("Connection refused")

        chunks = list(rag_anal.query_llm_streaming([], "http://local:8000/v1", "phi-4"))
        self.assertTrue("Error" in "".join(chunks))

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    def test_analyze_pipeline(self, mock_query_stream, mock_search):
        # Mock retrieval returning nothing
        mock_search.return_value = []
        res1 = "".join(rag_anal.analyze("query"))
        self.assertTrue("No relevant document excerpts" in res1)

        # Mock retrieval returning results
        mock_search.return_value = [
            {
                "original_filename": "report.pdf",
                "page_number": 1,
                "text": "sample text"
            }
        ]
        mock_query_stream.return_value = ["response content"]
        res2 = "".join(rag_anal.analyze("query", stream=True))
        self.assertEqual(res2, "response content")

    # ── DB Layer Expansion ──────────────────────────────────────

    @patch("rag.db.get_connection")
    def test_db_read_operations(self, mock_conn):
        mock_cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        
        # get_indexed_runs
        mock_cur.fetchall.return_value = [{"run_id": "run_1"}]
        runs = rag_db.get_indexed_runs()
        self.assertEqual(len(runs), 1)

        # get_all_runs
        rag_db.get_all_runs()

        # get_documents_for_run
        rag_db.get_documents_for_run("run_1")

        # get_all_documents
        rag_db.get_all_documents()

        # get_chunks_for_document
        rag_db.get_chunks_for_document("doc_1")

        # get_runs_with_stats
        mock_cur.fetchall.return_value = [
            {
                "run_id": "run_1",
                "run_dir": "dir",
                "total_documents": 2,
                "total_chunks": 10,
                "unique_authors": 1,
                "earliest_date": "2026-01-01",
                "latest_date": "2026-01-02",
                "indexed_at": "2026-01-01 00:00:00",
                "status": "indexed"
            }
        ]
        stats = rag_db.get_runs_with_stats()
        self.assertEqual(len(stats), 1)

        # get_authors_for_run
        mock_cur.fetchall.return_value = [("Author Name",)]
        authors = rag_db.get_authors_for_run("run_1")
        self.assertEqual(authors, ["Author Name"])

        # get_doc_types_for_run
        mock_cur.fetchall.return_value = [("Document Type",)]
        types = rag_db.get_doc_types_for_run("run_1")
        self.assertEqual(types, ["Document Type"])

        # get_date_range_for_run (case with values)
        mock_cur.fetchone.return_value = {"earliest": "2026-01-01", "latest": "2026-01-02"}
        date_range1 = rag_db.get_date_range_for_run("run_1")
        self.assertEqual(date_range1["earliest"], "2026-01-01")
        self.assertEqual(date_range1["latest"], "2026-01-02")

        # get_date_range_for_run (case with None)
        mock_cur.fetchone.return_value = None
        date_range2 = rag_db.get_date_range_for_run("run_1")
        self.assertIsNone(date_range2["earliest"])
        self.assertIsNone(date_range2["latest"])

        # delete_run_data
        rag_db.delete_run_data("run_1")

    # ── Storage Layer Expansion ──────────────────────────────────

    @patch("rag.storage.Minio")
    def test_storage_downloads(self, mock_minio):
        mock_client = mock_minio.return_value
        with patch("rag.storage.get_client", return_value=mock_client):
            # get_file_content
            mock_response = MagicMock()
            mock_response.read.return_value = b"file content"
            mock_client.get_object.return_value = mock_response
            
            content = rag_storage.get_file_content("bucket", "key")
            self.assertEqual(content, b"file content")

            # list_objects
            mock_obj = MagicMock()
            mock_obj.object_name = "key1"
            mock_client.list_objects.return_value = [mock_obj]
            self.assertEqual(rag_storage.list_objects("bucket"), ["key1"])

    # ── Cache Layer Expansion ────────────────────────────────────

    @patch("redis.Redis")
    def test_cache_embeddings_and_stats(self, mock_redis_class):
        mock_redis = mock_redis_class.return_value
        with patch("rag.cache.get_client", return_value=mock_redis):
            # cache_embedding
            rag_cache.cache_embedding("text", [0.1, 0.2], "model")
            mock_redis.set.assert_called_once()
            
            # get_cached_embedding
            mock_redis.get.return_value = '[0.1, 0.2]'
            emb = rag_cache.get_cached_embedding("text", "model")
            self.assertEqual(emb, [0.1, 0.2])

            # increment_stat / get_stat
            rag_cache.increment_stat("test_stat")
            mock_redis.incr.assert_called_once()

            mock_redis.get.return_value = '10'
            self.assertEqual(rag_cache.get_stat("test_stat"), 10)

    # ── Embedding Layer Expansion ────────────────────────────────

    def test_collection_naming(self):
        col_name = rag_emb.get_collection_name("model/name-here-v1")
        self.assertTrue("model_name-here-v1" in col_name)

    # ── Retriever Layer Expansion (MMR) ──────────────────────────

    def test_mmr_reranking(self):
        results = [
            {"score": 0.9, "text": "acute shoulder pain"},
            {"score": 0.85, "text": "shoulder pain acute"}, # duplicate
            {"score": 0.7, "text": "patient underwent knee surgery"} # diverse
        ]
        # Top 2 re-ranked should pick #1 and #3
        reranked = rag_ret._mmr_rerank(results, [0.1]*384, top_k=2, lambda_param=0.5)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["text"], "acute shoulder pain")
        self.assertEqual(reranked[1]["text"], "patient underwent knee surgery")

    # ── Infrastructure Status Expansion ──────────────────────────

    @patch("subprocess.run")
    def test_infra_service_status(self, mock_run):
        import rag_infra_manager
        
        # Mock ps command stdout returning json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"Service": "postgres", "State": "running", "Health": "healthy"}\n'
        )
        
        statuses = rag_infra_manager.get_rag_service_status()
        self.assertEqual(statuses["postgres"], "healthy")

        html = rag_infra_manager.get_rag_status_html()
        self.assertTrue("PostgreSQL" in html)


if __name__ == "__main__":
    unittest.main()
