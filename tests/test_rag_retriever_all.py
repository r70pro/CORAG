"""
Comprehensive unit tests for rag/retriever.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import datetime

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag.retriever as rag_ret
import rag.embedding as rag_emb


class TestRAGRetrieverAll(unittest.TestCase):

    def setUp(self):
        # Reset cached model to avoid test leakage
        rag_emb._embedding_model = None

    def test_mmr_rerank_edges(self):
        # 1. Empty/short results
        results = [{"score": 0.9, "text": "shoulder"}]
        self.assertEqual(rag_ret._mmr_rerank(results, [0.1]*10, top_k=5), results)

        # 2. Normal MMR diversity selections
        results2 = [
            {"score": 0.9, "text": "shoulder pain acute"},
            {"score": 0.85, "text": "acute shoulder pain"}, # redundant
            {"score": 0.7, "text": "knee replacement surgery"} # diverse
        ]
        reranked = rag_ret._mmr_rerank(results2, [0.1]*10, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["text"], "shoulder pain acute")
        self.assertEqual(reranked[1]["text"], "knee replacement surgery")

    def test_mmr_rerank_performance(self):
        # Verify that optimized MMR executes rapidly on large inputs (e.g. 600 candidates to top_k=100)
        import time
        large_results = [
            {"score": 0.9 - (i * 0.001), "text": f"some clinical note fragment about patient symptoms number {i}"}
            for i in range(600)
        ]
        start_time = time.time()
        reranked = rag_ret._mmr_rerank(large_results, [0.1]*10, top_k=100)
        duration = time.time() - start_time
        self.assertLess(duration, 0.5)
        self.assertEqual(len(reranked), 100)

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_with_filters(self, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        
        # Configure Qdrant query response
        mock_client = mock_qdrant_client.return_value
        mock_result = MagicMock()
        mock_result.id = "point1"
        mock_result.score = 0.85
        mock_result.payload = None
        mock_client.search.return_value = [mock_result]
        
        # Configure DB chunks return
        mock_db_chunks.return_value = [
            {
                "qdrant_point_id": "point1",
                "text": "The patient has shoulder pain",
                "original_filename": "report.pdf",
                "page_number": 2,
                "author": "Dr Ek",
                "date_extracted": "2020-08-27",
                "date_raw": "27/08/2020",
                "section_type": "history",
                "patient_name": "Francis"
            }
        ]

        # Call with all filters to hit parsing branches
        res = rag_ret.search_similar(
            query="query text",
            top_k=1,
            doc_type_filter="letter",
            author_filter="Dr Ek",
            date_from=1577836800.0,
            date_to=1609459200.0,
            run_id_filter="run1",
            doc_id_filter="doc1",
            score_threshold=0.1
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["text"], "The patient has shoulder pain")

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    @patch("rag.db.get_indexed_runs")
    def test_search_similar_run_id_path_resolution(self, mock_runs, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_runs.return_value = [
            {"run_id": "run_resolved_123", "run_dir": "/workspace/runs/run_resolved_123"}
        ]
        mock_client = mock_qdrant_client.return_value
        mock_result = MagicMock()
        mock_result.id = "point1"
        mock_result.score = 0.9
        mock_result.payload = None
        mock_client.search.return_value = [mock_result]
        mock_db_chunks.return_value = [
            {"qdrant_point_id": "point1", "text": "path resolved chunk"}
        ]

        res = rag_ret.search_similar(
            query="query",
            run_id_filter="/workspace/runs/run_resolved_123 (3 files)"
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["text"], "path resolved chunk")


    @patch("rag.retriever.encode_query")
    def test_search_similar_exception(self, mock_encode):
        mock_encode.side_effect = Exception("Embed generation failed")
        
        # Exception should bubble up
        with self.assertRaises(Exception):
            rag_ret.search_similar("query")

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_date_int_filters(self, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_qdrant_client.return_value
        mock_client.search.return_value = []
        mock_db_chunks.return_value = []

        # Call with date filters as standard ISO strings
        rag_ret.search_similar(
            query="test query",
            date_from="2024-07-19",
            date_to="2026-07-19"
        )

        # Retrieve the filter conditions passed to client.search
        self.assertTrue(mock_client.search.called)
        called_args, called_kwargs = mock_client.search.call_args
        query_filter = called_kwargs.get("query_filter")
        self.assertIsNotNone(query_filter)
        
        # Check that we have FieldCondition for date_int with gte=20240719 and lte=20260719
        conditions = query_filter.must
        self.assertEqual(len(conditions), 2)
        
        c1, c2 = conditions[0], conditions[1]
        self.assertEqual(c1.key, "date_int")
        self.assertEqual(c1.range.gte, 20240719.0)
        self.assertEqual(c2.key, "date_int")
        self.assertEqual(c2.range.lte, 20260719.0)

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_python_date_fallback(self, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        
        mock_client = mock_qdrant_client.return_value
        p1 = MagicMock()
        p1.id = "p1"
        p1.score = 0.9
        p1.payload = {"date_extracted": None}
        
        p2 = MagicMock()
        p2.id = "p2"
        p2.score = 0.8
        p2.payload = {"date_extracted": "2019-12-31"}

        p3 = MagicMock()
        p3.id = "p3"
        p3.score = 0.7
        p3.payload = {"date_extracted": "2021-01-01"}

        mock_client.search.return_value = [p1, p2, p3]
        mock_db_chunks.return_value = [
            {"qdrant_point_id": "p1", "text": "no date", "date_extracted": None},
            {"qdrant_point_id": "p2", "text": "too early", "date_extracted": "2019-12-31"},
            {"qdrant_point_id": "p3", "text": "too late", "date_extracted": "2021-01-01"},
        ]

        res = rag_ret.search_similar(
            query="test query",
            date_from="2020-01-01",
            date_to="2020-12-31"
        )
        self.assertEqual(len(res), 0)

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_date_int_exception(self, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_qdrant_client.return_value
        mock_client.search.return_value = []
        mock_db_chunks.return_value = []

        class MockInt(int):
            def __new__(cls, val, *args, **kwargs):
                if str(val) in ("20240719", "20260719"):
                    raise ValueError("Mock int error")
                return super().__new__(cls, val, *args, **kwargs)

        with patch("builtins.int", MockInt):
            rag_ret.search_similar(
                query="test query",
                date_from="2024-07-19",
                date_to="2026-07-19"
            )

    def test_normalize_iso_date_all_branches(self):
        self.assertIsNone(rag_ret._normalize_iso_date(None))
        self.assertIsNone(rag_ret._normalize_iso_date(""))
        self.assertIsNone(rag_ret._normalize_iso_date("   "))

        self.assertEqual(rag_ret._normalize_iso_date(1577836800.0), "2020-01-01")
        self.assertIsNone(rag_ret._normalize_iso_date(999999999999.0))

        self.assertEqual(rag_ret._normalize_iso_date("2020-08-27"), "2020-08-27")
        self.assertEqual(rag_ret._normalize_iso_date("2020-08"), "2020-08-01")
        self.assertEqual(rag_ret._normalize_iso_date("2020"), "2020-01-01")

        self.assertIsNone(rag_ret._normalize_iso_date("not-a-date"))
        self.assertIsNone(rag_ret._normalize_iso_date("2020-invalid-day"))
        self.assertIsNone(rag_ret._normalize_iso_date("2020-08-27-01"))

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_mmr_trigger(self, mock_db_chunks, mock_qdrant_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        
        mock_client = mock_qdrant_client.return_value
        p1 = MagicMock()
        p1.id = "p1"
        p1.score = 0.9
        p1.payload = {}
        
        p2 = MagicMock()
        p2.id = "p2"
        p2.score = 0.8
        p2.payload = {}

        mock_client.search.return_value = [p1, p2]
        mock_db_chunks.return_value = [
            {"qdrant_point_id": "p1", "text": "shoulder"},
            {"qdrant_point_id": "p2", "text": "knee"},
        ]

        # top_k = 1, so len(results) > top_k triggers MMR re-ranking
        res = rag_ret.search_similar(
            query="test query",
            top_k=1,
            use_reranker=False
        )
        self.assertEqual(len(res), 1)

    @patch("rag.db.get_corpus_stats")
    def test_get_available_filters_success(self, mock_db_stats):
        mock_db_stats.return_value = {
            "indexed_runs": 2,
            "indexed_documents": 5,
            "total_chunks": 10,
            "unique_authors": 2,
            "earliest_date": datetime.date(2020, 1, 1),
            "latest_date": datetime.date(2020, 12, 31)
        }
        filters = rag_ret.get_available_filters()
        self.assertEqual(filters["indexed_runs"], 2)
        self.assertEqual(filters["date_range"]["earliest"], "2020-01-01")

    @patch("rag.db.get_corpus_stats")
    def test_get_available_filters_exception(self, mock_db_stats):
        mock_db_stats.side_effect = Exception("DB connection failed")
        filters = rag_ret.get_available_filters()
        self.assertEqual(filters["indexed_runs"], 0)

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    def test_search_similar_no_results(self, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []
        
        res = rag_ret.search_similar("query")
        self.assertEqual(res, [])

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_db_enrich_warning(self, mock_db_chunks, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_result = MagicMock()
        mock_result.id = "point1"
        mock_client.search.return_value = [mock_result]
        
        # Force DB exception inside enrichment block
        mock_db_chunks.side_effect = Exception("DB timeout")
        
        res = rag_ret.search_similar("query")
        # Should not raise exception (caught internally), but still return enriched payload fallback
        self.assertEqual(len(res), 1)

    def test_format_context_for_llm_empty(self):
        res = rag_ret.format_context_for_llm([])
        self.assertEqual(res, "No relevant document excerpts found.")

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    def test_search_similar_date_from_only(self, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []
        
        res = rag_ret.search_similar("query", date_from=1000.0)
        self.assertEqual(res, [])

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    def test_search_similar_date_to_only(self, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []
        
        res = rag_ret.search_similar("query", date_to=2000.0)
        self.assertEqual(res, [])

    def test_mmr_rerank_low_scores(self):
        # 200->187: test when candidate mmr_score is not better than best_score
        results = [
            {"score": 0.9, "text": "high relevance"},
            {"score": 0.1, "text": "low relevance 1"},
            {"score": 0.1, "text": "low relevance 2"}
        ]
        # First item (high relevance) is picked first.
        # Candidate 2 has same low text similarity and low score as Candidate 1,
        # so when Candidate 2 is checked, its score will not be strictly greater than Candidate 1's score.
        reranked = rag_ret._mmr_rerank(results, [0.1]*10, top_k=2)
        self.assertEqual(len(reranked), 2)

        # 204->183: test best_candidate is None branch (by setting candidate scores to -inf)
        results_inf = [
            {"score": 0.9, "text": "high relevance"},
            {"score": -float("inf"), "text": "low relevance 1"},
            {"score": -float("inf"), "text": "low relevance 2"}
        ]
        reranked_inf = rag_ret._mmr_rerank(results_inf, [0.1]*10, top_k=2)
        self.assertEqual(len(reranked_inf), 1)

    def test_format_context_for_llm_with_various_metadata(self):
        # Case A: all metadata present
        results_all = [{
            "text": "sample text",
            "original_filename": "doc.pdf",
            "page_number": 3,
            "page_start": 3,
            "page_end": 3,
            "provenance_type": "original_pdf",
            "author": "Dr Ek",
            "date_extracted": "2026-07-11",
            "document_type": "specialist_letter"
        }]
        res_all = rag_ret.format_context_for_llm(results_all)
        self.assertIn("File: doc.pdf", res_all)
        self.assertIn("Page: 3", res_all)
        self.assertIn("Author: Dr Ek", res_all)
        self.assertIn("Date: 2026-07-11", res_all)
        self.assertIn("Type: Specialist Letter", res_all)

        # Case B: all metadata missing or empty
        results_none = [{
            "text": "sample text",
            "original_filename": "",
            "page_number": None,
            "author": None,
            "date_extracted": None,
            "document_type": "unknown"
        }]
        res_none = rag_ret.format_context_for_llm(results_none)
        self.assertIn("PDF page provenance: not present in source metadata", res_none)
        self.assertTrue(res_none.strip().endswith("sample text"))

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    def test_search_similar_with_date_strings(self, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []
        
        res = rag_ret.search_similar("query", date_from="1971-11-28", date_to="2020-08-27")
        self.assertEqual(res, [])

    @patch("sentence_transformers.CrossEncoder")
    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_with_reranker(self, mock_get_chunks, mock_qdrant_client, mock_encode, mock_cross_encoder_class):
        mock_encode.return_value = [0.1, 0.2]
        
        # Configure DB chunks return
        mock_get_chunks.return_value = [
            {"qdrant_point_id": "point1", "chunk_id": "c1", "text": "shoulder pain acute"},
            {"qdrant_point_id": "point2", "chunk_id": "c2", "text": "knee replacement surgery"}
        ]
        
        # Configure Qdrant query response
        mock_client = mock_qdrant_client.return_value
        mock_res1 = MagicMock()
        mock_res1.id = "point1"
        mock_res1.score = 0.9
        mock_res1.payload = {"chunk_id": "c1", "text_preview": "shoulder pain acute"}
        
        mock_res2 = MagicMock()
        mock_res2.id = "point2"
        mock_res2.score = 0.8
        mock_res2.payload = {"chunk_id": "c2", "text_preview": "knee replacement surgery"}
        
        mock_client.search.return_value = [mock_res1, mock_res2]
        
        # Configure CrossEncoder predict returning raw logits
        mock_encoder = mock_cross_encoder_class.return_value
        mock_encoder.predict.return_value = [0.0, 2.0]
        
        # Reset cached reranker model references
        import rag.embedding as rag_emb
        rag_emb._reranker_model = None
        rag_emb._reranker_model_name = None
        
        # Run search similar with use_reranker=True
        res = rag_ret.search_similar(
            query="knee surgery",
            top_k=2,
            use_reranker=True,
            reranker_device="cpu"
        )
        # Should rerank and return knee surgery first due to higher logit score
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["chunk_id"], "c2")
        self.assertEqual(res[1]["chunk_id"], "c1")

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("settings_manager.load_settings")
    def test_search_similar_settings_fallback(self, mock_load_settings, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_load_settings.return_value = {
            "use_reranker": False,
            "reranker_model": "some-reranker",
            "reranker_device": "cpu"
        }
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []
        
        # Omit reranker args to trigger settings lookup
        res = rag_ret.search_similar("query", use_reranker=None, reranker_model=None, reranker_device=None)
        self.assertEqual(res, [])
        mock_load_settings.assert_called()

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.retriever.init_collection")
    def test_search_similar_qdrant_exceptions(self, mock_init, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value

        # Scenario 1: Qdrant search raises "doesn't exist" -> triggers init_collection -> retry search succeeds
        mock_client.search.side_effect = [
            Exception("Collection doesn't exist"),
            []
        ]
        res1 = rag_ret.search_similar("query", use_reranker=False)
        self.assertEqual(res1, [])
        mock_init.assert_called_once()

        # Scenario 2: Qdrant search raises "doesn't exist" -> triggers init_collection -> retry search fails
        mock_init.reset_mock()
        mock_client.search.side_effect = [
            Exception("Collection doesn't exist"),
            Exception("Search still fails")
        ]
        res2 = rag_ret.search_similar("query", use_reranker=False)
        self.assertEqual(res2, [])
        mock_init.assert_called_once()

        # Scenario 3: Qdrant search raises other Exception (like Connection Refused) -> returns empty list immediately
        mock_init.reset_mock()
        mock_client.search.side_effect = Exception("Connection Refused")
        res3 = rag_ret.search_similar("query", use_reranker=False)
        self.assertEqual(res3, [])
        mock_init.assert_not_called()

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    def test_search_similar_empty_results_with_reranker(self, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = []

        # Reranker is True, but search returns empty list
        res = rag_ret.search_similar("query", use_reranker=True)
        self.assertEqual(res, [])

    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.embedding.load_reranker_model")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_reranker_exception(self, mock_get_chunks, mock_load_reranker, mock_get_client, mock_encode):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_result = MagicMock()
        mock_result.id = "point1"
        mock_result.score = 0.8
        mock_result.payload = None
        mock_client.search.return_value = [mock_result]
        mock_get_chunks.return_value = [{"qdrant_point_id": "point1", "chunk_id": "c1", "text": "shoulder pain"}]

        # Reranker loading raises Exception
        mock_load_reranker.side_effect = Exception("Reranker loading failed")

        res = rag_ret.search_similar("query", use_reranker=True)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "c1")
        self.assertEqual(res[0]["score"], 0.8)

    def test_mmr_rerank_with_empty_texts(self):
        results = [
            {"score": 0.9, "text": ""},
            {"score": 0.8, "text": "shoulder"},
            {"score": 0.7, "text": ""}
        ]
        reranked = rag_ret._mmr_rerank(results, [0.1]*10, top_k=2)
        self.assertEqual(len(reranked), 2)

    @patch("rag.db.get_chunks_by_qdrant_ids")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.retriever.encode_query")
    def test_search_similar_no_reranker_success(self, mock_encode, mock_get_client, mock_get_chunks):
        mock_encode.return_value = [0.1, 0.2]
        mock_client = mock_get_client.return_value
        mock_res = MagicMock()
        mock_res.id = "point1"
        mock_res.score = 0.85
        mock_res.payload = {"chunk_id": "c1"}
        mock_client.search.return_value = [mock_res]
        mock_get_chunks.return_value = [{"qdrant_point_id": "point1", "chunk_id": "c1", "text": "my text"}]

        progress_calls = []
        def progress_callback(pct, msg):
            progress_calls.append((pct, msg))

        res = rag_ret.search_similar(
            "query",
            use_reranker=False,
            top_k=2,
            progress_callback=progress_callback
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["text"], "my text")
        self.assertTrue(len(progress_calls) > 0)

    @patch("sentence_transformers.CrossEncoder")
    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_with_reranker_and_progress_callback(self, mock_get_chunks, mock_qdrant_client, mock_encode, mock_cross_encoder_class):
        mock_encode.return_value = [0.1, 0.2]
        mock_get_chunks.return_value = [
            {"qdrant_point_id": "point1", "chunk_id": "c1", "text": "shoulder pain acute"}
        ]
        mock_client = mock_qdrant_client.return_value
        mock_res1 = MagicMock()
        mock_res1.id = "point1"
        mock_res1.score = 0.9
        mock_res1.payload = {"chunk_id": "c1", "text_preview": "shoulder pain"}
        mock_client.search.return_value = [mock_res1]
        
        mock_encoder = mock_cross_encoder_class.return_value
        mock_encoder.predict.return_value = [1.0]

        import rag.embedding as rag_emb
        rag_emb._reranker_model = None
        rag_emb._reranker_model_name = None

        progress_calls = []
        def progress_cb(pct, msg):
            progress_calls.append((pct, msg))

        res = rag_ret.search_similar(
            query="knee surgery",
            top_k=2,
            use_reranker=True,
            reranker_device="cpu",
            progress_callback=progress_cb
        )
        self.assertEqual(len(res), 1)
        self.assertTrue(any(pct == 0.4 for pct, _ in progress_calls))
        self.assertTrue(any(pct == 0.8 for pct, _ in progress_calls))

    @patch("sentence_transformers.CrossEncoder")
    @patch("rag.retriever.encode_query")
    @patch("rag.retriever.get_qdrant_client")
    @patch("rag.db.get_chunks_by_qdrant_ids")
    def test_search_similar_reranker_exception_with_progress_callback(self, mock_get_chunks, mock_qdrant_client, mock_encode, mock_cross_encoder_class):
        mock_encode.return_value = [0.1, 0.2]
        mock_get_chunks.return_value = [
            {"qdrant_point_id": "point1", "chunk_id": "c1", "text": "shoulder pain"}
        ]
        mock_client = mock_qdrant_client.return_value
        mock_res1 = MagicMock()
        mock_res1.id = "point1"
        mock_res1.score = 0.9
        mock_client.search.return_value = [mock_res1]
        
        mock_encoder = mock_cross_encoder_class.return_value
        mock_encoder.predict.side_effect = Exception("Predict failed")

        import rag.embedding as rag_emb
        rag_emb._reranker_model = None

        progress_calls = []
        def progress_cb(pct, msg):
            progress_calls.append((pct, msg))

        res = rag_ret.search_similar(
            query="knee surgery",
            top_k=2,
            use_reranker=True,
            reranker_device="cpu",
            progress_callback=progress_cb
        )
        self.assertEqual(len(res), 1)
        self.assertTrue(any("failed" in msg.lower() for _, msg in progress_calls))


if __name__ == "__main__":
    unittest.main()
