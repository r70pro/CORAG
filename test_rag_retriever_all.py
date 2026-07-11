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

    def test_text_similarity(self):
        # 1. Matching
        self.assertAlmostEqual(rag_ret._text_similarity("shoulder pain", "shoulder pain"), 1.0)
        
        # 2. Part matching
        self.assertAlmostEqual(rag_ret._text_similarity("shoulder pain", "knee pain"), 0.3333333, places=5)
        
        # 3. Empty strings
        self.assertEqual(rag_ret._text_similarity("", "shoulder"), 0.0)
        self.assertEqual(rag_ret._text_similarity("shoulder", ""), 0.0)

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
    def test_search_similar_exception(self, mock_encode):
        mock_encode.side_effect = Exception("Embed generation failed")
        
        # Exception should bubble up
        with self.assertRaises(Exception):
            rag_ret.search_similar("query")

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
        self.assertEqual(res_none.strip(), "[Source 1]\nsample text")


if __name__ == "__main__":
    unittest.main()
