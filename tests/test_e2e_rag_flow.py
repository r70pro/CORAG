"""
End-to-End RAG Integration Test.

Verifies full query -> retrieval -> LLM prompt assembly -> response & citation generation flow,
ensuring strict compliance with medicolegal document citation rules.
"""

import unittest
from unittest.mock import MagicMock, patch

from rag import analyzer as rag_analyzer
from rag import retriever as rag_retriever


class TestE2ERagIntegrationFlow(unittest.TestCase):
    def setUp(self):
        self.mock_chunks = [
            {
                "chunk_id": "chunk_e2e_001",
                "doc_id": "doc_e2e_001",
                "run_id": "run_e2e_001",
                "chunk_index": 0,
                "original_filename": "source_record.pdf",
                "text": "Dr. Gavin Weekes (Orthopaedic Surgeon) examined John Doe on 2024-03-15 for right shoulder tear. Ref No: 2024AL0008570-1.",
                "page_number": 4,
                "page_start": 4,
                "page_end": 4,
                "provenance_type": "original_pdf",
                "document_type": "Specialist Correspondence",
                "author": "Dr. Gavin Weekes",
                "date_extracted": "2024-03-15",
                "section_type": "Specialist Examination",
                "score": 0.95,
            },
            {
                "chunk_id": "chunk_e2e_002",
                "doc_id": "doc_e2e_001",
                "run_id": "run_e2e_001",
                "chunk_index": 1,
                "original_filename": "source_record.pdf",
                "text": "Operation Record: Arthroscopic rotator cuff repair performed at St Vincent's Hospital on 2024-04-02 by Dr. Gavin Weekes. Accession Number: 77.50382801.",
                "page_number": 8,
                "page_start": 8,
                "page_end": 8,
                "provenance_type": "original_pdf",
                "document_type": "Operation Record",
                "author": "Dr. Gavin Weekes",
                "date_extracted": "2024-04-02",
                "section_type": "Surgical Procedure",
                "score": 0.89,
            },
        ]

    def test_context_formatting_includes_verification_details(self):
        """Verify context formatter includes PDF page ranges, doc types, authors, and report IDs."""
        formatted_context = rag_retriever.format_context_for_llm(self.mock_chunks)

        self.assertIn("Page: 4", formatted_context)
        self.assertIn("Author: Dr. Gavin Weekes", formatted_context)
        self.assertIn("Type: Specialist Correspondence", formatted_context)
        self.assertIn("Ref No: 2024AL0008570-1", formatted_context)
        self.assertIn("Page: 8", formatted_context)
        self.assertIn("Type: Operation Record", formatted_context)
        self.assertIn("Accession Number: 77.50382801", formatted_context)

    @patch("httpx.post")
    @patch("rag.analyzer.search_similar")
    def test_full_rag_query_flow_non_streaming(self, mock_search, mock_httpx_post):
        """Test full query -> search -> LLM prompt assembly -> completion response flow."""
        mock_search.return_value = self.mock_chunks

        # Mock LLM HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "On 2024-03-15, John Doe was examined for a right shoulder tear by Dr. Gavin Weekes "
                            "(Specialist Correspondence, Dr. Gavin Weekes, Ref No: 2024AL0008570-1, PDF Page 4). "
                            "On 2024-04-02, an arthroscopic rotator cuff repair was performed (Operation Record, "
                            "Dr. Gavin Weekes, Accession Number: 77.50382801, PDF Page 8)."
                        )
                    }
                }
            ]
        }
        mock_httpx_post.return_value = mock_response

        # Execute RAG analysis
        with patch("rag.analyzer._resolve_loaded_model", return_value=("test-model", False)):
            response_generator = rag_analyzer.analyze(
                query="What shoulder procedures and consultations were performed for John Doe?",
                mode="free_qa",
                stream=False,
                server_url="http://127.0.0.1:8000/v1",
                model_name="test-model",
            )
            response_text = "".join(list(response_generator))

        # Assert full flow results
        # Interactive Free Q&A uses the primary query plus two bounded evidence
        # facets, all with identical case/metadata filters.
        self.assertEqual(mock_search.call_count, 3)
        self.assertIn("Dr. Gavin Weekes", response_text)
        self.assertIn("Specialist Correspondence", response_text)
        self.assertIn("Operation Record", response_text)
        self.assertIn("PDF Page 4", response_text)
        self.assertIn("PDF Page 8", response_text)

        # Verify no raw system source tags (like [Source 26]) are present in final LLM response
        self.assertNotIn("[Source ", response_text)

    @patch("httpx.stream")
    @patch("rag.analyzer.search_similar")
    def test_full_rag_query_flow_streaming(self, mock_search, mock_httpx_stream):
        """Test full query -> search -> streaming LLM token generation flow."""
        mock_search.return_value = self.mock_chunks

        # Mock SSE stream lines
        mock_sse_lines = [
            'data: {"choices": [{"delta": {"content": "According to "}}]}\n',
            'data: {"choices": [{"delta": {"content": "Operation Record by Dr. Gavin Weekes (PDF Page 8, Accession Number: 77.50382801), "}}]}\n',
            'data: {"choices": [{"delta": {"content": "the surgery occurred on 2024-04-02."}}]}\n',
            'data: [DONE]\n',
        ]

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.status_code = 200
        mock_stream_ctx.iter_lines.return_value = mock_sse_lines
        mock_httpx_stream.return_value.__enter__.return_value = mock_stream_ctx

        # Execute RAG streaming query
        with patch("rag.analyzer._resolve_loaded_model", return_value=("test-model", False)):
            chunks = list(
                rag_analyzer.analyze(
                    query="When was the surgery performed?",
                    mode="free_qa",
                    stream=True,
                    server_url="http://127.0.0.1:8000/v1",
                    model_name="test-model",
                )
            )

        full_stream_text = "".join(chunks)
        self.assertIn("Operation Record by Dr. Gavin Weekes", full_stream_text)
        self.assertIn("PDF Page 8", full_stream_text)
        self.assertNotIn("[Source ", full_stream_text)


if __name__ == "__main__":
    unittest.main()
