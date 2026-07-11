"""
Comprehensive unit tests for rag/analyzer.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import httpx

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag.analyzer as rag_anz


class MockStreamResponse:
    def __init__(self, status_code, lines):
        self.status_code = status_code
        self._lines = lines

    def iter_lines(self):
        return self._lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestRAGAnalyzerAll(unittest.TestCase):

    def test_get_analysis_modes(self):
        modes = rag_anz.get_analysis_modes()
        self.assertIn("free_qa", modes)
        self.assertIn("timeline", modes)

    def test_build_prompt_basic(self):
        # 1. Basic build without history
        msgs = rag_anz.build_prompt("my query", "my context", "free_qa")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("my context", msgs[1]["content"])

    def test_build_prompt_with_history(self):
        # 2. Build with long chat history (> 6 messages)
        history = [
            {"role": "user", "content": f"q{i}"} for i in range(10)
        ]
        msgs = rag_anz.build_prompt("my query", "my context", "timeline", history)
        # Should keep system + last 6 messages + new user message = 8 messages
        self.assertEqual(len(msgs), 8)
        self.assertEqual(msgs[1]["content"], "q4")
        self.assertEqual(msgs[-1]["role"], "user")

    @patch("httpx.stream")
    def test_query_llm_streaming_success(self, mock_stream):
        lines = [
            "data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}",
            "",
            "data: [DONE]"
        ]
        mock_stream.return_value = MockStreamResponse(200, lines)
        
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertEqual(res, "hello")

    @patch("httpx.stream")
    def test_query_llm_streaming_non_200(self, mock_stream):
        mock_stream.return_value = MockStreamResponse(500, [])
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertIn("Error", res)
        self.assertIn("HTTP 500", res)

    @patch("httpx.stream")
    def test_query_llm_streaming_invalid_json(self, mock_stream):
        lines = [
            "data: {invalid json}",
            "data: [DONE]"
        ]
        mock_stream.return_value = MockStreamResponse(200, lines)
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertEqual(res, "")

    @patch("httpx.stream")
    def test_query_llm_streaming_connection_error(self, mock_stream):
        mock_stream.side_effect = httpx.ConnectError("Failed to connect")
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertIn("Cannot connect", res)

    @patch("httpx.stream")
    def test_query_llm_streaming_timeout_error(self, mock_stream):
        mock_stream.side_effect = httpx.ReadTimeout("Timeout")
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertIn("timed out", res)

    @patch("httpx.stream")
    def test_query_llm_streaming_generic_error(self, mock_stream):
        mock_stream.side_effect = Exception("Generic crash")
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4")
        res = "".join(list(gen))
        self.assertIn("Generic crash", res)

    @patch("httpx.post")
    def test_query_llm_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "static answer"}}]
        }
        mock_post.return_value = mock_resp

        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")
        self.assertEqual(res, "static answer")

    @patch("httpx.post")
    def test_query_llm_empty_choices(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}
        mock_post.return_value = mock_resp

        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")
        self.assertEqual(res, "No response generated.")

    @patch("httpx.post")
    def test_query_llm_non_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")
        self.assertIn("HTTP 404", res)

    @patch("httpx.post")
    def test_query_llm_connect_error(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")
        self.assertIn("Cannot connect", res)

    @patch("httpx.post")
    def test_query_llm_generic_error(self, mock_post):
        mock_post.side_effect = Exception("Crash")
        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")
        self.assertIn("Crash", res)

    @patch("rag.analyzer.search_similar")
    def test_analyze_no_results(self, mock_search):
        mock_search.return_value = []
        gen = rag_anz.analyze("query")
        res = "".join(list(gen))
        self.assertIn("No relevant document excerpts found", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.format_context_for_llm")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("rag.analyzer.query_llm")
    def test_analyze_routing(self, mock_query_llm, mock_stream_llm, mock_format, mock_search):
        mock_search.return_value = [{"chunk_id": "c1"}]
        mock_format.return_value = "formatted context"
        
        # 1. Streaming route
        mock_stream_llm.return_value = iter(["stream chunk"])
        res1 = "".join(list(rag_anz.analyze("query", stream=True)))
        self.assertEqual(res1, "stream chunk")

        # 2. Non-streaming route
        mock_query_llm.return_value = "non-stream result"
        res2 = "".join(list(rag_anz.analyze("query", stream=False)))
        self.assertEqual(res2, "non-stream result")

    @patch("httpx.Client.send")
    def test_query_llm_streaming_edge_branches(self, mock_send):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "part1"}}]}',
            'data: {"choices": []}', # 184->173 branch (empty choices)
            'data: {"choices": [{"delta": {"content": ""}}]}', # 187->173 branch (empty content)
            'data: {"choices": [{"delta": {"content": "part2"}}]}'
            # 173->exit branch: no "[DONE]" line at the end, exits normally
        ]
        mock_send.return_value = mock_response

        res = list(rag_anz.query_llm_streaming("prompt", "http://localhost:8000/v1", "phi-4"))
        self.assertEqual(res, ["part1", "part2"])


if __name__ == "__main__":
    unittest.main()
