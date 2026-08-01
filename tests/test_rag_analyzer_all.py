"""
Comprehensive unit tests for rag/analyzer.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

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
        self.assertEqual(
            set(modes),
            {
                "free_qa",
                "timeline",
                "injury_summary",
                "inconsistency_finder",
                "medication_tracker",
            },
        )

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
    def test_query_llm_marks_output_limit_truncation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"content": "partial table row"},
                    "finish_reason": "length",
                }
            ]
        }
        mock_post.return_value = mock_resp

        res = rag_anz.query_llm([], "http://localhost:8000/v1", "phi-4")

        self.assertIn("partial table row", res)
        self.assertIn("Incomplete response", res)
        self.assertIn("Maximum Output Tokens", res)

    @patch("httpx.stream")
    def test_query_llm_streaming_marks_output_limit_truncation(self, mock_stream):
        mock_stream.return_value = MockStreamResponse(
            200,
            [
                'data: {"choices": [{"delta": {"content": "partial"}}]}',
                'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}',
                "data: [DONE]",
            ],
        )

        res = "".join(rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "phi-4"))

        self.assertIn("partial", res)
        self.assertIn("Incomplete response", res)

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

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.format_context_for_llm")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("httpx.get")
    def test_analyze_model_fallback(self, mock_get, mock_stream_llm, mock_format, mock_search):
        mock_search.return_value = [{"chunk_id": "c1"}]
        mock_format.return_value = "formatted context"
        mock_stream_llm.return_value = iter(["answer"])

        # Mock /models response showing that "phi-4" is not loaded, but "olmocr" is.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "allenai/olmOCR-2-7B-1025-FP8"}]
        }
        mock_get.return_value = mock_response

        # Temporarily disable TESTING env var to allow network call execution
        with patch.dict(os.environ, {"TESTING": "false"}):
            gen = rag_anz.analyze("query", model_name="microsoft/Phi-4-reasoning-plus", stream=True)
            res = list(gen)

            # Check that it warns the user and falls back
            self.assertTrue(any("not loaded in vLLM. Falling back to" in item for item in res))
            # Verify stream called with resolved model
            mock_stream_llm.assert_called_once()
            self.assertEqual(mock_stream_llm.call_args[0][2], "allenai/olmOCR-2-7B-1025-FP8")

    @patch("httpx.stream")
    def test_query_llm_streaming_reasoning_model(self, mock_stream):
        lines = [
            "data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}",
            "data: [DONE]"
        ]
        mock_stream.return_value = MockStreamResponse(200, lines)

        # Using a model name that contains "reasoning"
        gen = rag_anz.query_llm_streaming([], "http://localhost:8000/v1", "Phi-4-reasoning-plus")
        res = "".join(list(gen))
        self.assertEqual(res, "hello")

        # Verify repetition_penalty was added to payload
        mock_stream.assert_called_once()
        payload = mock_stream.call_args[1]["json"]
        self.assertEqual(payload["repetition_penalty"], 1.05)

    @patch("httpx.post")
    def test_query_llm_reasoning_model(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "static answer"}}]
        }
        mock_post.return_value = mock_resp

        # Using a model name containing "reasoning"
        res = rag_anz.query_llm([], "http://localhost:8000/v1", "Phi-4-reasoning-plus")
        self.assertEqual(res, "static answer")

        # Verify repetition_penalty was added to payload
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["repetition_penalty"], 1.05)

    @patch("rag.analyzer.query_llm", return_value="safe answer")
    def test_query_llm_streaming_uses_safe_qwen3_content_envelope(self, mock_query):
        chunks = list(
            rag_anz.query_llm_streaming(
                [], "http://localhost:8000/v1", "Qwen/Qwen3.6-35B-A3B"
            )
        )

        self.assertEqual(chunks, ["safe answer"])
        mock_query.assert_called_once_with(
            [],
            "http://localhost:8000/v1",
            "Qwen/Qwen3.6-35B-A3B",
            temperature=0.1,
            max_tokens=16000,
        )

    @patch("httpx.post")
    def test_query_llm_disables_qwen3_thinking(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }
        mock_post.return_value = mock_response

        self.assertEqual(
            rag_anz.query_llm(
                [], "http://localhost:8000/v1", "Qwen/Qwen3.6-35B-A3B"
            ),
            "answer",
        )
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.format_context_for_llm")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("httpx.get")
    def test_analyze_preflight_variants(self, mock_get, mock_stream_llm, mock_format, mock_search):
        mock_search.return_value = [{"chunk_id": "c1"}]
        mock_format.return_value = "formatted context"
        mock_stream_llm.return_value = iter(["answer"])

        # Variant 1: model_name in loaded_models
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "my_model"}]
        }
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"TESTING": "false"}):
            list(rag_anz.analyze("query", model_name="my_model", stream=True))
            mock_stream_llm.assert_called_once()
            self.assertEqual(mock_stream_llm.call_args[0][2], "my_model")
            mock_stream_llm.reset_mock()

        # Variant 2: model_name is equivalent to loaded model
        # Equivalents defined: equivalents = {"microsoft/Phi-4-reasoning-plus": "nvidia/Phi-4-reasoning-plus-NVFP4"}
        # Case A: request microsoft/Phi-4-reasoning-plus, loaded nvidia/Phi-4-reasoning-plus-NVFP4
        mock_response.json.return_value = {
            "data": [{"id": "nvidia/Phi-4-reasoning-plus-NVFP4"}]
        }
        with patch.dict(os.environ, {"TESTING": "false"}):
            list(rag_anz.analyze("query", model_name="microsoft/Phi-4-reasoning-plus", stream=True))
            # Should resolve to equivalent "nvidia/Phi-4-reasoning-plus-NVFP4"
            self.assertEqual(mock_stream_llm.call_args[0][2], "nvidia/Phi-4-reasoning-plus-NVFP4")

        # Case B: request nvidia/Phi-4-reasoning-plus-NVFP4, loaded microsoft/Phi-4-reasoning-plus
        mock_response.json.return_value = {
            "data": [{"id": "microsoft/Phi-4-reasoning-plus"}]
        }
        with patch.dict(os.environ, {"TESTING": "false"}):
            list(rag_anz.analyze("query", model_name="nvidia/Phi-4-reasoning-plus-NVFP4", stream=True))
            self.assertEqual(mock_stream_llm.call_args[0][2], "microsoft/Phi-4-reasoning-plus")

        # Variant 3: json decoding exception
        mock_response.json.side_effect = Exception("Malformed JSON")
        with patch.dict(os.environ, {"TESTING": "false"}):
            list(rag_anz.analyze("query", model_name="my_model", stream=True))
            # Falls back to query model as resolves exception
            self.assertEqual(mock_stream_llm.call_args[0][2], "my_model")

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    def test_analyze_context_truncation_streaming(self, mock_load, mock_stream, mock_search):
        # OCR container length is intentionally irrelevant to analysis budgeting.
        mock_load.return_value = {
            "docker_max_model_len": 5200,
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        }

        # Setup 3 chunks of results, each long enough to exceed the 2048 prompt limit
        mock_search.return_value = [
            {"text": "very long text " * 4000, "chunk_id": "c1"},
            {"text": "very long text " * 4000, "chunk_id": "c2"},
            {"text": "very long text " * 4000, "chunk_id": "c3"},
        ]
        mock_stream.return_value = iter(["answer"])

        # Execute analyze
        gen = rag_anz.analyze("query", stream=True)
        res = list(gen)

        # Verify it yielded a warning message
        self.assertTrue(any("too large for the model's context window" in item for item in res))
        self.assertIn("answer", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm")
    @patch("settings_manager.load_settings")
    def test_analyze_context_truncation_non_streaming(self, mock_load, mock_query, mock_search):
        mock_load.return_value = {
            "docker_max_model_len": 5200,
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        }

        mock_search.return_value = [
            {"text": "very long text " * 4000, "chunk_id": "c1"},
            {"text": "very long text " * 4000, "chunk_id": "c2"},
            {"text": "very long text " * 4000, "chunk_id": "c3"},
        ]
        mock_query.return_value = "answer"

        # Execute analyze
        gen = rag_anz.analyze("query", stream=False)
        res = list(gen)

        # Yields a single item with warning + answer
        self.assertEqual(len(res), 1)
        self.assertTrue("too large for the model's context window" in res[0])
        self.assertTrue("answer" in res[0])

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    @patch("tiktoken.get_encoding")
    def test_analyze_context_truncation_fallback(self, mock_get_enc, mock_load, mock_stream, mock_search):
        mock_get_enc.side_effect = Exception("No internet or file not cached")
        mock_load.return_value = {
            "docker_max_model_len": 5200,
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        }

        mock_search.return_value = [
            {"text": "very long text " * 4000, "chunk_id": "c1"},
            {"text": "very long text " * 4000, "chunk_id": "c2"},
            {"text": "very long text " * 4000, "chunk_id": "c3"},
        ]
        mock_stream.return_value = iter(["answer"])

        gen = rag_anz.analyze("query", stream=True)
        res = list(gen)

        self.assertTrue(any("too large for the model's context window" in item for item in res))
        self.assertIn("answer", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    def test_analyze_token_estimation_branches_and_name_key(self, mock_load, mock_stream, mock_search):
        mock_load.return_value = {"docker_max_model_len": 131072}
        mock_search.return_value = [{"text": "context fragment", "chunk_id": "c1"}]
        mock_stream.side_effect = lambda *args, **kwargs: iter(["answer"])

        chat_hist = [
            {"role": "user", "content": [{"text": "list text"}, "plain text string"], "name": "tester"}
        ]

        # 1. With tiktoken working
        gen = rag_anz.analyze("query", chat_history=chat_hist, run_id_filter="run1", stream=True)
        res = list(gen)
        self.assertIn("answer", res)

        # 2. With tiktoken failing (fallback token estimation)
        with patch("tiktoken.get_encoding", side_effect=Exception("Disabled")):
            gen = rag_anz.analyze("query", chat_history=chat_hist, run_id_filter="run1", stream=True)
            res = list(gen)
            self.assertIn("answer", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("httpx.get")
    def test_analyze_preflight_server_not_200_or_no_models(self, mock_get, mock_stream_llm, mock_search):
        mock_search.return_value = [{"text": "ok", "chunk_id": "c1"}]
        mock_stream_llm.side_effect = lambda *args, **kwargs: iter(["answer"])

        # Case A: status_code is 500
        mock_get.return_value = MagicMock(status_code=500)
        with patch.dict(os.environ, {"TESTING": "false"}):
            res = list(rag_anz.analyze("query", stream=True))
            self.assertIn("answer", res)

        # Case B: status_code is 200 but data is empty
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp
        with patch.dict(os.environ, {"TESTING": "false"}):
            res = list(rag_anz.analyze("query", stream=True))
            self.assertIn("answer", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    def test_analyze_context_truncation_breaks_early(self, mock_load, mock_stream, mock_search):
        mock_load.return_value = {
            "docker_max_model_len": 5200,
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        }
        mock_search.return_value = [
            {"text": "short", "chunk_id": "c1"},
            {"text": "short", "chunk_id": "c2"},
            {"text": "very long text " * 10000, "chunk_id": "c3"},
        ]
        mock_stream.return_value = iter(["answer"])

        gen = rag_anz.analyze("query", stream=True)
        res = list(gen)
        self.assertTrue(any("too large for the model's context window" in item for item in res))
        self.assertIn("answer", res)

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("httpx.get")
    def test_analyze_preflight_is_equivalent_m1_eq_m2(self, mock_get, mock_stream_llm, mock_search):
        mock_search.return_value = [{"text": "ok", "chunk_id": "c1"}]
        mock_stream_llm.return_value = iter(["answer"])

        class StatefulModelName:
            def __init__(self, name):
                self.name = name
                self.calls = 0
            def __eq__(self, other):
                self.calls += 1
                if self.calls == 1:
                    return False
                return self.name == other
            def __str__(self):
                return self.name
            def __hash__(self):
                return hash(self.name)

        stateful_name = StatefulModelName("my_model")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"data": [{"id": "my_model"}]}
        mock_get.return_value = mock_resp

        with patch.dict(os.environ, {"TESTING": "false"}):
            res = list(rag_anz.analyze("query", model_name=stateful_name, stream=True))
            self.assertIn("answer", res)

    def test_source_tag_replacements(self):
        results = [
            {
                "original_filename": "souki_enclosures.pdf",
                "page_number": 37,
                "page_start": 37,
                "page_end": 37,
                "provenance_type": "original_pdf",
                "author": "Dr. Gavin Weekes",
                "date_extracted": "2021-02-14",
                "document_type": "specialist_correspondence",
                "text": "Ref No: 2021AL0008570-1 some other info"
            },
            {
                "original_filename": "medical_report.pdf",
                "page_number": 6,
                "page_start": 6,
                "page_end": 6,
                "provenance_type": "original_pdf",
                "author": "",
                "date_extracted": "",
                "document_type": "unknown",
                "text": "plain text chunk"
            }
        ]

        # 1. Bracketed single tag replacement
        input1 = "The patient denied pain [Source 1]."
        output1 = rag_anz.replace_source_tags_in_string(input1, results)
        self.assertIn("Dr. Gavin Weekes", output1)
        self.assertIn("Specialist Correspondence", output1)
        self.assertIn("2021-02-14", output1)
        self.assertIn("p. 37", output1)
        self.assertIn("Ref No: 2021AL0008570-1", output1)
        self.assertIn("souki_enclosures.pdf", output1)

        # 2. Minimal info fallback (filename should be included)
        input2 = "Degenerative changes noted [Source 2]."
        output2 = rag_anz.replace_source_tags_in_string(input2, results)
        self.assertIn("medical_report.pdf", output2)
        self.assertIn("p. 6", output2)

        # 3. Multiple sources replacement
        input3 = "Issues mentioned in [Source 1, 2]."
        output3 = rag_anz.replace_source_tags_in_string(input3, results)
        self.assertIn("Dr. Gavin Weekes", output3)
        self.assertIn("medical_report.pdf", output3)

        # 4. Out of bounds index
        input4 = "Out of bounds [Source 3]."
        output4 = rag_anz.replace_source_tags_in_string(input4, results)
        self.assertEqual(output4, "Out of bounds (Source 3).")

        # 5. Streaming wrapper test
        def mock_generator():
            yield "Hello, see "
            yield "" # empty chunk (line 462)
            yield "[Source"
            yield " 1] and [So"
            yield "urce 2]."
            yield " Testing "
            yield "Source" # potential unbracketed prefix
            yield " 3"
            yield "[unclosed"
            yield "a" * 160 # len > 150 without close bracket (line 495-496)
            yield " leftover" # leftover buffer at the end (line 505)

        stream_out = "".join(list(rag_anz.replace_source_tags_streaming(mock_generator(), results)))
        self.assertIn("Dr. Gavin Weekes", stream_out)
        self.assertIn("medical_report.pdf", stream_out)
        self.assertNotIn("[Source", stream_out)
        self.assertIn("leftover", stream_out)

    def test_equivalence_helpers(self):
        # Test _is_equivalent (line 84)
        self.assertTrue(rag_anz._is_equivalent("microsoft/Phi-4-reasoning-plus", "nvidia/Phi-4-reasoning-plus-NVFP4"))
        self.assertTrue(rag_anz._is_equivalent("nvidia/Phi-4-reasoning-plus-NVFP4", "microsoft/Phi-4-reasoning-plus"))
        self.assertFalse(rag_anz._is_equivalent("other", "other2"))

        # Test _map_equivalent (lines 90-95)
        loaded = ["nvidia/Phi-4-reasoning-plus-NVFP4"]
        self.assertEqual(
            rag_anz._map_equivalent("microsoft/Phi-4-reasoning-plus", loaded),
            "nvidia/Phi-4-reasoning-plus-NVFP4"
        )
        self.assertEqual(
            rag_anz._map_equivalent("other", loaded),
            "other"
        )
        self.assertEqual(
            rag_anz._map_equivalent("nvidia/Phi-4-reasoning-plus-NVFP4", loaded),
            "nvidia/Phi-4-reasoning-plus-NVFP4"
        )

    @patch("httpx.get")
    def test_model_cache_with_testing_disabled(self, mock_get):
        # Configure httpx mock
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"data": [{"id": "model_x"}]}
        mock_get.return_value = mock_resp

        # Clear cache first
        rag_anz._model_cache.clear()

        # Temporarily remove TESTING env var
        with patch.dict(os.environ):
            os.environ.pop("TESTING", None)

            # First call: populates cache
            models1 = rag_anz._get_loaded_models("http://localhost:8000")
            self.assertEqual(models1, ["model_x"])
            self.assertEqual(mock_get.call_count, 1)

            # Second call: uses cached values (line 45)
            models2 = rag_anz._get_loaded_models("http://localhost:8000")
            self.assertEqual(models2, ["model_x"])
            self.assertEqual(mock_get.call_count, 1)

    @patch("re.findall", return_value=[])
    def test_citations_empty_indices(self, mock_findall):
        # Empty indices coverage (line 451)
        res = rag_anz.replace_source_tags_in_string("Source 1", [{"text": "hello"}])
        self.assertEqual(res, "Source 1")

    def test_replacer_group_none(self):
        # Capture and test nested replacer function when group(1) and group(2) are both None (line 439)
        mock_pattern = MagicMock()
        with patch("re.compile", return_value=mock_pattern):
            replacer_fn = None
            def mock_sub(replacer, text):
                nonlocal replacer_fn
                replacer_fn = replacer
                return "mocked_result"
            mock_pattern.sub.side_effect = mock_sub

            rag_anz.replace_source_tags_in_string("text", [])

            # Call replacer_fn directly with mock match
            mock_match = MagicMock()
            mock_match.group.side_effect = lambda idx: None if idx in (1, 2) else "matched_text"
            res = replacer_fn(mock_match)
            self.assertEqual(res, "matched_text")

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm")
    def test_analyze_structured_custom_threshold(self, mock_query, mock_search):
        # Call analyze in timeline mode with score_threshold in search_kwargs (line 553->557)
        mock_search.return_value = [{"text": "chunk1", "page_number": 2}]
        mock_query.return_value = "Response"

        res = list(rag_anz.analyze(
            query="test",
            mode="timeline",
            stream=False,
            score_threshold=0.1
        ))
        self.assertEqual(res, ["Response"])

    def test_analysis_context_length_precedence_excludes_ocr_settings(self):
        lengths = {
            "resolved-analysis": 32_768,
            "configured-analysis": 131_072,
            "ocr-model": 1_048_576,
        }

        self.assertEqual(
            rag_anz._analysis_context_length(
                "resolved-analysis", "configured-analysis", lengths
            ),
            32_768,
        )
        self.assertEqual(
            rag_anz._analysis_context_length(
                "unknown-resolved", "configured-analysis", lengths
            ),
            131_072,
        )
        self.assertEqual(
            rag_anz._analysis_context_length(
                "unknown-resolved", "unknown-configured", lengths
            ),
            rag_anz.CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH,
        )

    @patch("rag.analyzer.httpx.get")
    def test_served_model_context_length_is_authoritative(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {
                        "id": "Qwen/Qwen3.6-35B-A3B",
                        "max_model_len": 15_360,
                    }
                ]
            },
        )

        self.assertEqual(
            rag_anz._get_served_model_context_length(
                "http://localhost:8000/v1", "Qwen/Qwen3.6-35B-A3B"
            ),
            15_360,
        )

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    def test_ocr_context_setting_cannot_override_analysis_model(
        self, mock_load, mock_stream, mock_search
    ):
        mock_load.return_value = {
            "docker_max_model_len": 1_048_576,
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        }
        mock_search.return_value = [
            {"text": "very long text " * 4000, "chunk_id": f"c{index}"}
            for index in range(3)
        ]
        mock_stream.return_value = iter(["answer"])

        response = list(
            rag_anz.analyze(
                "query",
                model_name="nvidia/Phi-4-reasoning-plus-NVFP4",
                stream=True,
            )
        )

        self.assertTrue(
            any("too large for the model's context window" in item for item in response)
        )

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm_streaming")
    @patch("settings_manager.load_settings")
    def test_output_tokens_are_checked_and_forwarded(
        self, mock_load, mock_stream, mock_search
    ):
        mock_load.return_value = {
            "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4"
        }
        mock_search.return_value = [{"text": "short context", "chunk_id": "c1"}]
        mock_stream.return_value = iter(["answer"])

        self.assertEqual(
            list(rag_anz.analyze("query", max_tokens=1234, stream=True)),
            ["answer"],
        )
        self.assertEqual(mock_stream.call_args.kwargs["max_tokens"], 1234)

        with self.assertRaises(rag_anz.ContextWindowError):
            list(rag_anz.analyze("query", max_tokens=32_768, stream=True))

        with self.assertRaises(rag_anz.ContextWindowError):
            list(rag_anz.analyze("query", max_tokens=32_700, stream=True))

        # The served chat template adds special tokens after KIRAG's message
        # estimate. Requests must retain the explicit overhead reserve even
        # when the visible prompt and output would otherwise fit exactly.
        with self.assertRaisesRegex(
            rag_anz.ContextWindowError, "chat-template overhead"
        ):
            list(
                rag_anz.analyze(
                    "query",
                    max_tokens=(
                        rag_anz.CONSERVATIVE_ANALYSIS_CONTEXT_LENGTH
                        - rag_anz.CHAT_TEMPLATE_TOKEN_RESERVE
                    ),
                    stream=True,
                )
            )

    @patch("rag.analyzer.search_similar")
    @patch("rag.analyzer.query_llm")
    def test_estimate_tokens_fallback_and_list(self, mock_query, mock_search):
        mock_search.return_value = [{"text": "chunk1", "page_number": 2}]
        mock_query.return_value = "Response"

        chat_history = [
            {"role": "user", "name": "Alice", "content": "hello"},
            {"role": "assistant", "content": [{"text": "list content"}, "other string"]}
        ]

        # 1. Normal path with name and list content (lines 620-624, 625-626)
        list(rag_anz.analyze(
            query="test",
            stream=False,
            chat_history=chat_history
        ))

        # 2. Fallback path (encoding is None) (lines 605-610)
        with patch("tiktoken.get_encoding", side_effect=Exception("no tiktoken")):
            list(rag_anz.analyze(
                query="test",
                stream=False,
                chat_history=chat_history
            ))


if __name__ == "__main__":
    unittest.main()
