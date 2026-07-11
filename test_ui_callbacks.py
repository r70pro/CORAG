"""
Unit tests targeting remaining code coverage in rag_ui.py callbacks.
"""

import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import gradio as gr

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag_ui


class TestUICallbacks(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_get_available_runs(self):
        # 1. Workspace does not exist
        with patch("rag_ui.WORKSPACE_DIR", "/nonexistent"):
            self.assertEqual(rag_ui.get_available_runs(), [])

        # 2. Workspace exists with multiple run directory structures
        with patch("rag_ui.WORKSPACE_DIR", self.tmp_dir):
            # Normal run folder with markdown files
            run_ok = os.path.join(self.tmp_dir, "run_ok")
            os.makedirs(os.path.join(run_ok, "markdown", "inputs"))
            with open(os.path.join(run_ok, "markdown", "inputs", "doc.md"), "w") as f:
                f.write("# doc")
            
            # Normal run folder without markdown files
            run_empty = os.path.join(self.tmp_dir, "run_empty")
            os.makedirs(os.path.join(run_empty, "markdown", "inputs"))

            # Non-run folder name
            not_run = os.path.join(self.tmp_dir, "other_folder")
            os.makedirs(os.path.join(not_run, "markdown", "inputs"))

            # Non-directory file
            file_run = os.path.join(self.tmp_dir, "run_file")
            with open(file_run, "w") as f:
                f.write("")

            runs = rag_ui.get_available_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0][1], run_ok)

    @patch("os.path.exists")
    @patch("rag_ui.load_settings")
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.is_run_indexed")
    def test_index_run_already_indexed(self, mock_is_indexed, mock_chunk, mock_settings, mock_exists):
        mock_exists.return_value = True
        mock_settings.return_value = {}
        mock_is_indexed.return_value = True

        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("already indexed" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    def test_index_run_invalid_dir(self, mock_exists):
        # Trigger directory exists to False
        mock_exists.return_value = False
        updates = list(rag_ui.index_run(None))
        self.assertTrue("Invalid run directory" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed")
    def test_index_run_status_check_exception(self, mock_is_indexed, mock_exists):
        # Trigger exception on is_run_indexed check
        mock_is_indexed.side_effect = Exception("DB failure")
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Could not check index status" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    def test_index_run_chunk_exception(self, mock_chunk, mock_settings, mock_is_indexed, mock_exists):
        mock_chunk.side_effect = Exception("Chunker crash")
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Chunking failed" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run", return_value={})
    def test_index_run_no_chunks(self, mock_chunk, mock_settings, mock_is_indexed, mock_exists):
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("No markdown files found" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    def test_index_run_db_registration_exception(self, mock_reg_run, mock_chunk, mock_settings, mock_is_indexed, mock_exists):
        mock_chunk.return_value = {"doc_1": {"md_file": "report.md", "md_path": "/tmp/report.md", "chunks": []}}
        mock_reg_run.side_effect = Exception("DB write error")
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Database registration failed" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.embedding.upsert_chunks")
    @patch("rag.db.insert_chunks")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.db.mark_run_indexed")
    def test_index_run_storage_upload_warning(
        self, mock_mark_run, mock_mark_doc, mock_insert, mock_upsert,
        mock_upload, mock_reg_doc, mock_reg_run, mock_chunk, mock_settings,
        mock_is_indexed, mock_exists
    ):
        mock_chunk.return_value = {"doc_1": {"md_file": "report.md", "md_path": "/tmp/report.md", "chunks": []}}
        mock_upload.side_effect = Exception("Storage upload error")
        mock_upsert.return_value = []
        updates = list(rag_ui.index_run("/mock/run"))
        full_text = "".join(updates)
        self.assertTrue("Storage upload warning" in full_text)
        self.assertTrue("Successfully indexed" in full_text)

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.embedding.upsert_chunks")
    def test_index_run_embeddings_exception(self, mock_upsert, mock_upload, mock_reg_doc, mock_reg_run, mock_chunk, mock_settings, mock_is_indexed, mock_exists):
        mock_chunk.return_value = {"doc_1": {"md_file": "report.md", "md_path": "/tmp/report.md", "chunks": [{"text": "chunk"}]}}
        mock_upsert.side_effect = Exception("Embedding computation failure")
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Embedding/indexing failed" in "".join(updates))

    @patch("os.path.exists", return_value=True)
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.embedding.upsert_chunks")
    @patch("rag.db.insert_chunks")
    def test_index_run_db_insert_chunks_exception(self, mock_insert, mock_upsert, mock_upload, mock_reg_doc, mock_reg_run, mock_chunk, mock_settings, mock_is_indexed, mock_exists):
        mock_chunk.return_value = {"doc_1": {"md_file": "report.md", "md_path": "/tmp/report.md", "chunks": [{"text": "chunk"}]}}
        mock_upsert.return_value = [{"qdrant_point_id": "p1"}]
        mock_insert.side_effect = Exception("DB chunk insertion fail")
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Embedding/indexing failed" in "".join(updates))

    @patch("rag_ui.get_available_runs")
    @patch("rag_ui.index_run")
    def test_index_all_runs(self, mock_index_run, mock_runs):
        mock_runs.return_value = [("run1", "/mock/run1")]
        mock_index_run.return_value = ["processing run1"]
        
        updates = list(rag_ui.index_all_runs())
        self.assertTrue("processing run1" in "".join(updates))

        # Empty runs case
        mock_runs.return_value = []
        updates2 = list(rag_ui.index_all_runs())
        self.assertTrue("No completed OCR runs found" in "".join(updates2))

    @patch("rag.analyzer.analyze")
    def test_chat_respond(self, mock_analyze):
        mock_analyze.return_value = ["chunk1", "chunk2"]
        
        # Test free Q&A
        res = list(rag_ui.chat_respond(
            message="hello",
            history=[],
            analysis_mode="💬 Free Q&A",
            analysis_model_url="http://local",
            analysis_model_name="phi4",
            top_k=5
        ))
        self.assertEqual(res[-1], "chunk1chunk2")

        # Test empty input early yield
        res2 = list(rag_ui.chat_respond("", [], "💬 Free Q&A", "http://local", "phi4", 5))
        self.assertEqual(res2, [""])

        # Test chat history mapping
        res3 = list(rag_ui.chat_respond(
            message="hello",
            history=[["user_q", "assistant_a"]],
            analysis_mode="📅 Timeline Generator",
            analysis_model_url="http://local",
            analysis_model_name="phi4",
            top_k=5
        ))
        self.assertEqual(res3[-1], "chunk1chunk2")

    @patch("rag.db.get_corpus_stats")
    @patch("rag.embedding.get_collection_info")
    def test_get_corpus_info_success_and_exception(self, mock_qdrant_info, mock_db_stats):
        # 1. Success
        mock_db_stats.return_value = {
            "indexed_runs": 2,
            "indexed_documents": 5,
            "total_chunks": 100,
            "unique_authors": 3,
            "earliest_date": "2020-01-01",
            "latest_date": "2020-12-31"
        }
        mock_qdrant_info.return_value = {"points_count": 100}

        info = rag_ui.get_corpus_info()
        self.assertTrue("Corpus Statistics" in info)
        self.assertTrue("2" in info)

        # 2. Exception
        mock_db_stats.side_effect = Exception("DB error")
        info2 = rag_ui.get_corpus_info()
        self.assertTrue("Could not fetch corpus stats" in info2)

    @patch("rag_infra_manager.start_and_init_rag")
    @patch("rag_infra_manager.stop_rag_infrastructure")
    @patch("rag_infra_manager.get_rag_status_html")
    def test_rag_infra_ui_ops(self, mock_status, mock_stop, mock_start):
        mock_start.return_value = (True, "Started")
        mock_stop.return_value = (True, "Stopped")
        mock_status.return_value = "<span class='badge-running'>Running</span>"

        # Start
        msg, status_html = rag_ui.start_rag_infra_ui()
        self.assertEqual(msg, "Started")
        self.assertEqual(status_html, "<span class='badge-running'>Running</span>")

        # Stop
        msg2, status_html2 = rag_ui.stop_rag_infra_ui()
        self.assertEqual(msg2, "Stopped")

        # Refresh
        status_html4 = rag_ui.refresh_rag_status()
        self.assertEqual(status_html4, "<span class='badge-running'>Running</span>")

    @patch("rag_infra_manager.start_and_init_rag", side_effect=Exception("Infra failure"))
    @patch("rag_infra_manager.stop_rag_infrastructure", side_effect=Exception("Infra stop failure"))
    @patch("rag_infra_manager.get_rag_status_html", side_effect=Exception("Html fail"))
    def test_rag_infra_ui_ops_exceptions(self, mock_status, mock_stop, mock_start):
        msg, status = rag_ui.start_rag_infra_ui()
        self.assertIn("Error", msg)
        self.assertIn("failed", status)

        msg2, status2 = rag_ui.stop_rag_infra_ui()
        self.assertIn("Error", msg2)

        status3 = rag_ui.refresh_rag_status()
        self.assertIn("Unknown", status3)

    @patch("rag_ui.get_available_runs")
    def test_refresh_runs_dropdown(self, mock_runs):
        # Empty
        mock_runs.return_value = []
        res = rag_ui.refresh_runs_dropdown()
        self.assertEqual(res["value"], None)

        # Has values
        mock_runs.return_value = [("run1", "/mock/run1")]
        res2 = rag_ui.refresh_runs_dropdown()
        self.assertEqual(res2["value"], "/mock/run1")

    @patch("rag_ui.get_corpus_info")
    def test_refresh_corpus_display(self, mock_info):
        mock_info.return_value = "stats"
        self.assertEqual(rag_ui.refresh_corpus_display(), "stats")

    @patch("rag.analyzer.analyze")
    def test_inner_gradio_callbacks(self, mock_analyze):
        with gr.Blocks() as demo:
            rag_ui.build_analysis_ui()

        # Extract registered functions from demo.fns.values()
        registered_fns = {}
        for block_fn in demo.fns.values():
            fn = block_fn.fn
            if fn:
                name = getattr(fn, "__name__", "lambda")
                registered_fns[name] = fn

        # Extract registered functions
        user_submit = registered_fns.get("user_message_submit")
        bot_respond = registered_fns.get("bot_respond")
        save_settings = registered_fns.get("save_analysis_settings")

        self.assertIsNotNone(user_submit)
        self.assertIsNotNone(bot_respond)
        self.assertIsNotNone(save_settings)

        # 1. Test user_message_submit
        self.assertEqual(user_submit("", []), ("", []))
        self.assertEqual(user_submit("hello", []), ("", [{"role": "user", "content": "hello"}]))

        # 2. Test bot_respond empty/missing user msg
        rag_ui.RAG_LOG_BUFFER.clear()
        self.assertEqual(list(bot_respond([], "Free Q&A", "http://", "phi", 5)), [([], "")])
        rag_ui.RAG_LOG_BUFFER.clear()
        self.assertEqual(list(bot_respond([{"role": "assistant", "content": ""}], "Free Q&A", "http://", "phi", 5)), [([{"role": "assistant", "content": ""}], "")])

        # 3. Test bot_respond success stream
        mock_analyze.return_value = ["res1", "res2"]
        history = [{"role": "user", "content": "hello"}]
        res_stream = list(bot_respond(history, "Free Q&A", "http://", "phi", 5))
        self.assertEqual(res_stream[-1][0][-1]["content"], "res1res2")

        # 4. Test bot_respond exception
        mock_analyze.side_effect = Exception("Generation crash")
        history2 = [{"role": "user", "content": "hello"}]
        res_stream_fail = list(bot_respond(history2, "Free Q&A", "http://", "phi", 5))
        self.assertIn("Error", res_stream_fail[-1][0][-1]["content"])

        # 5. Test save_analysis_settings success/failure
        with patch("settings_manager.save_settings") as mock_save:
            status = save_settings("url", "name", 5, "emb")
            self.assertIn("saved successfully", status)

            mock_save.side_effect = Exception("Disk full")
            status2 = save_settings("url", "name", 5, "emb")
            self.assertIn("Error", status2)

    @patch("os.path.exists")
    @patch("rag.db.is_run_indexed", return_value=False)
    @patch("rag_ui.load_settings", return_value={})
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.embedding.upsert_chunks")
    @patch("rag.db.insert_chunks")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.db.mark_run_indexed")
    @patch("rag.cache.invalidate_query_cache")
    def test_index_run_edge_paths(
        self, mock_invalidate, mock_mark_run, mock_mark_doc, mock_insert, mock_upsert,
        mock_upload, mock_reg_doc, mock_reg_run, mock_chunk, mock_settings,
        mock_is_indexed, mock_exists
    ):
        # 1. Custom dict subclass that yields empty items() to trigger 139->155 branch
        class SneakyDict(dict):
            def __bool__(self):
                return True
            def __len__(self):
                return 1
            def items(self):
                return []

        mock_chunk.return_value = SneakyDict()
        mock_is_indexed.return_value = False
        updates = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Successfully indexed" in "".join(updates))

        # 2. Test page_ranges length logic (line 121) and file path exists paths (lines 141, 147)
        normal_dict = {
            "doc_1": {
                "md_file": "report.md",
                "md_path": "/tmp/report.md",
                "page_ranges": [[0, 100, 1]], # triggers line 121
                "chunks": []
            },
            "doc_2": {
                "md_file": "missing.md",
                "md_path": "/tmp/missing.md",
                "page_ranges": [[0, 10, 1]],
                "chunks": []
            }
        }
        mock_chunk.return_value = normal_dict
        
        # Configure exists_side_effect to return True for run_dir and markdown path but False for PDF path
        def exists_side_effect(path):
            if path in ["/mock/run", "/tmp/report.md"]:
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        
        updates2 = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Successfully indexed" in "".join(updates2))

        # 3. Test invalidate_query_cache raising exception (lines 182-183)
        mock_invalidate.side_effect = Exception("Cache invalidation failed")
        updates3 = list(rag_ui.index_run("/mock/run"))
        self.assertTrue("Successfully indexed" in "".join(updates3))

    @patch("rag.analyzer.analyze")
    def test_chat_respond_history_branches(self, mock_analyze):
        mock_analyze.return_value = ["response"]
        
        # Test empty message early return
        res = list(rag_ui.chat_respond(message="", history=[], analysis_mode="💬 Free Q&A", analysis_model_url="http://", analysis_model_name="phi", top_k=5))
        self.assertEqual(res, [""])

        # Test empty user_msg and empty assistant_msg branches in history loop (lines 234, 236)
        history = [
            ["", "assistant_only"],
            ["user_only", ""],
            ["", ""]
        ]
        res2 = list(rag_ui.chat_respond(message="hello", history=history, analysis_mode="💬 Free Q&A", analysis_model_url="http://", analysis_model_name="phi", top_k=5))
        self.assertEqual(res2[-1], "response")

        # Test analyze exception handler in chat_respond (lines 265-266)
        mock_analyze.side_effect = Exception("Analyzer crash")
        res3 = list(rag_ui.chat_respond(message="hello", history=[], analysis_mode="💬 Free Q&A", analysis_model_url="http://", analysis_model_name="phi", top_k=5))
        self.assertTrue("Error: Analyzer crash" in res3[-1])

    @patch("rag.analyzer.analyze")
    def test_bot_respond_history_pairs(self, mock_analyze):
        # Build analysis UI structure for testing callbacks
        with gr.Blocks() as demo:
            rag_ui.build_analysis_ui()

        registered_fns = {}
        for block_fn in demo.fns.values():
            fn = block_fn.fn
            if fn:
                name = getattr(fn, "__name__", "lambda")
                registered_fns[name] = fn

        bot_respond = registered_fns.get("bot_respond")
        self.assertIsNotNone(bot_respond)

        # Test bot_respond with history of length > 1 (e.g. 3) to cover line 536
        mock_analyze.return_value = ["ok"]
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}
        ]
        res_stream = list(bot_respond(history, "Free Q&A", "http://", "phi", 5))
        self.assertEqual(res_stream[-1][0][-1]["content"], "ok")

    def test_extract_text_content(self):
        from rag_ui import extract_text_content
        # 1. string
        self.assertEqual(extract_text_content("hello"), "hello")
        # 2. list of string
        self.assertEqual(extract_text_content(["hello", " ", "world"]), "hello world")
        # 3. Gradio 6 style list of dicts
        self.assertEqual(extract_text_content([{"text": "what is claimant's name?", "type": "text"}]), "what is claimant's name?")
        # 4. dict
        self.assertEqual(extract_text_content({"text": "test"}), "test")
        # 5. empty/None
        self.assertEqual(extract_text_content(None), "")


if __name__ == "__main__":
    unittest.main()
