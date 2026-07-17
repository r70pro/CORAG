"""
Unit tests for rag_ui_handlers.py, rag_ui_dashboard.py, and rag_ui_state.py targeting 100% statement and branch coverage.
"""

import os
import unittest
from unittest.mock import patch
import gradio as gr

# Prevent system operations during import
os.environ["TESTING"] = "true"

import rag_ui_handlers
import rag_ui_dashboard
import rag_ui_state


class TestRAGUIHandlers(unittest.TestCase):

    def test_text_extraction_formats(self):
        # 1. string content
        self.assertEqual(rag_ui_state.extract_text_content("hello"), "hello")
        
        # 2. empty content
        self.assertEqual(rag_ui_state.extract_text_content(None), "")

        # 3. dict with "text" key
        self.assertEqual(rag_ui_state.extract_text_content({"text": "dict_text"}), "dict_text")
        # 4. dict without "text" key
        self.assertEqual(rag_ui_state.extract_text_content({"other": "val"}), "{'other': 'val'}")

        # 5. list with string, dict, other type
        lst = [
            "str_item",
            {"text": "dict_item"},
            {"other": "skip"},
            123 # other type
        ]
        self.assertEqual(rag_ui_state.extract_text_content(lst), "str_itemdict_item")

        # 6. other type fallback (like float)
        self.assertEqual(rag_ui_state.extract_text_content(3.14), "3.14")

    @patch("rag_ui_handlers.get_available_runs")
    def test_refresh_runs_dropdown(self, mock_runs):
        # 1. Populated dropdown
        mock_runs.return_value = [("lbl1", "run1"), ("lbl2", "run2")]
        res1 = rag_ui_handlers.refresh_runs_dropdown()
        self.assertEqual(res1.get("choices"), [("lbl1", "run1"), ("lbl2", "run2")])
        self.assertEqual(res1.get("value"), "run1")

        # 2. Empty dropdown
        mock_runs.return_value = []
        res2 = rag_ui_handlers.refresh_runs_dropdown()
        self.assertEqual(res2.get("choices"), [])
        self.assertIsNone(res2.get("value"))

    def test_infrastructure_and_stats_wrappers(self):
        with patch("rag_ui_handlers.start_rag_infra_ui", return_value=("started", "<span class='badge-active'>Active</span>")):
            res = rag_ui_handlers.start_rag_infra_ui_wrapper()
            self.assertEqual(res[0], "started")
            self.assertIn("Active", res[1])

        with patch("rag_ui_handlers.stop_rag_infra_ui", return_value=("stopped", "<span class='badge-idle'>Idle</span>")):
            res2 = rag_ui_handlers.stop_rag_infra_ui_wrapper()
            self.assertEqual(res2[0], "stopped")
            self.assertIn("Idle", res2[1])

        with patch("rag_ui_handlers.get_corpus_info", return_value="corpus stats"):
            res3 = rag_ui_handlers.refresh_corpus_display()
            self.assertEqual(res3, "corpus stats")

        # forward declarations
        with patch("rag_ui_dashboard._get_indexed_run_choices", return_value=["c1"]):
            self.assertEqual(rag_ui_handlers._get_indexed_run_choices(), ["c1"])
        with patch("rag_ui_dashboard._build_dashboard_html", return_value="html"):
            self.assertEqual(rag_ui_handlers._build_dashboard_html(), "html")

    def test_indexing_wrappers(self):
        # 1. index_run_ui_wrapper
        with patch("rag_ui_handlers.index_run", return_value=iter(["updating", "completed"])):
            gen1 = rag_ui_handlers.index_run_ui_wrapper("/path/to/run")
            res1 = list(gen1)
            self.assertEqual(res1[0][0], "updating")
            self.assertEqual(res1[1][0], "updatingcompleted")

        # 2. index_all_runs_ui_wrapper
        with patch("rag_ui_handlers.index_all_runs", return_value=iter(["all_updating"])):
            gen2 = rag_ui_handlers.index_all_runs_ui_wrapper()
            res2 = list(gen2)
            self.assertEqual(res2[0][0], "all_updating")

        # 3. upload_and_index_markdown_ui_wrapper
        with patch("rag_ui_handlers.upload_and_index_markdown", return_value=iter(["upload_updating"])):
            gen3 = rag_ui_handlers.upload_and_index_markdown_ui_wrapper([], "option", "name")
            res3 = list(gen3)
            self.assertEqual(res3[0][0], "upload_updating")

    @patch("rag.analyzer.analyze")
    def test_bot_respond_filters_and_exceptions(self, mock_analyze):
        # 1. Empty history
        res1 = list(rag_ui_handlers.bot_respond([], "mode", "url", "model", 5, None, None, None, None, None))
        self.assertEqual(res1[0][0], [])

        # 2. History without user message
        res2 = list(rag_ui_handlers.bot_respond([{"role": "assistant", "content": "hello"}], "mode", "url", "model", 5, None, None, None, None, None))
        self.assertEqual(res2[0][0][0]["content"], "hello")

        # 3. Valid respond with all filters enabled
        mock_analyze.return_value = iter(["bot ", "response"])
        history = [{"role": "user", "content": "hello"}]
        res3 = list(rag_ui_handlers.bot_respond(
            history, "mode", "url", "model", 5,
            active_case="case_r123", doc_type="letter", author="Dr Ek",
            date_from="2020-01-01", date_to="2020-12-31"
        ))
        # Last item in generator should have complete response
        final_history = res3[-1][0]
        self.assertEqual(final_history[1]["content"], "bot response")

        # 4. Exception raised during analyze
        mock_analyze.side_effect = Exception("Model timed out")
        res4 = list(rag_ui_handlers.bot_respond(
            [{"role": "user", "content": "hello"}], "mode", "url", "model", 5, None, None, None, None, None
        ))
        self.assertIn("Error: Model timed out", res4[-1][0][1]["content"])

    @patch("settings_manager.save_settings")
    def test_save_analysis_settings(self, mock_save):
        res = rag_ui_handlers.save_analysis_settings("url", "name", 5, "emb_model", None, None, None)
        self.assertEqual(res, "✅ Analysis configuration saved successfully.")
        mock_save.assert_called_once()

        mock_save.side_effect = Exception("disk full")
        res2 = rag_ui_handlers.save_analysis_settings("url", "name", 5, "emb_model", None, None, None)
        self.assertTrue(res2.startswith("❌ Error:"))

    @patch("rag_ui_handlers._get_indexed_run_choices")
    @patch("rag_export.export_chat_markdown")
    @patch("rag_export.export_chat_text")
    @patch("rag_export.export_timeline_csv")
    def test_export_chat_formats(self, mock_csv, mock_txt, mock_md, mock_choices):
        # Setup mock choices to test case name resolution
        mock_choices.return_value = [("📁 case_name", "case_r123")]

        # 1. markdown export success
        mock_md.return_value = "/path/to/chat.md"
        res_md = rag_ui_handlers._do_export_md([], "mode", "case_r123")
        self.assertEqual(res_md.get("value"), "/path/to/chat.md")
        mock_md.assert_called_with([], "mode", "📁 case_name")

        # markdown export fails (returns empty update)
        mock_md.return_value = None
        self.assertFalse(rag_ui_handlers._do_export_md([], "mode", "case_r123").get("visible"))

        # 2. text export success
        mock_txt.return_value = "/path/to/chat.txt"
        res_txt = rag_ui_handlers._do_export_txt([], "mode", "case_r123")
        self.assertEqual(res_txt.get("value"), "/path/to/chat.txt")
        mock_txt.assert_called_with([], "mode", "📁 case_name")

        # text export fails (returns empty update)
        mock_txt.side_effect = Exception("Export error")
        self.assertFalse(rag_ui_handlers._do_export_txt([], "mode", "case_r123").get("visible"))
        mock_txt.side_effect = None

        # 3. CSV export success
        mock_csv.return_value = "/path/to/timeline.csv"
        res_csv = rag_ui_handlers._do_export_csv([], "case_r123")
        self.assertEqual(res_csv.get("value"), "/path/to/timeline.csv")
        mock_csv.assert_called_with([], "📁 case_name")

        # CSV export returns None (no table data)
        mock_csv.return_value = None
        self.assertFalse(rag_ui_handlers._do_export_csv([], "case_r123").get("visible"))

    @patch("rag.db.get_runs_with_stats")
    def test_dashboard_indexed_choices_and_exceptions(self, mock_get_runs):
        # 1. _build_dashboard_html exception
        mock_get_runs.side_effect = Exception("DB offline")
        html = rag_ui_dashboard._build_dashboard_html()
        self.assertIn("Cannot load cases", html)

        # 2. _get_indexed_run_choices exception
        choices = rag_ui_dashboard._get_indexed_run_choices()
        # Should catch exception and return default choices list
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0][1], "")

    @patch("rag_ui_dashboard.get_rag_ui_fn")
    def test_refresh_active_case_after_upload(self, mock_get_fn):
        mock_get_fn.return_value = lambda: [("run_label", "run_id")]
        
        # Scenario 1: LAST_CREATED_RUN_ID is set
        rag_ui_state.LAST_CREATED_RUN_ID = "r999"
        res1 = rag_ui_dashboard._refresh_active_case_after_upload()
        self.assertEqual(res1.get("value"), "r999")

        # Scenario 2: LAST_CREATED_RUN_ID is None
        rag_ui_state.LAST_CREATED_RUN_ID = None
        res2 = rag_ui_dashboard._refresh_active_case_after_upload()
        self.assertEqual(res2.get("value"), "")

    @patch("rag.db.delete_run_data")
    @patch("rag.embedding.delete_run_vectors")
    @patch("rag.storage.delete_run_objects")
    @patch("rag.cache.invalidate_query_cache")
    def test_delete_case_handling(self, mock_invalidate, mock_del_objects, mock_del_vectors, mock_del_db):
        # Mock functions for return html/choices
        with gr.Blocks():
            with patch("rag_ui_dashboard._build_dashboard_html", return_value="dashboard"):
                with patch("rag_ui_dashboard._get_indexed_run_choices", return_value=[("lbl", "r1")]):
                    components = rag_ui_dashboard.build_case_dashboard_ui()
                    delete_case_fn = components["delete_fn"]

                    # 1. No case selected
                    res1 = delete_case_fn("")
                    self.assertEqual(res1[2], "⚠️ No case selected.")

                    # 2. Success deletion
                    res2 = delete_case_fn("r1")
                    self.assertEqual(res2[2], "✅ Case deleted successfully.")
                    mock_del_db.assert_called_with("r1")
                    mock_del_vectors.assert_called_with("r1")
                    mock_del_objects.assert_called_with("r1")
                    mock_invalidate.assert_called_once()

                    # 3. Exceptions during delete parts
                    mock_del_db.side_effect = Exception("db error")
                    mock_del_vectors.side_effect = Exception("vector error")
                    mock_del_objects.side_effect = Exception("storage error")
                    mock_invalidate.side_effect = Exception("cache error")
                    
                    # Should not crash, catches internally
                    res3 = delete_case_fn("r1")
                    self.assertEqual(res3[2], "✅ Case deleted successfully.")

    def test_build_case_dashboard_ui(self):
        # Trigger dashboard components layout
        with gr.Blocks():
            with patch("rag_ui_dashboard._build_dashboard_html", return_value="dashboard"):
                components = rag_ui_dashboard.build_case_dashboard_ui()
                self.assertIsNotNone(components["dashboard_html"])
                
                # Trigger refresh function
                with patch("rag_ui_dashboard._get_indexed_run_choices", return_value=[("lbl", "r1")]):
                    html, update_choices, status = components["refresh_fn"]()
                    self.assertEqual(html, "dashboard")
                    self.assertEqual(update_choices.get("choices"), [("lbl", "r1")])


if __name__ == "__main__":
    unittest.main()
