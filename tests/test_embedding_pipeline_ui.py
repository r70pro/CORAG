"""
Unit tests for embedding_pipeline_ui.py targeting 100% coverage.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["TESTING"] = "true"

import gradio as gr
import embedding_pipeline_ui


class TestEmbeddingPipelineUI(unittest.TestCase):

    def test_get_embedding_telemetry_html_success_auto_cpu(self):
        with patch("embedding_pipeline_ui.load_settings", return_value={"embedding_model": "bge", "embedding_device": "auto"}), \
             patch("embedding_pipeline_ui.get_collection_name", return_value="col_name_test"), \
             patch("embedding_pipeline_ui.get_collection_info", return_value={"points_count": 42}), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("rag.cache.is_healthy", return_value=True), \
             patch("rag.cache.get_cache_info", return_value={"cached_embeddings": 100}):
            html = embedding_pipeline_ui.get_embedding_telemetry_html()
            self.assertIn("42 Points", html)
            self.assertIn("100 vectors", html)
            self.assertIn("CPU MODE", html)

    def test_get_embedding_telemetry_html_success_auto_cuda(self):
        with patch("embedding_pipeline_ui.load_settings", return_value={"embedding_model": "bge", "embedding_device": "auto"}), \
             patch("embedding_pipeline_ui.get_collection_name", return_value="col_name_test"), \
             patch("embedding_pipeline_ui.get_collection_info", return_value={"points_count": 10}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("rag.cache.is_healthy", return_value=False):
            html = embedding_pipeline_ui.get_embedding_telemetry_html()
            self.assertIn("CUDA GPU", html)
            self.assertIn("N/A", html)

    def test_get_embedding_telemetry_html_torch_import_error(self):
        with patch("embedding_pipeline_ui.load_settings", return_value={"embedding_model": "bge", "embedding_device": "auto"}), \
             patch("embedding_pipeline_ui.get_collection_name", return_value="col_name_test"), \
             patch("embedding_pipeline_ui.get_collection_info", return_value={"points_count": 0}), \
             patch.dict("sys.modules", {"torch": None}):
            html = embedding_pipeline_ui.get_embedding_telemetry_html()
            self.assertIn("CPU MODE", html)

    def test_get_embedding_telemetry_html_exception(self):
        with patch("embedding_pipeline_ui.load_settings", side_effect=Exception("DB Error")):
            html = embedding_pipeline_ui.get_embedding_telemetry_html()
            self.assertIn("Error loading telemetry: DB Error", html)

    def test_save_embedding_pipeline_settings_success(self):
        with patch("embedding_pipeline_ui.save_settings", return_value="OK"), \
             patch("embedding_pipeline_ui.log_to_rag") as mock_log:
            msg = embedding_pipeline_ui.save_embedding_pipeline_settings(
                "model_x", "cuda", 500, 50, 32
            )
            self.assertIn("✅", msg)
            mock_log.assert_called_once()

    def test_save_embedding_pipeline_settings_failure(self):
        with patch("embedding_pipeline_ui.save_settings", side_effect=Exception("Disk Full")), \
             patch("embedding_pipeline_ui.log_to_rag") as mock_log:
            msg = embedding_pipeline_ui.save_embedding_pipeline_settings(
                "model_x", "cuda", 500, 50, 32
            )
            self.assertIn("❌ Save error: Disk Full", msg)
            mock_log.assert_called_once()

    def test_purge_embedding_cache_success(self):
        with patch("rag.cache.invalidate_embedding_cache") as mock_purge, \
             patch("embedding_pipeline_ui.log_to_rag") as mock_log:
            msg = embedding_pipeline_ui.purge_embedding_cache()
            self.assertIn("✅ Redis embedding cache cleared!", msg)
            mock_purge.assert_called_once()
            mock_log.assert_called_once()

    def test_purge_embedding_cache_failure(self):
        with patch("rag.cache.invalidate_embedding_cache", side_effect=Exception("Redis Offline")):
            msg = embedding_pipeline_ui.purge_embedding_cache()
            self.assertIn("❌ Cache purge error: Redis Offline", msg)

    def test_build_embedding_pipeline_ui(self):
        with patch("embedding_pipeline_ui.load_settings", return_value={
            "embedding_device": "auto",
            "embedding_model": "custom-model",
            "embedding_batch_size": 64,
            "chunk_size": 800,
            "chunk_overlap": 100,
        }), patch("embedding_pipeline_ui.get_available_runs", return_value=[("run_1", "/path/run_1")]):
            with gr.Blocks() as demo:
                result = embedding_pipeline_ui.build_embedding_pipeline_ui()
            self.assertIn("refresh_fn", result)
            self.assertIn("telemetry_comp", result)

            # Test internal callback functions
            callbacks = {}
            for block_fn in demo.fns.values():
                fn = block_fn.fn
                if fn:
                    callbacks[getattr(fn, "__name__", "")] = fn

            if "_get_updated_case_choices" in callbacks:
                with patch("embedding_pipeline_ui.get_available_runs", return_value=[("Run X", "/path/x")]):
                    res = callbacks["_get_updated_case_choices"]()
                    self.assertIn("choices", res)

            if "_get_updated_run_selector_choices" in callbacks:
                with patch("embedding_pipeline_ui.get_available_runs", return_value=[("Run X", "/path/x")]):
                    res = callbacks["_get_updated_run_selector_choices"]()
                    self.assertIn("choices", res)


if __name__ == "__main__":
    unittest.main()
