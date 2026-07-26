import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

from rag_ui import upload_and_index_markdown, upload_and_index_markdown_ui_wrapper

class TestExternalMDUpload(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_upload_validation_empty_files(self):
        # Test validation for empty files list
        updates = list(upload_and_index_markdown([], "new", "New Case"))
        self.assertIn("No files uploaded", "".join(updates))

    def test_upload_validation_missing_case_name(self):
        # Test validation for missing new case name when option is "new"
        mock_file = MagicMock()
        updates = list(upload_and_index_markdown([mock_file], "new", ""))
        self.assertIn("New case name is required", "".join(updates))

    @patch("rag_ui.load_settings", return_value={"chunk_size": 800, "chunk_overlap": 100})
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.db.insert_chunks")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.db.mark_run_indexed")
    @patch("rag.db.get_connection")
    @patch("rag.chunker.chunk_document")
    @patch("rag.embedding.upsert_chunks_generator")
    @patch("rag.storage.upload_markdown")
    @patch("rag.cache.invalidate_query_cache")
    def test_upload_new_case_success(self, mock_invalidate, mock_upload_md, mock_upsert, mock_chunk, mock_conn, mock_mark_run, mock_mark_doc, mock_insert_chunks, mock_register_doc, mock_register_run, mock_settings):
        # Mock file object with .name attribute
        file_path = os.path.join(self.tmp_dir, "test_file.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Hello World\nThis is a test markdown file.")

        mock_file = MagicMock()
        mock_file.name = file_path

        mock_chunk.return_value = [{"chunk_id": "c1", "text": "Hello World"}]
        def mock_upsert_side_effect(chunks, *args, **kwargs):
            for i, c in enumerate(chunks):
                c["qdrant_point_id"] = f"p{i+1}"
            yield {"stage": "embedding", "current": len(chunks), "total": len(chunks)}
            yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}
        mock_upsert.side_effect = mock_upsert_side_effect

        # Mock Postgres connection to return total count
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]  # 1 chunk in run
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

        delegated_updates = [
            "📁 Creating new case: **My Custom Case**...\n",
            "📄 Staged **test_file.md** for case storage.\n",
            "✅ Successfully uploaded and indexed.\n",
        ]
        with patch(
            "indexing_service.CorpusIndexingService.add_markdown_to_case",
            return_value=delegated_updates,
        ) as mock_service:
            updates = list(upload_and_index_markdown([mock_file], "new", "My Custom Case"))
            full_status = "".join(updates)

            self.assertIn("Creating new case", full_status)
            self.assertIn("Staged **test_file.md**", full_status)
            self.assertIn("Successfully uploaded and indexed", full_status)
            mock_service.assert_called_once_with(
                [mock_file], "new", "My Custom Case"
            )

    @patch("rag_ui.load_settings", return_value={"chunk_size": 800, "chunk_overlap": 100})
    @patch("rag.db.get_runs_with_stats")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.db.insert_chunks")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.db.mark_run_indexed")
    @patch("rag.db.get_connection")
    @patch("rag.chunker.chunk_document")
    @patch("rag.embedding.upsert_chunks_generator")
    @patch("rag.storage.upload_markdown")
    @patch("rag.cache.invalidate_query_cache")
    def test_upload_existing_case_success(self, mock_invalidate, mock_upload_md, mock_upsert, mock_chunk, mock_conn, mock_mark_run, mock_mark_doc, mock_insert_chunks, mock_register_doc, mock_register_run, mock_get_runs, mock_settings):
        # Prepare existing case run folder
        existing_run_name = "run_existing_123"
        existing_run_dir = os.path.join(self.tmp_dir, existing_run_name)
        os.makedirs(existing_run_dir)

        mock_get_runs.return_value = [{"run_id": "r123", "run_dir": existing_run_dir}]

        file_path = os.path.join(self.tmp_dir, "another.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Another file\nContent here.")

        mock_file = MagicMock()
        mock_file.name = file_path

        mock_chunk.return_value = [{"chunk_id": "c2", "text": "Another file"}]
        def mock_upsert_side_effect(chunks, *args, **kwargs):
            for i, c in enumerate(chunks):
                c["qdrant_point_id"] = f"p{i+1}"
            yield {"stage": "embedding", "current": len(chunks), "total": len(chunks)}
            yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}
        mock_upsert.side_effect = mock_upsert_side_effect

        # Mock Postgres connection to return total count
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [[0], [1]]  # First SELECT COUNT(*) for docs (0), second for chunks (1)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

        delegated_updates = [
            "📁 Adding to existing case: **r123**...\n",
            "📄 Staged **another.md** for case storage.\n",
            "✅ Successfully uploaded and indexed.\n",
        ]
        with patch(
            "indexing_service.CorpusIndexingService.add_markdown_to_case",
            return_value=delegated_updates,
        ) as mock_service:
            updates = list(upload_and_index_markdown([mock_file], "r123", ""))
            full_status = "".join(updates)

            self.assertIn("Adding to existing case", full_status)
            self.assertIn("Staged **another.md**", full_status)
            self.assertIn("Successfully uploaded and indexed", full_status)
            mock_service.assert_called_once_with([mock_file], "r123", "")

    @patch("rag_ui.upload_and_index_markdown")
    @patch("rag_ui.log_to_rag")
    @patch("rag_ui.get_rag_logs", return_value="Log content")
    def test_ui_wrapper(self, mock_get_logs, mock_log, mock_upload):
        mock_upload.return_value = ["step1\n", "step2\n"]
        res = list(upload_and_index_markdown_ui_wrapper([], "new", ""))
        self.assertEqual(len(res), 2)
        self.assertIn("RAG Indexing Progress", res[0][0])
        self.assertIn("RAG Indexing Progress", res[1][0])

    @patch("rag_ui._get_indexed_run_choices", return_value=[("run_display", "run1")])
    def test_refresh_active_case_after_upload(self, mock_choices):
        import rag_ui
        rag_ui.LAST_CREATED_RUN_ID = "run1"
        res = rag_ui._refresh_active_case_after_upload()
        self.assertEqual(res.get("value"), "run1")
        self.assertEqual(res.get("choices"), [("run_display", "run1")])

    @patch("rag.db.get_runs_with_stats")
    def test_upload_existing_case_fetch_stats_exception(self, mock_get_runs):
        mock_get_runs.side_effect = Exception("DB error")
        updates = list(upload_and_index_markdown([MagicMock()], "r123", ""))
        self.assertIn("Failed to fetch case information", "".join(updates))

    @patch("rag.db.get_runs_with_stats", return_value=[])
    @patch("rag.db.get_connection")
    def test_upload_existing_case_query_db_fallback_success(self, mock_conn, mock_get_runs):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ["/mock/existing/dir"]
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        updates = list(upload_and_index_markdown([MagicMock()], "r123", ""))
        self.assertIn("Could not locate existing case directory", "".join(updates))

    @patch("rag.db.get_runs_with_stats", return_value=[])
    @patch("rag.db.get_connection")
    def test_upload_existing_case_query_db_fallback_exception(self, mock_conn, mock_get_runs):
        mock_conn.side_effect = Exception("Conn error")
        updates = list(upload_and_index_markdown([MagicMock()], "r123", ""))
        self.assertIn("Failed to retrieve run directory", "".join(updates))

    @patch("rag.db.get_runs_with_stats", return_value=[])
    @patch("rag.db.get_connection")
    def test_upload_existing_case_not_found(self, mock_conn, mock_get_runs):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        updates = list(upload_and_index_markdown([MagicMock()], "r123", ""))
        self.assertIn("Could not locate existing case directory", "".join(updates))

    @patch("rag_ui.WORKSPACE_DIR", "/mock")
    @patch("os.path.exists", return_value=True)
    @patch("shutil.copy")
    @patch("os.makedirs")
    def test_upload_copy_exception(self, mock_makedirs, mock_copy, mock_exists):
        mock_copy.side_effect = Exception("Copy failed")
        mock_file = MagicMock()
        mock_file.name = "test.md"
        updates = list(upload_and_index_markdown([mock_file], "new", "MyCase"))
        self.assertIn("Could not copy test.md", "".join(updates))
        self.assertIn("No files were successfully copied", "".join(updates))


if __name__ == "__main__":
    unittest.main()
