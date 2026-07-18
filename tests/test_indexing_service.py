"""
Unit tests for indexing_service.py targeting 100% statement and branch coverage.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

from indexing_service import CorpusIndexingService


class TestIndexingService(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmp_dir, "run_test_1")
        os.makedirs(self.run_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @patch("indexing_service.load_settings")
    def test_index_run_basic_scenarios(self, mock_load_settings):
        # 1. Invalid run dir
        updates1 = list(CorpusIndexingService.index_run(None))
        self.assertIn("Invalid run directory", "".join(updates1))

        # 2. Already indexed
        with patch("rag.db.is_run_indexed", return_value=True):
            updates2 = list(CorpusIndexingService.index_run(self.run_dir))
            self.assertIn("already indexed", "".join(updates2))

        # 3. Status check raises Exception
        with patch("rag.db.is_run_indexed", side_effect=Exception("DB down")):
            # It should yield a warning but continue
            with patch("rag.chunker.chunk_documents_from_run", side_effect=Exception("Chunk failed")):
                updates3 = list(CorpusIndexingService.index_run(self.run_dir))
                self.assertIn("Could not check index status", "".join(updates3))
                self.assertIn("Chunking failed", "".join(updates3))

    @patch("rag.chunker.chunk_documents_from_run")
    def test_index_run_no_markdown_files(self, mock_chunk):
        mock_chunk.return_value = {}
        with patch("rag.db.is_run_indexed", return_value=False):
            updates = list(CorpusIndexingService.index_run(self.run_dir))
            self.assertIn("No markdown files found", "".join(updates))

    @patch("rag.db.mark_run_indexed")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.chunker.chunk_documents_from_run")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.embedding.upsert_chunks_generator")
    @patch("rag.db.insert_chunks")
    def test_index_run_full_flow_with_exceptions(self, mock_insert, mock_upsert, mock_upload, mock_reg_doc, mock_register_run, mock_chunk, mock_mark_doc, mock_mark_run):
        mock_chunk.return_value = {
            "doc1": {
                "md_file": "0001_doc.md",
                "md_path": os.path.join(self.run_dir, "0001_doc.md"),
                "page_ranges": [0, 1],
                "chunks": [{"chunk_id": "c1"}]
            }
        }
        def mock_upsert_side_effect(chunks, *args, **kwargs):
            for i, c in enumerate(chunks):
                c["qdrant_point_id"] = f"p{i+1}"
            yield {"stage": "embedding", "current": len(chunks), "total": len(chunks)}
            yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}
        mock_upsert.side_effect = mock_upsert_side_effect

        # Create the md file so upload doesn't skip
        with open(os.path.join(self.run_dir, "0001_doc.md"), "w") as f:
            f.write("# markdown")

        # 1. Database registration fails
        mock_register_run.side_effect = Exception("DB registration failed")
        with patch("rag.db.is_run_indexed", return_value=False):
            updates1 = list(CorpusIndexingService.index_run(self.run_dir))
            self.assertIn("Database registration failed", "".join(updates1))

        # Reset registry side effect
        mock_register_run.side_effect = None

        # 2. Storage upload fails (yields warning, but doesn't crash)
        mock_upload.side_effect = Exception("MinIO error")
        mock_upsert.side_effect = Exception("Upsert failed") # to break early after upload
        with patch("rag.db.is_run_indexed", return_value=False):
            updates2 = list(CorpusIndexingService.index_run(self.run_dir))
            self.assertIn("Storage upload warning", "".join(updates2))
            self.assertIn("Embedding/indexing failed", "".join(updates2))

        # 3. Cache invalidation fails (caught internally)
        mock_upload.side_effect = None
        mock_upsert.side_effect = None
        with patch("rag.db.is_run_indexed", return_value=False):
            with patch("rag.cache.invalidate_query_cache", side_effect=Exception("Cache error")):
                updates3 = list(CorpusIndexingService.index_run(self.run_dir))
                self.assertIn("Successfully indexed", "".join(updates3))

    def test_index_all_runs(self):
        # 1. No runs found
        mock_get_runs = MagicMock(return_value=[])
        updates1 = list(CorpusIndexingService.index_all_runs(mock_get_runs))
        self.assertIn("No completed OCR runs found", "".join(updates1))

        # 2. Runs found -> triggers index_run
        mock_get_runs.return_value = [("run1", self.run_dir)]
        with patch("indexing_service.CorpusIndexingService.index_run") as mock_idx:
            mock_idx.return_value = ["processing run1"]
            updates2 = list(CorpusIndexingService.index_all_runs(mock_get_runs))
            self.assertIn("processing run1", "".join(updates2))

        # 3. Default function trigger when None passed
        with patch("settings_manager.get_available_runs", return_value=[]):
            updates3 = list(CorpusIndexingService.index_all_runs(None))
            self.assertIn("No completed OCR runs found", "".join(updates3))

    @patch("rag.db.get_runs_with_stats")
    @patch("rag.db.get_connection")
    def test_add_markdown_to_case_options_and_errors(self, mock_conn, mock_get_runs):
        # 1. No files uploaded
        updates1 = list(CorpusIndexingService.add_markdown_to_case([], "new", "NewCase"))
        self.assertIn("No files uploaded", "".join(updates1))

        # 2. Case name required for 'new'
        mock_file = MagicMock()
        updates2 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "new", ""))
        self.assertIn("New case name is required", "".join(updates2))

        # 3. Case option 'existing' but get_runs_with_stats raises Exception
        mock_get_runs.side_effect = Exception("DB failure")
        updates3 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "existing_run_id", ""))
        self.assertIn("Failed to fetch case information", "".join(updates3))

        # 4. Case option 'existing', run dir lookup fails in get_runs and DB query raises Exception
        mock_get_runs.side_effect = None
        mock_get_runs.return_value = [] # no matching runs

        # Database context manager mock raises exception on query
        mock_conn_inst = mock_conn.return_value.__enter__.return_value
        mock_cur = mock_conn_inst.cursor.return_value.__enter__.return_value
        mock_cur.execute.side_effect = Exception("DB query failed")

        updates4 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "existing_run_id", ""))
        self.assertIn("Failed to retrieve run directory", "".join(updates4))

        # 5. Case option 'existing', DB query succeeds but returns no row
        mock_cur.execute.side_effect = None
        mock_cur.fetchone.return_value = None
        updates5 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "existing_run_id", ""))
        self.assertIn("Could not locate existing case directory", "".join(updates5))

        # 6. Folder creation fails
        # Patch WORKSPACE_DIR locally on indexing_service module
        with patch("indexing_service.WORKSPACE_DIR", "/sys/readonly/workspace"):
            updates6 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "new", "Case"))
            self.assertIn("Failed to create directories", "".join(updates6))

    @patch("rag.db.mark_run_indexed")
    @patch("rag.db.mark_document_indexed")
    @patch("rag.db.get_connection")
    @patch("rag.db.get_runs_with_stats")
    @patch("rag.db.register_run")
    @patch("rag.db.register_document")
    @patch("rag.storage.upload_markdown")
    @patch("rag.chunker.chunk_document")
    @patch("rag.embedding.upsert_chunks_generator")
    @patch("rag.db.insert_chunks")
    def test_add_markdown_to_case_processing_variants(self, mock_insert, mock_upsert, mock_chunk, mock_upload, mock_reg_doc, mock_reg_run, mock_get_runs, mock_conn, mock_mark_doc, mock_mark_run):
        # Configure get_connection mock
        mock_conn_inst = mock_conn.return_value.__enter__.return_value
        mock_cur = mock_conn_inst.cursor.return_value.__enter__.return_value
        mock_cur.fetchone.return_value = [0]  # Return 0 documents count by default

        # Setup source file
        src_dir = tempfile.mkdtemp()
        src_file = os.path.join(src_dir, "uploaded.md")
        with open(src_file, "w") as f:
            f.write("# markdown file contents")
        
        mock_file = MagicMock()
        mock_file.name = src_file

        # Setup runs matching the test
        mock_get_runs.return_value = [
            {"run_id": "r123", "run_dir": self.run_dir}
        ]

        # 1. Copied files list is empty (source file doesn't exist)
        orig_exists = os.path.exists
        def mock_exists_fn(p):
            if p == src_file:
                return False
            return orig_exists(p)

        with patch("os.path.exists", side_effect=mock_exists_fn):
            updates1 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
            self.assertIn("No files were successfully copied", "".join(updates1))

        # 2. shutil.copy raises Exception
        with patch("shutil.copy", side_effect=Exception("Copy failed")):
            updates2 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
            self.assertIn("Warning: Could not copy", "".join(updates2))
            self.assertIn("No files were successfully copied", "".join(updates2))

        # 3. Database run registration fails
        mock_reg_run.side_effect = Exception("Run register failed")
        updates3 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
        self.assertIn("Database run registration failed", "".join(updates3))
        mock_reg_run.side_effect = None

        # 4. Open markdown file raises Exception
        orig_open = open
        def mock_open_fn(file, mode="r", *args, **kwargs):
            if isinstance(file, str) and "markdown/inputs" in file and "r" in mode:
                raise Exception("Read error")
            return orig_open(file, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_fn):
            updates4 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
            self.assertIn("Error reading uploaded.md", "".join(updates4))
            self.assertIn("No chunks generated", "".join(updates4))

        # 5. Document registration in database fails (skips document)
        mock_reg_doc.side_effect = Exception("Doc register failed")
        updates5 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
        self.assertIn("Database document registration failed for uploaded.md", "".join(updates5))
        self.assertIn("No chunks generated", "".join(updates5))
        mock_reg_doc.side_effect = None

        # 6. MinIO storage upload fails (yields warning, but continues)
        mock_upload.side_effect = Exception("MinIO error")
        mock_chunk.return_value = [{"chunk_id": "c1", "text": "text", "char_start": 0, "char_end": 4}]
        def mock_upsert_side_effect(chunks, *args, **kwargs):
            for i, c in enumerate(chunks):
                c["qdrant_point_id"] = f"p{i+1}"
            yield {"stage": "embedding", "current": len(chunks), "total": len(chunks)}
            yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}
        mock_upsert.side_effect = mock_upsert_side_effect

        updates6 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
        self.assertIn("Storage upload warning for uploaded.md", "".join(updates6))
        self.assertIn("Created **1** chunk(s)", "".join(updates6))
        self.assertIn("Successfully uploaded and indexed", "".join(updates6))
        mock_upload.side_effect = None

        # 7. Chunking fails
        mock_chunk.side_effect = Exception("Chunking failed")
        updates7 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
        self.assertIn("Chunking failed for uploaded.md", "".join(updates7))
        self.assertIn("No chunks generated", "".join(updates7))
        mock_chunk.side_effect = None

        # 8. Embedding/Upsert fails
        mock_upsert.side_effect = Exception("Embedding failed")
        mock_chunk.return_value = [{"chunk_id": "c1", "text": "text", "char_start": 0, "char_end": 4}]
        
        with patch("rag.db.get_connection") as mock_conn_stats:
            mock_cur_stats = mock_conn_stats.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            mock_cur_stats.fetchone.return_value = [5] # count of docs
            
            updates8 = list(CorpusIndexingService.add_markdown_to_case([mock_file], "r123", ""))
            self.assertIn("Embedding/indexing failed", "".join(updates8))

        # Cleanup source dir
        shutil.rmtree(src_dir)


if __name__ == "__main__":
    unittest.main()
