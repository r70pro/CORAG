"""
Unit tests for pdf_manager.py.
"""

import os
import io
import unittest
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image

# Prevent system operations during import
os.environ["TESTING"] = "true"

import pdf_manager
import state


class TestPdfManager(unittest.TestCase):

    def test_pil_to_base64(self):
        # Create a tiny mock image
        img = Image.new("RGB", (10, 10), color="red")
        b64 = pdf_manager.pil_to_base64(img)
        self.assertTrue(isinstance(b64, str))
        self.assertTrue(b64.startswith("data:image/png;base64,"))
        self.assertEqual(pdf_manager.pil_to_base64(None), "")

    @patch("os.path.exists")
    @patch("pypdfium2.PdfDocument")
    def test_render_pdf_page(self, mock_pdf_doc, mock_exists):
        mock_exists.return_value = True
        
        # Mock document render page
        mock_doc = mock_pdf_doc.return_value
        mock_doc.__len__.return_value = 5
        mock_page = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_bitmap = mock_page.render.return_value
        mock_pil = mock_bitmap.to_pil.return_value
        
        res = pdf_manager.render_pdf_page("/mock/path.pdf", 1)
        self.assertEqual(res, mock_pil)

        # Out of bounds page check
        self.assertIsNone(pdf_manager.render_pdf_page("/mock/path.pdf", 10))

    @patch("os.path.exists")
    def test_get_page_mapping_and_pdf_path_no_run(self, mock_exists):
        mock_exists.return_value = False
        path, total_pages, page_ranges = pdf_manager.get_page_mapping_and_pdf_path("doc.md", "run1")
        self.assertIsNone(path)
        self.assertEqual(total_pages, 0)
        self.assertEqual(page_ranges, [])

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("pdf_manager.PdfReader")
    def test_get_page_mapping_and_pdf_path_with_run(self, mock_reader, mock_listdir, mock_exists):
        # Return True for exists check on files and folders
        mock_exists.return_value = True
        mock_listdir.return_value = ["output_123.jsonl"]
        
        # Mock PdfReader instance
        mock_reader_instance = mock_reader.return_value
        mock_reader_instance.pages = [1, 2] # 2 pages
        
        # Mock active_runs
        state.active_runs["run1"] = {"run_dir": "/tmp/run"}
        
        # Path-aware open mock
        def open_mock(filename, mode="r", *args, **kwargs):
            if "jsonl" in filename:
                return io.StringIO(
                    '{"metadata": {"Source-File": "inputs/doc.pdf"}, "attributes": {"pdf_page_numbers": [[0, 100, 1]]}}\n'
                )
            return io.BytesIO(b"%PDF-1.4 dummy contents")

        with patch("builtins.open", side_effect=open_mock):
            path, total_pages, page_ranges = pdf_manager.get_page_mapping_and_pdf_path("doc.md", "run1")
            self.assertEqual(path, "/tmp/run/inputs/doc.pdf")
            self.assertEqual(total_pages, 2)
            self.assertEqual(page_ranges, [[0, 100, 1]])
        
        state.active_runs.clear()

    def test_get_markdown_for_page(self):
        # Case 1: no markdown
        self.assertEqual(pdf_manager.get_markdown_for_page("", [], 1), "")

        # Case 2: empty page_ranges
        self.assertEqual(pdf_manager.get_markdown_for_page("full markdown text", [], 1), "full markdown text")

        # Case 3: match range
        full_md = "0123456789abcdefghij"
        ranges = [[0, 10, 1], [10, 20, 2]]
        self.assertEqual(pdf_manager.get_markdown_for_page(full_md, ranges, 1), "0123456789")
        self.assertEqual(pdf_manager.get_markdown_for_page(full_md, ranges, 2), "abcdefghij")
        self.assertEqual(pdf_manager.get_markdown_for_page(full_md, ranges, 3), "")

    @patch("pdf_manager.get_page_mapping_and_pdf_path")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="hello markdown text")
    def test_on_file_selected(self, mock_file, mock_exists, mock_mapping):
        mock_mapping.return_value = ("/tmp/doc.pdf", 2, [[0, 10, 1]])
        mock_exists.return_value = True
        
        state.active_runs["run1"] = {"run_dir": "/tmp/run"}
        
        res = pdf_manager.on_file_selected("doc.md", "run1")
        # Returns (pdf_path, total_pages, page_ranges, full_markdown, gr.update(...), pdf_path)
        self.assertEqual(res[0], "/tmp/doc.pdf")
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], [[0, 10, 1]])
        self.assertEqual(res[3], "hello markdown text")
        self.assertEqual(res[5], "/tmp/doc.pdf")
        
        state.active_runs.clear()

    @patch("pdf_manager.render_pdf_page")
    @patch("pdf_manager.pil_to_base64")
    @patch("os.path.exists")
    def test_update_view(self, mock_exists, mock_b64, mock_render):
        mock_exists.return_value = True
        mock_render.return_value = MagicMock()
        mock_b64.return_value = "data:image/png;base64,123"

        # Case 1: no selected file
        res = pdf_manager.update_view("", "Page-by-Page", 1, "/tmp/doc.pdf", 2, [[0, 10, 1]], "markdown text")
        self.assertTrue("Select a processed document" in res[0])

        # Case 2: Page-by-Page with rendering
        res = pdf_manager.update_view("doc.md", "Page-by-Page", 1, "/tmp/doc.pdf", 2, [[0, 10, 1]], "markdown text")
        self.assertTrue("data:image/png;base64,123" in res[0])
        self.assertTrue("markdown" in res[2])

        # Case 3: Full Document
        res = pdf_manager.update_view("doc.md", "Full Document", 1, "/tmp/doc.pdf", 2, [[0, 10, 1]], "markdown text")
        self.assertTrue("iframe" in res[0])
        self.assertEqual(res[2], "markdown text")

    def test_is_safe_filename(self):
        self.assertTrue(pdf_manager.is_safe_filename("document.md"))
        self.assertTrue(pdf_manager.is_safe_filename("0_doc.pdf"))
        self.assertFalse(pdf_manager.is_safe_filename(""))
        self.assertFalse(pdf_manager.is_safe_filename("../etc/passwd"))
        self.assertFalse(pdf_manager.is_safe_filename("inputs/../../etc/passwd"))
        self.assertFalse(pdf_manager.is_safe_filename("/etc/passwd"))
        self.assertFalse(pdf_manager.is_safe_filename("c:\\windows\\win.ini"))

    def test_load_markdown_content_traversal(self):
        res = pdf_manager.load_markdown_content("../../../etc/passwd", "run1")
        self.assertEqual(res[0], "Invalid file path.")
        self.assertEqual(res[1], "Invalid file path.")
        self.assertIsNone(res[2])

    def test_get_page_mapping_and_pdf_path_traversal(self):
        path, total_pages, page_ranges = pdf_manager.get_page_mapping_and_pdf_path("../../../etc/passwd", "run1")
        self.assertIsNone(path)
        self.assertEqual(total_pages, 0)
        self.assertEqual(page_ranges, [])

    def test_on_file_selected_traversal(self):
        res = pdf_manager.on_file_selected("../../../etc/passwd", "run1")
        self.assertEqual(res[0], "")
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], [])
        self.assertEqual(res[3], "Invalid file path.")


if __name__ == "__main__":
    unittest.main()
