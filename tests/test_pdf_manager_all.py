"""
Comprehensive unit tests for pdf_manager.py targeting 100% statement and branch coverage.
"""

import os
import unittest
import tempfile
import zipfile
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image

# Prevent system operations during import
os.environ["TESTING"] = "true"

import process_state
import pdf_manager as pm


class TestPDFManagerAll(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmp_dir, "run_case")
        os.makedirs(self.run_dir)
        self.workspace_patcher = patch("pdf_manager.WORKSPACE_DIR", self.tmp_dir)
        self.workspace_patcher.start()
        process_state.active_runs.clear()

    def tearDown(self):
        self.workspace_patcher.stop()
        shutil.rmtree(self.tmp_dir)
        process_state.active_runs.clear()

    def test_make_zip(self):
        # Create dummy MD files
        md_dir = os.path.join(self.tmp_dir, "markdown")
        os.makedirs(md_dir)
        with open(os.path.join(md_dir, "test1.md"), "w", encoding="utf-8") as f:
            f.write("# Hello")
        with open(os.path.join(md_dir, "test2.txt"), "w", encoding="utf-8") as f:
            f.write("text")

        zip_path = os.path.join(self.tmp_dir, "output.zip")
        pm.make_zip(md_dir, zip_path)
        
        self.assertTrue(os.path.exists(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("test1.md", namelist)
            self.assertNotIn("test2.txt", namelist)

    def test_load_markdown_content_edges(self):
        # 1. Empty arguments
        self.assertEqual(pm.load_markdown_content("", ""), ("", "", None))

        # 2. Missing run_id in state
        self.assertEqual(pm.load_markdown_content("doc.md", "run1")[0], "Run info not found.")

        # 3. File not found
        process_state.active_runs["run1"] = {"run_dir": self.run_dir}
        self.assertEqual(pm.load_markdown_content("missing.md", "run1")[0], "File not found.")

        # 4. File read exception
        inputs_dir = os.path.join(self.run_dir, "markdown", "inputs")
        os.makedirs(inputs_dir)
        file_path = os.path.join(inputs_dir, "bad.md")
        # Create a directory to force permission/read error
        os.makedirs(file_path)
        content, _, _ = pm.load_markdown_content("bad.md", "run1")
        self.assertEqual(content, "File not found.")

        # 5. Success
        good_path = os.path.join(inputs_dir, "good.md")
        os.rmdir(file_path)
        with open(good_path, "w", encoding="utf-8") as f:
            f.write("markdown text")
        content, _, path = pm.load_markdown_content("good.md", "run1")
        self.assertEqual(content, "markdown text")
        self.assertEqual(path, good_path)

    def test_pil_to_base64(self):
        # 1. None img
        self.assertEqual(pm.pil_to_base64(None), "")
        
        # 2. Valid image
        img = Image.new("RGB", (10, 10), color="red")
        b64 = pm.pil_to_base64(img)
        self.assertTrue(b64.startswith("data:image/png;base64,"))

    @patch("pdf_manager.pdfium.PdfDocument")
    def test_render_pdf_page(self, mock_doc_cls):
        # 1. None/missing pdf
        self.assertIsNone(pm.render_pdf_page(None, 1))
        self.assertIsNone(pm.render_pdf_page("missing.pdf", 1))

        # Create dummy file to pass path check
        dummy_pdf = os.path.join(self.tmp_dir, "dummy.pdf")
        with open(dummy_pdf, "wb") as f:
            f.write(b"pdf data")

        # 2. Out of bounds page
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 5
        mock_doc_cls.return_value = mock_doc
        self.assertIsNone(pm.render_pdf_page(dummy_pdf, 0))
        self.assertIsNone(pm.render_pdf_page(dummy_pdf, 6))

        # 3. Success page render
        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = "pil_image"
        mock_page.render.return_value = mock_bitmap
        mock_doc.__getitem__.return_value = mock_page
        
        self.assertEqual(pm.render_pdf_page(dummy_pdf, 2), "pil_image")

        # 4. Exception fallback
        mock_doc_cls.side_effect = Exception("Doc load failed")
        self.assertIsNone(pm.render_pdf_page(dummy_pdf, 2))

    @patch("pdf_manager.PdfReader")
    def test_get_page_mapping_and_pdf_path(self, mock_reader):
        # 1. Empty arguments
        self.assertEqual(pm.get_page_mapping_and_pdf_path("", ""), (None, 0, []))

        # 2. Missing run_id in state
        self.assertEqual(pm.get_page_mapping_and_pdf_path("doc.md", "run1"), (None, 0, []))

        # 3. PDF file not found
        process_state.active_runs["run1"] = {"run_dir": self.run_dir}
        self.assertEqual(pm.get_page_mapping_and_pdf_path("doc.md", "run1"), (None, 0, []))

        # Create dummy PDF file
        inputs_dir = os.path.join(self.run_dir, "inputs")
        os.makedirs(inputs_dir)
        pdf_path = os.path.join(inputs_dir, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"pdf data")

        # 4. PdfReader exception
        mock_reader.side_effect = Exception("Corrupt pdf")
        self.assertEqual(pm.get_page_mapping_and_pdf_path("doc.md", "run1"), (pdf_path, 0, []))

        # 5. Success with jsonl page ranges parsing
        mock_reader.side_effect = None
        mock_reader_inst = MagicMock()
        mock_reader_inst.pages = [MagicMock(), MagicMock()]
        mock_reader.return_value = mock_reader_inst

        results_dir = os.path.join(self.run_dir, "results")
        os.makedirs(results_dir)
        
        # Write test jsonl files
        jsonl_path = os.path.join(results_dir, "res.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"metadata": {"Source-File": "inputs/doc.pdf"}, "attributes": {"pdf_page_numbers": [[0, 100, 1]]}}\n')
            f.write('{"metadata": {"Source-File": "inputs/other.pdf"}, "attributes": {"pdf_page_numbers": [[0, 100, 1]]}}\n')
            f.write('\n') # empty line
            f.write('{bad json}\n') # invalid json

        path, total, ranges = pm.get_page_mapping_and_pdf_path("doc.md", "run1")
        self.assertEqual(path, pdf_path)
        self.assertEqual(total, 2)
        self.assertEqual(ranges, [[0, 100, 1]])

    def test_get_markdown_for_page(self):
        # 1. Empty markdown
        self.assertEqual(pm.get_markdown_for_page("", [], 1), "")

        # 2. Empty page ranges
        self.assertEqual(pm.get_markdown_for_page("full text", [], 1), "full text")

        # 3. Match range
        ranges = [[0, 4, 1], [5, 10, 2]]
        self.assertEqual(pm.get_markdown_for_page("01234567890", ranges, 2), "56789")
        
        # 4. No match range
        self.assertEqual(pm.get_markdown_for_page("01234567890", ranges, 3), "")

    @patch("pdf_manager.get_page_mapping_and_pdf_path")
    def test_on_file_selected(self, mock_mapping):
        # 1. Empty
        self.assertEqual(pm.on_file_selected("", ""), ("", 0, [], "", {"maximum": 2, "value": 1, "interactive": False}, None))

        # 2. Success path
        mock_mapping.return_value = ("pdf_path", 5, [[0, 10, 1]])
        process_state.active_runs["run1"] = {"run_dir": self.run_dir}
        
        # Create dummy MD file
        inputs_dir = os.path.join(self.run_dir, "markdown", "inputs")
        os.makedirs(inputs_dir)
        with open(os.path.join(inputs_dir, "doc.md"), "w", encoding="utf-8") as f:
            f.write("page 1 text")

        path, total, ranges, content, update, preview_path = pm.on_file_selected("doc.md", "run1")
        self.assertEqual(path, "pdf_path")
        self.assertEqual(total, 5)
        self.assertEqual(ranges, [[0, 10, 1]])
        self.assertEqual(content, "page 1 text")
        self.assertEqual(preview_path, os.path.join(self.run_dir, "markdown", "inputs", "doc.md"))

        # 3. File reading exception
        with patch("pathlib.Path.read_text", side_effect=OSError("Read fail")):
            _, _, _, content2, _, _ = pm.on_file_selected("doc.md", "run1")
            self.assertIn("Error reading file", content2)

    @patch("pdf_manager.render_pdf_page")
    def test_update_view(self, mock_render):
        # 1. No selected file
        self.assertEqual(pm.update_view("", "Full Document", 1, "", 0, [], "")[2], "Select a processed document to preview.")

        # 2. Full Document view mode - PDF exists
        pdf_path = os.path.join(self.tmp_dir, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"pdf data")
        
        pdf_html, md_html, md_text = pm.update_view("doc.md", "Full Document", 1, pdf_path, 2, [], "# Head")
        self.assertIn("iframe", pdf_html)
        self.assertIn("sync-scroll-target", md_html)
        self.assertEqual(md_text, "# Head")

        # 3. Full Document view mode - PDF missing
        pdf_html_missing, _, _ = pm.update_view("doc.md", "Full Document", 1, "missing.pdf", 2, [], "# Head")
        self.assertIn("PDF file not found", pdf_html_missing)

        # 4. Page by Page view mode - render success
        mock_img = Image.new("RGB", (10, 10), color="blue")
        mock_render.return_value = mock_img
        
        pdf_html_page, _, _ = pm.update_view("doc.md", "Page by Page", 1, pdf_path, 2, [[0, 5, 1]], "# Page")
        self.assertIn("img src", pdf_html_page)

        # 5. Page by Page view mode - render failure
        mock_render.return_value = None
        pdf_html_fail, _, _ = pm.update_view("doc.md", "Page by Page", 1, pdf_path, 2, [[0, 5, 1]], "# Page")
        self.assertIn("Failed to render page", pdf_html_fail)

        # 6. Page by Page view mode - PDF missing
        pdf_html_page_missing, _, _ = pm.update_view("doc.md", "Page by Page", 1, "missing.pdf", 2, [[0, 5, 1]], "# Page")
        self.assertIn("PDF file not found", pdf_html_page_missing)

    @patch("pdf_manager.PdfReader")
    def test_get_page_mapping_and_pdf_path_edge_cases(self, mock_reader):
        mock_reader_inst = MagicMock()
        mock_reader_inst.pages = [MagicMock()]
        mock_reader.return_value = mock_reader_inst

        process_state.active_runs["run_edge"] = {"run_dir": self.run_dir}
        inputs_dir = os.path.join(self.run_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        pdf_path = os.path.join(inputs_dir, "doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"pdf data")

        # 1. results_dir exists but is empty (87->108 branch)
        results_dir = os.path.join(self.run_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        _, _, ranges1 = pm.get_page_mapping_and_pdf_path("doc.md", "run_edge")
        self.assertEqual(ranges1, [])

        # 2. file in results_dir does not end with .jsonl (88->87 branch)
        with open(os.path.join(results_dir, "ignored.txt"), "w") as f:
            f.write("text")
        _, _, ranges2 = pm.get_page_mapping_and_pdf_path("doc.md", "run_edge")
        self.assertEqual(ranges2, [])

        # 3. jsonl lines are empty or have mismatches (exits normally to line 105)
        jsonl_path = os.path.join(results_dir, "res.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            # - empty line (94)
            f.write("\n")
            # - mismatched source_file (98->92)
            f.write('{"metadata": {"Source-File": "inputs/mismatch.pdf"}, "attributes": {"pdf_page_numbers": [[0, 10, 1]]}}\n')

        _, _, ranges3 = pm.get_page_mapping_and_pdf_path("doc.md", "run_edge")
        self.assertEqual(ranges3, [])

        # 4. jsonl lines have invalid json (raises exception and exits to 103)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{bad_json}\n')

        _, _, ranges4 = pm.get_page_mapping_and_pdf_path("doc.md", "run_edge")
        self.assertEqual(ranges4, [])
        process_state.active_runs.clear()

    def test_get_markdown_for_page_edge(self):
        # range_info has length < 3 (line 117->116 branch)
        ranges = [[0, 10]] # length 2
        self.assertEqual(pm.get_markdown_for_page("markdown text", ranges, 1), "")

    @patch("pdf_manager.get_page_mapping_and_pdf_path")
    def test_on_file_selected_missing_run_or_file(self, mock_mapping):
        # 1. active run info missing (133->144 branch)
        mock_mapping.return_value = ("pdf_path", 5, [])
        res1 = pm.on_file_selected("doc.md", "missing_run")
        self.assertEqual(res1[3], "")

        # 2. active run dir exists but markdown file does not exist (136->144 branch)
        process_state.active_runs["run_missing_file"] = {"run_dir": self.run_dir}
        res2 = pm.on_file_selected("doc.md", "run_missing_file")
        self.assertEqual(res2[3], "")
        process_state.active_runs.clear()


if __name__ == "__main__":
    unittest.main()
