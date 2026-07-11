"""
Extended unit tests targeting remaining code coverage gaps in chunker.py and rag_infra_manager.py.
"""

import os
import io
import unittest
from unittest.mock import patch, MagicMock

# Prevent system operations during import
os.environ["TESTING"] = "true"

from rag import chunker
import rag_infra_manager


class TestRAGExtended(unittest.TestCase):

    # ── Chunker Splitting Coverage ──────────────────────────────

    def test_split_section_into_chunks_long(self):
        # Create a text longer than 800 characters to trigger splitting logic
        paragraph_1 = "This is the first paragraph. " * 30 # ~900 chars
        paragraph_2 = "This is the second paragraph. " * 30 # ~900 chars
        long_text = f"{paragraph_1}\n\n{paragraph_2}"
        
        chunks = chunker.chunk_document(
            markdown_text=long_text,
            doc_id="doc_long",
            run_id="run_long",
            max_chunk_size=800,
            chunk_overlap=100
        )
        # Verify splitting happened
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0]["doc_id"], "doc_long")

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_chunk_documents_from_run_empty(self, mock_listdir, mock_exists):
        mock_exists.return_value = False
        res = chunker.chunk_documents_from_run("/mock/run", "run_id")
        self.assertEqual(res, {})

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_chunk_documents_from_run_success(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        
        # Mock folder listing
        def listdir_side_effect(path):
            if "results" in path:
                return ["output.jsonl", "ignored_file.txt"]  # non-jsonl file included
            if "inputs" in path:
                return ["report.md"]
            return []
        mock_listdir.side_effect = listdir_side_effect

        # Path-aware open mock for jsonl and md
        def open_mock(filename, mode="r", *args, **kwargs):
            if "jsonl" in filename:
                return io.StringIO(
                    '{"metadata": {"Source-File": "inputs/report.pdf"}, "attributes": {"pdf_page_numbers": [[0, 100, 1]]}}\n'
                    '\n' # empty line
                )
            if "report.md" in filename:
                return io.StringIO("# Report\nPatient Name: Francis\nDear Dr. Eugene Ek,\nhello.")
            return io.StringIO("")

        with patch("builtins.open", side_effect=open_mock):
            res = chunker.chunk_documents_from_run("/mock/run", "run_id")
            self.assertTrue(len(res) > 0)
            doc_id = list(res.keys())[0]
            self.assertEqual(res[doc_id]["md_file"], "report.md")

    def test_chunker_additional_coverage(self):
        # 1. 2-digit year conversion logic
        self.assertEqual(chunker._parse_date("Cons 12.02.18"), "2018-02-12")
        self.assertEqual(chunker._parse_date("Cons 12.02.88"), "1988-02-12")

        # 2. Match groups raising ValueError
        mock_match = MagicMock()
        mock_match.groups.return_value = ("abc", "def", "ghi")
        mock_pattern = MagicMock()
        mock_pattern.search.return_value = mock_match
        with patch.object(chunker, "DATE_PATTERNS", [mock_pattern]):
            self.assertIsNone(chunker._parse_date("trigger exception"))

        # 3. _find_page_for_position edge cases
        page = chunker._find_page_for_position(char_pos=1000, page_ranges=[[0, 10], [0, 50, 1]])
        self.assertIsNone(page)

        # 4. Paragraph splitting with no periods and only newlines
        long_no_periods = "A" * 700 + "\n" + "B" * 200
        chunks = chunker.chunk_document(long_no_periods, "doc1", "run1", max_chunk_size=800)
        self.assertTrue(len(chunks) > 0)

        # 5. Empty markdown_text
        self.assertEqual(chunker.chunk_document("", "doc1", "run1"), [])

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_chunk_documents_from_run_error_branches(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        
        # Mock file listing with non-md file and md file
        def listdir_side_effect(path):
            if "results" in path:
                return ["output.jsonl"]
            if "inputs" in path:
                return ["report.md", "image.png"] # png is skipped
            return []
        mock_listdir.side_effect = listdir_side_effect

        def open_mock(filename, mode="r", *args, **kwargs):
            if "jsonl" in filename:
                # Returns empty lines, metadata missing, and invalid line to raise JSONDecodeError
                return io.StringIO("\n\n{}\nmalformed_json\n")
            if "report.md" in filename:
                raise IOError("Read failure")
            return io.StringIO("")

        with patch("builtins.open", side_effect=open_mock):
            res = chunker.chunk_documents_from_run("/mock/run", "run_id")
            # Should run successfully catching exceptions and returning empty dict
            self.assertEqual(res, {})

    # ── Chunker 100% Statement and Branch Coverage Additions ──

    def test_parse_date_out_of_bounds(self):
        # 1. Day/Month check (day = 45, month = 2) -> should fail (202->193 branch)
        self.assertIsNone(chunker._parse_date("Cons 45.02.18"))
        # 2. Named month out of bounds (month = 0 or day = 45) -> (216->193 branch)
        self.assertIsNone(chunker._parse_date("Cons 45 Feb 2018"))
        self.assertIsNone(chunker._parse_date("Cons 12 NotAMonth 2018"))
        # 3. Invalid named month variant format
        self.assertIsNone(chunker._parse_date("Cons NotAMonth 45 2018"))

    def test_letter_boundary_patterns_at_zero(self):
        # Letter boundary at index 0 (pos = 0)
        # Should not call boundary_positions.add(pos) inside 'if pos > 0'
        # but 0 is already added at the beginning.
        text = "Dear Dr. Ek,\nThis is section 1."
        sections = chunker._split_into_sections(text)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][2], text)

    def test_letter_boundary_patterns_greater_than_200(self):
        # Multiple boundaries separated by >= 200 chars to cover filtered_positions append
        text = "Dear Dr. Ek,\n" + "A" * 250 + "\nDear Dr. Ek,\n" + "B" * 250
        sections = chunker._split_into_sections(text)
        self.assertEqual(len(sections), 2)

    def test_letter_boundary_empty_section(self):
        # Adjacent boundaries leading to empty section_text.strip() (322->319 branch)
        text = "Dear Dr. Ek,\nDear Dr. Ek,\nThis is valid."
        sections = chunker._split_into_sections(text)
        # Should only return one non-empty section
        self.assertEqual(len(sections), 1)

    def test_split_section_into_chunks_paragraph_formatting(self):
        # Consecutive newlines causing empty paragraphs (line 361 continue)
        text = "Paragraph 1\n\n\n\nParagraph 2"
        chunks = chunker._split_section_into_chunks(text, 0, max_chunk_size=800)
        self.assertEqual(len(chunks), 1)

    def test_split_section_into_chunks_natural_splits(self):
        # Test paragraph splitting when para is long but splits at newline/sentence ends
        # Case: no sentence end '. ', but has newline
        para_long = "A" * 600 + "\n" + "B" * 300
        chunks = chunker._split_section_into_chunks(para_long, 0, max_chunk_size=800)
        self.assertEqual(len(chunks), 2)
        
        # Case: no sentence end '. ' and no newline (splits at max_chunk_size)
        para_continuous = "A" * 1000
        chunks2 = chunker._split_section_into_chunks(para_continuous, 0, max_chunk_size=800)
        self.assertEqual(len(chunks2), 2)

    def test_chunk_document_empty_chunk_filtering(self):
        # Case: chunk_text.strip() is empty inside loop (line 468 continue)
        # We trigger this by having a chunk piece with only spaces.
        # We can mock _split_section_into_chunks to return a list with whitespace chunk
        with patch("rag.chunker._split_section_into_chunks", return_value=[(0, 5, "   ")]):
            chunks = chunker.chunk_document("Dear Dr. Ek,\nHello.", "doc1", "run1")
            self.assertEqual(len(chunks), 0)

    @patch("os.path.exists")
    def test_chunk_documents_from_run_no_results_dir(self, mock_exists):
        # md_inputs_dir exists, but results_dir does not exist (532->553 branch)
        def exists_side_effect(path):
            if "markdown" in path:
                return True
            return False
        mock_exists.side_effect = exists_side_effect

        with patch("os.listdir", return_value=[]):
            res = chunker.chunk_documents_from_run("/mock/run", "run_id")
            self.assertEqual(res, {})

    # ── Infrastructure Status & Lifecycle coverage ──────────────

    @patch("subprocess.run")
    def test_start_rag_infrastructure(self, mock_run):
        # Mock docker-compose up successful run
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = rag_infra_manager.start_rag_infrastructure()
        self.assertTrue(success)
        self.assertTrue("successfully" in msg)

    @patch("subprocess.run")
    def test_start_rag_infrastructure_failure(self, mock_run):
        # Mock docker-compose up failure
        mock_run.return_value = MagicMock(returncode=1, stderr=b"Compose error")
        success, msg = rag_infra_manager.start_rag_infrastructure()
        self.assertFalse(success)
        self.assertTrue("Failed to start" in msg)

    @patch("subprocess.run")
    def test_stop_rag_infrastructure(self, mock_run):
        # Mock docker-compose down successful run
        mock_run.return_value = MagicMock(returncode=0)
        success, msg = rag_infra_manager.stop_rag_infrastructure()
        self.assertTrue(success)
        self.assertTrue("stopped" in msg)

    def test_parse_date_named_month_invalid_day(self):
        # 216->193 branch (invalid day like 45)
        self.assertIsNone(chunker._parse_date("February 45, 2018"))

    def test_parse_date_custom_pattern_groups(self):
        # 204->193 branch (len(groups) != 3)
        import re
        original_patterns = chunker.DATE_PATTERNS
        chunker.DATE_PATTERNS = original_patterns + [re.compile(r'\b(Jan|Feb)\s+(\d{4})\b')]
        try:
            # Matches "Feb 2018", groups is ("Feb", "2018") -> len is 2.
            # Bypasses both len == 3 blocks, loop continues and returns None.
            self.assertIsNone(chunker._parse_date("Feb 2018"))
        finally:
            chunker.DATE_PATTERNS = original_patterns

    def test_split_into_sections_empty_section_text(self):
        # 322->319 branch
        import re
        original_boundary_patterns = chunker.LETTER_BOUNDARY_PATTERNS
        chunker.LETTER_BOUNDARY_PATTERNS = [re.compile(r' ')]
        try:
            # 300 spaces. Every space matches the boundary pattern.
            # Positions are filtered to be >= 200 apart: [0, 200].
            # Loop start=0, end=200 has 200 spaces (stripped is empty -> skips).
            # Loop start=200, end=300 has 100 spaces (stripped is empty -> skips).
            sections = chunker._split_into_sections(" " * 300)
            self.assertEqual(sections, [])
        finally:
            chunker.LETTER_BOUNDARY_PATTERNS = original_boundary_patterns

    def test_split_section_into_chunks_word_fallback(self):
        # Paragraph with only spaces but length > 800 (triggers 361 if not words: continue)
        para_spaces = " " * 900
        chunks1 = chunker._split_section_into_chunks(para_spaces, 0, max_chunk_size=800)
        self.assertEqual(chunks1, [])

        # Paragraph with no natural splits but length > 800 (triggers 364-367 loop fallback)
        para_words = "word " * 180 # length 900
        chunks2 = chunker._split_section_into_chunks(para_words, 0, max_chunk_size=800)
        self.assertTrue(len(chunks2) > 0)

    def test_split_section_into_chunks_para_append(self):
        # Trigger line 365 (current_chunk += "\n\n" + para)
        text = "Para1\n\nPara2\n\n" + "Para3" * 15 # total length > 80
        chunks = chunker._split_section_into_chunks(text, 0, max_chunk_size=40)
        self.assertTrue(len(chunks) > 0)

    def test_split_section_into_chunks_empty_chunk_and_remaining(self):
        # 393->397 (empty chunk_text)
        para_empty_start = " " * 850 + "A"
        chunks = chunker._split_section_into_chunks(para_empty_start, 0, max_chunk_size=800)
        self.assertTrue(len(chunks) > 0)

        # 400->407 (empty remaining current_chunk.strip())
        para_empty_end = "A" + " " * 850
        chunks2 = chunker._split_section_into_chunks(para_empty_end, 0, max_chunk_size=800)
        self.assertTrue(len(chunks2) > 0)


if __name__ == "__main__":
    unittest.main()
