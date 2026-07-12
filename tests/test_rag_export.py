"""
Unit tests for the OLMOCR RAG export subsystem.

Tests:
- Case name extraction and cleaning
- History-to-pairs parsing
- Markdown export generation and contents
- Plain text export generation and markdown stripping
- Timeline CSV export parsing and extraction
"""

import os
import unittest
import csv

# Prevent system operations during import
os.environ["TESTING"] = "true"

from rag_export import (
    EXPORT_DIR,
    _extract_case_name,
    _history_to_pairs,
    _make_export_filename,
    export_chat_markdown,
    export_chat_text,
    export_timeline_csv,
)


class TestRAGExport(unittest.TestCase):

    def setUp(self):
        # We ensure the export directory exists
        os.makedirs(EXPORT_DIR, exist_ok=True)
        self.created_files = []

    def tearDown(self):
        # Clean up any files created during the test
        for path in self.created_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def track_file(self, path):
        if path:
            self.created_files.append(path)
        return path

    def test_extract_case_name(self):
        self.assertEqual(_extract_case_name("All Cases"), "all_cases")
        self.assertEqual(_extract_case_name(""), "all_cases")
        self.assertEqual(_extract_case_name(None), "all_cases")
        self.assertEqual(_extract_case_name("workspace/run_case123"), "case123")
        self.assertEqual(_extract_case_name("case@name#special!"), "case_name_special_")
        # Long name truncation
        long_name = "a" * 100
        truncated = _extract_case_name(long_name)
        self.assertEqual(len(truncated), 60)
        self.assertEqual(truncated, "a" * 60)

    def test_make_export_filename(self):
        filename = _make_export_filename("test", "md")
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".md"))

    def test_history_to_pairs_empty(self):
        self.assertEqual(_history_to_pairs(None), [])
        self.assertEqual(_history_to_pairs([]), [])

    def test_history_to_pairs_dicts(self):
        history = [
            {"role": "user", "content": "What is the diagnosis?"},
            {"role": "assistant", "content": "The diagnosis is a SLAP tear."},
        ]
        pairs = _history_to_pairs(history)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], ("user", "What is the diagnosis?"))
        self.assertEqual(pairs[1], ("assistant", "The diagnosis is a SLAP tear."))

    def test_history_to_pairs_tuples_lists(self):
        history = [
            ("Hello", None),
            (None, "Hi there"),
            ["Question", "Answer"],
        ]
        pairs = _history_to_pairs(history)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], ("user", "Hello"))
        self.assertEqual(pairs[1], ("assistant", "Hi there"))
        self.assertEqual(pairs[2], ("user", "Question"))

    def test_history_to_pairs_malformed(self):
        history = [
            "not a dict or list",
            {"role": "user"},  # missing content
            [123],             # too short list
        ]
        pairs = _history_to_pairs(history)
        # It should ignore malformed entries or handle them safely
        # {"role": "user"} returns ("user", "")
        # The string and [123] are skipped
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("user", ""))

    def test_export_chat_markdown_empty(self):
        self.assertIsNone(export_chat_markdown(None))
        self.assertIsNone(export_chat_markdown([]))

    def test_export_chat_markdown_valid(self):
        history = [
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "I am helping."},
        ]
        path = self.track_file(export_chat_markdown(history, mode="timeline", active_case="case1"))
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("# RAG Analysis Export", content)
        self.assertIn("**Mode:** timeline", content)
        self.assertIn("## 👤 User Query", content)
        self.assertIn("Help me.", content)
        self.assertIn("## 🤖 Analysis Response", content)
        self.assertIn("I am helping.", content)

    def test_export_chat_text_empty(self):
        self.assertIsNone(export_chat_text(None))
        self.assertIsNone(export_chat_text([]))

    def test_export_chat_text_valid(self):
        history = [
            {"role": "user", "content": "Help me."},
            {
                "role": "assistant",
                "content": "# Heading 1\nThis is **bold** and *italic* text. Check [Google](https://google.com)."
            },
        ]
        path = self.track_file(export_chat_text(history, mode="timeline", active_case="case2"))
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("RAG ANALYSIS EXPORT", content)
        self.assertIn("USER QUERY:", content)
        self.assertIn("ANALYSIS RESPONSE:", content)
        # Check formatting is stripped
        self.assertIn("Heading 1", content)
        self.assertNotIn("# Heading 1", content)
        self.assertIn("This is bold and italic text.", content)
        self.assertNotIn("**bold**", content)
        self.assertIn("Check Google.", content)
        self.assertNotIn("[Google](https://google.com)", content)

    def test_export_timeline_csv_empty(self):
        self.assertIsNone(export_timeline_csv(None))
        self.assertIsNone(export_timeline_csv([]))

    def test_export_timeline_csv_no_tables(self):
        history = [
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "No table here."},
        ]
        path = self.track_file(export_timeline_csv(history))
        self.assertIsNone(path)

    def test_export_timeline_csv_valid_table(self):
        table_markdown = """
Here is the timeline:

| Date | Event | Provider |
|---|---|---|
| 2020-01-01 | Injury | Employer |
| 2020-01-02 | Surgery | Dr Smith |

End of timeline.
"""
        history = [
            {"role": "user", "content": "Give me a table"},
            {"role": "assistant", "content": table_markdown},
        ]
        path = self.track_file(export_timeline_csv(history, active_case="case3"))
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

        # Read CSV file and verify content
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)

        self.assertEqual(len(rows), 3)  # Header + 2 data rows
        self.assertEqual(rows[0], ["Date", "Event", "Provider"])
        self.assertEqual(rows[1], ["2020-01-01", "Injury", "Employer"])
        self.assertEqual(rows[2], ["2020-01-02", "Surgery", "Dr Smith"])
