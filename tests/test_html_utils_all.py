"""
Comprehensive unit tests for html_utils.py targeting 100% statement and branch coverage.
"""

import unittest
import html_utils as hu


class TestHTMLUtilsAll(unittest.TestCase):

    def test_make_progress_bar_html_time_formatting(self):
        # 1. ETA: under 60s, elapsed: under 60s
        res = hu.make_progress_bar_html(completed=5, total=10, elapsed_secs=10)
        self.assertIn("10s remaining", res)
        self.assertIn("10s elapsed", res)

        # 2. ETA: under 3600s, elapsed: under 3600s
        res2 = hu.make_progress_bar_html(completed=1, total=10, elapsed_secs=60)
        self.assertIn("9m 0s remaining", res2)
        self.assertIn("1m 0s elapsed", res2)

        # 3. ETA: over 3600s, elapsed: over 3600s
        res3 = hu.make_progress_bar_html(completed=1, total=10, elapsed_secs=3600)
        self.assertIn("9h 0m remaining", res3)
        self.assertIn("1h 0m elapsed", res3)

        # 4. Completed >= total
        res4 = hu.make_progress_bar_html(completed=10, total=10, elapsed_secs=10)
        self.assertIn("Complete", res4)

        # 5. Total is 0
        res5 = hu.make_progress_bar_html(completed=0, total=0, elapsed_secs=0)
        self.assertIn("0/0 Pages", res5)

    def test_make_file_status_html(self):
        file_mapping = {0: "doc1.pdf", 1: "doc2.pdf", 2: "doc3.pdf"}
        file_page_counts = {0: 5, 1: 10}
        completed = {0}
        failed = {1}
        
        # Testing with failed status and completed status and default/pending status
        res = hu.make_file_status_html(file_mapping, file_page_counts, completed, failed)
        self.assertIn("✓ Done", res)
        self.assertIn("✗ Failed", res)
        self.assertIn("⏳ Pending", res)

        # Testing with failed_files_set = None fallback
        res2 = hu.make_file_status_html(file_mapping, file_page_counts, completed, None)
        self.assertIn("✓ Done", res2)

    def test_make_upload_manifest_html(self):
        file_mapping = {0: "small.pdf", 1: "med.pdf", 2: "large.pdf"}
        file_page_counts = {0: 1, 1: "unknown", 2: 10}
        file_sizes = {0: 500, 1: 5000, 2: 5000000}
        
        res = hu.make_upload_manifest_html(file_mapping, file_page_counts, file_sizes)
        self.assertIn("500 B", res)
        self.assertIn("4.9 KB", res)
        self.assertIn("4.8 MB", res)
        self.assertIn("Total (3 files)", res)

    def test_all_html_generators_escape_untrusted_text_and_attributes(self):
        payload = "\"><svg onload=alert(1)>"

        file_status = hu.make_file_status_html(
            {0: payload}, {0: payload}, completed_files_set=set()
        )
        manifest = hu.make_upload_manifest_html(
            {0: payload}, {0: payload}, {0: 1}
        )
        banner = hu.make_case_banner_html(payload)
        dashboard = hu.make_case_dashboard_html(
            [
                {
                    "run_id": payload,
                    "run_dir": payload,
                    "total_documents": payload,
                    "total_chunks": payload,
                    "unique_authors": payload,
                    "earliest_date": payload,
                }
            ],
            {
                payload: {
                    "names": [payload],
                    "dob": payload,
                    "injuries": [payload],
                }
            },
        )

        for generated in (file_status, manifest, banner, dashboard):
            self.assertNotIn("<svg", generated.lower())
            self.assertIn("&lt;svg", generated.lower())
        self.assertNotIn("onclick=", dashboard.lower())

    def test_structured_data_helpers(self):
        # get_progress_data
        prog = hu.get_progress_data(completed=5, total=10, elapsed_secs=10)
        self.assertEqual(prog["percentage"], 50)
        self.assertEqual(prog["eta_str"], "10s remaining")

        # get_file_status_data
        file_mapping = {0: "doc1.pdf", 1: "doc2.pdf"}
        file_page_counts = {0: 5, 1: 10}
        statuses = hu.get_file_status_data(file_mapping, file_page_counts, completed_files_set={0}, failed_files_set={1})
        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[0]["status"], "completed")
        self.assertEqual(statuses[1]["status"], "failed")

        # get_progress_data minutes & hours branches
        prog_m = hu.get_progress_data(completed=1, total=10, elapsed_secs=60)
        self.assertIn("m", prog_m["eta_str"])
        self.assertIn("m", prog_m["elapsed_str"])

        prog_h = hu.get_progress_data(completed=1, total=10, elapsed_secs=3600)
        self.assertIn("h", prog_h["eta_str"])
        self.assertIn("h", prog_h["elapsed_str"])

        prog_done = hu.get_progress_data(completed=10, total=10, elapsed_secs=10)
        self.assertEqual(prog_done["eta_str"], "Complete")

        # get_file_status_data defaults
        stat_default = hu.get_file_status_data({0: "a.pdf"}, {0: 1}, completed_files_set=set(), failed_files_set=None)
        self.assertEqual(stat_default[0]["status"], "pending")

        # get_case_dashboard_data with None cases_metadata
        runs_data = [{"run_id": "run_1", "run_dir": "/workspace/run_1", "total_documents": 2}]
        dash_none = hu.get_case_dashboard_data(runs_data, None)
        self.assertEqual(len(dash_none), 1)


if __name__ == "__main__":
    unittest.main()
