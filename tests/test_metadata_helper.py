import unittest
from unittest.mock import patch, MagicMock
from rag.metadata_helper import get_case_metadata

class TestMetadataHelper(unittest.TestCase):

    @patch("rag.metadata_helper.get_connection")
    def test_get_case_metadata_success(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database responses
        # 1. Patient names
        mock_cur.fetchall.side_effect = [
            [("Francis VAN ROSSUM",), ("Francis Van Rossum",), ("Mr. Francis",)],
            [("Re: Francis VAN ROSSUM DOB: 28.11.1971\nInjury: traumatic supraspinatus tear\n",),
             ("Re: MR. Francis Van Rossum DOB 28/11/1971\nDiagnosis: right shoulder pain",)]
        ]

        meta = get_case_metadata("run123")
        self.assertEqual(meta["names"], ["Francis Van Rossum"])
        self.assertEqual(meta["dob"], "28/11/1971")
        self.assertEqual(meta["injuries"], ["Traumatic supraspinatus tear", "Right shoulder pain"])

    @patch("rag.metadata_helper.get_connection")
    def test_get_case_metadata_fallback_on_error(self, mock_get_conn):
        # Trigger connection failure
        mock_get_conn.side_effect = Exception("Database connection failed")

        meta = get_case_metadata("run123")
        self.assertEqual(meta["names"], [])
        self.assertEqual(meta["dob"], "—")
        self.assertIn("Error loading metadata: Database connection failed", meta["injuries"][0])

if __name__ == "__main__":
    unittest.main()
