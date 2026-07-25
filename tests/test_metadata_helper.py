import unittest
from unittest.mock import patch, MagicMock
from rag.metadata_helper import get_case_metadata, get_all_cases_metadata

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

    @patch("rag.metadata_helper.get_connection")
    def test_get_all_cases_metadata_success(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock query results: first call fetchall (names), second call fetchall (texts)
        mock_cur.fetchall.side_effect = [
            [("run1", "Alice"), ("run2", "Bob")],
            [("run1", "DOB: 01/01/1990\nInjury: fracture"), ("run2", "DOB: 02/02/1991\nDiagnosis: sprain")]
        ]

        res = get_all_cases_metadata(["run1", "run2"])
        self.assertIn("run1", res)
        self.assertEqual(res["run1"]["names"], ["Alice"])
        self.assertEqual(res["run1"]["dob"], "01/01/1990")
        self.assertEqual(res["run1"]["injuries"], ["Fracture"])

        self.assertIn("run2", res)
        self.assertEqual(res["run2"]["names"], ["Bob"])
        self.assertEqual(res["run2"]["dob"], "02/02/1991")
        self.assertEqual(res["run2"]["injuries"], ["Sprain"])

    def test_get_all_cases_metadata_empty(self):
        res = get_all_cases_metadata([])
        self.assertEqual(res, {})

    @patch("rag.metadata_helper.get_connection")
    def test_get_all_cases_metadata_error(self, mock_get_conn):
        mock_get_conn.side_effect = Exception("DB error")
        res = get_all_cases_metadata(["run1"])
        self.assertIn("run1", res)
        self.assertEqual(res["run1"]["names"], [])
        self.assertEqual(res["run1"]["dob"], "—")
        self.assertIn("Error loading metadata: DB error", res["run1"]["injuries"][0])

    @patch("rag.metadata_helper.get_connection")
    def test_metadata_helper_edge_cases(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database responses to trigger all fallback branches in _build_metadata
        mock_cur.fetchall.side_effect = [
            [
                (None,),              # 33->34: empty/none name
                ("A",),               # 34: too short name
                ("Patient",),         # 36: ignore keyword
                ("Mr. John Doe",),    # Valid name Mr. John Doe -> John Doe
            ],
            [
                (None,),              # 49->50: empty text
                ("Re: John Doe\nNo DOB here\n",), # 52->55: DOB regex miss
                ("Injury: condition\n",),        # 59: ignore keyword in phrase ("condition")
                ("Injury: short\n",),            # 63: too short injury phrase (< 5 chars)
                ("Injury: " + "a"*100 + "\n",),   # 63: too long injury phrase (> 90 chars)
                ("Injury: supraspinatus tear\n",),
                ("Injury: supraspinatus tear\n",), # 64: duplicate injury
                ("Injury: our client has pain\n",), # 73: generic words ("our client") -> filtered out
            ]
        ]

        meta = get_case_metadata("run123")
        self.assertEqual(meta["names"], ["John Doe"])
        self.assertEqual(meta["dob"], "—")
        self.assertEqual(meta["injuries"], ["Supraspinatus tear"])

        # Test bulk metadata loader fallback (lines 152-153) where _build_metadata throws TypeError
        mock_cur.fetchall.side_effect = [
            [("run1", "Alice")],
            [("run1", "text")]
        ]
        with patch("rag.metadata_helper._build_metadata", side_effect=TypeError("mock error")):
            res = get_all_cases_metadata(["run1"])
            self.assertIn("run1", res)
            self.assertEqual(res["run1"]["names"], [])
            self.assertEqual(res["run1"]["dob"], "—")
            self.assertEqual(res["run1"]["injuries"], [])

    @patch("rag.metadata_helper.get_connection")
    def test_get_case_timeline(self, mock_get_conn):
        from rag.metadata_helper import get_case_timeline

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock rows: page_number, document_type, author, date_extracted, date_raw, text
        mock_cur.fetchall.return_value = [
            (1, "physio_notes", "Dr. Edwards", "2024-01-10", None, "Edwards treated patient reported right shoulder dislocation during accident. Ref No: 12345"),
            (2, "specialist_letter", None, None, "15/02/2024", "Ek Borbas reviewed Abdomen strain evaluation. Claim No: ABC-99"),
            (3, "imaging_scan", None, None, None, "10 Mar 2024 Camberwell SLAP tear and bankart lesion observed. Accession: ACC-777\n\nFull summary text goes here."),
            (4, "operation_surg", None, None, None, "12/04/2024 De Villiers performed biceps tenodesis surgery."),
            (5, "physio", "Camberwell", None, None, "15/05/2024 physiotherapy gym flexion exercises."),
            (6, "other_doc", "Dr. Smith", None, None, "20/06/2024 General report summary."),
            (7, "skip_doc", None, None, "1971-11-28", "Skip DOB matching text"),
            (8, "empty_doc", None, None, None, "")
        ]

        events = get_case_timeline("run123")
        self.assertGreater(len(events), 0)
        dates = [e["date"] for e in events]
        self.assertIn("2024-01-10", dates)
        self.assertIn("15/02/2024", dates)

        # Test empty return
        mock_cur.fetchall.return_value = []
        events_empty = get_case_timeline("run123")
        self.assertEqual(events_empty, [])

        # Test exception fallback
        mock_get_conn.side_effect = Exception("DB error")
        events_err = get_case_timeline("run123")
        self.assertEqual(events_err, [])

    @patch("rag.metadata_helper.get_connection")
    def test_text_fallback_name_matching(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_cur.fetchall.side_effect = [
            [],  # No patient names in column
            [
                ("Re: Mr. Arthur Pendelton (Patient)\nDOB: 01.01.1980\n",),
                ("Client Sarah Connor\nDOB: 02/02/1985\n",),
                ("Patient: Michael Scott\nDOB: 03-03-1990\n",),
                ("Bruce Wayne\nDate of Birth 04/04/1975\n",),
            ]
        ]

        meta = get_case_metadata("run_text_names")
        self.assertIn("Arthur Pendelton", meta["names"])



if __name__ == "__main__":
    unittest.main()

