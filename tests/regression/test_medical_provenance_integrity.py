"""Acceptance tests for exact medical provenance and evidence integrity."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.analyzer import replace_source_tags_in_string
from rag.chunker import _parse_date, chunk_document, chunk_documents_from_run
from rag.metadata_helper import get_case_timeline
from rag.metadata_helper import _build_metadata
from rag.retriever import _normalize_iso_date, format_context_for_llm


def test_migration_adds_exact_fields_without_promoting_legacy_offsets():
    migration = (
        Path(__file__).parents[2] / "migrations" / "20260726_medical_provenance.sql"
    ).read_text(encoding="utf-8")

    for field in (
        "source_char_start",
        "source_char_end",
        "page_start",
        "page_end",
        "provenance_type",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {field}" in migration
    assert "SET source_char_start = char_start" not in migration
    assert "SET source_char_end = char_end" not in migration
    assert "SET page_end = page_number" not in migration


def test_chunks_are_exact_source_slices_and_cross_page_ranges_are_exact():
    markdown = "\n\n  Page one evidence extends across the boundary into page two findings."
    boundary = markdown.index("into")
    chunks = chunk_document(
        markdown,
        doc_id="doc-exact",
        run_id="run-exact",
        page_ranges=[[0, boundary, 4], [boundary, len(markdown), 5]],
        max_chunk_size=800,
        chunk_overlap=0,
        original_filename="record.pdf",
        provenance_type="original_pdf",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source_char_start"] == markdown.index("Page")
    assert chunk["source_char_end"] == len(markdown)
    assert markdown[chunk["source_char_start"] : chunk["source_char_end"]] == chunk["text"]
    assert (chunk["page_start"], chunk["page_end"]) == (4, 5)


def test_invalid_calendar_dates_are_not_normalized():
    assert _parse_date("Dated 29/02/2024") == "2024-02-29"
    assert _parse_date("Dated 31/02/2024") is None
    assert _parse_date("Dated February 30, 2024") is None
    assert _normalize_iso_date("2024-02-31") is None

    metadata = _build_metadata([], [("DOB: 31/02/2024",)])
    assert metadata["dob"] == "Not present in source"
    assert metadata["dob_unparsed_raw"] == ["31/02/2024"]


def test_ocr_run_preserves_source_pdf_filename_for_number_prefixed_markdown(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run_fixture"
    markdown_dir = run_dir / "markdown" / "inputs"
    results_dir = run_dir / "results"
    markdown_dir.mkdir(parents=True)
    results_dir.mkdir()
    markdown = "Cross-page source evidence."
    (markdown_dir / "0_record.md").write_text(markdown, encoding="utf-8")
    (results_dir / "result.jsonl").write_text(
        json.dumps(
            {
                "metadata": {"Source-File": "inputs/record.pdf"},
                "attributes": {
                    "pdf_page_numbers": [[0, 10, 1], [10, len(markdown), 2]]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("settings_manager.WORKSPACE_DIR", str(tmp_path))

    documents = chunk_documents_from_run(str(run_dir), "run-fixture")

    document = next(iter(documents.values()))
    chunk = document["chunks"][0]
    assert document["original_filename"] == "record.pdf"
    assert chunk["original_filename"] == "record.pdf"
    assert chunk["provenance_type"] == "original_pdf"
    assert (chunk["page_start"], chunk["page_end"]) == (1, 2)


@patch("rag.metadata_helper.get_connection")
def test_timeline_retains_raw_dates_sorts_normalized_and_never_invents_metadata(
    mock_get_connection,
):
    cursor = MagicMock()
    mock_get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchall.return_value = [
        (
            2,
            2,
            3,
            10,
            180,
            "original_pdf",
            "specialist_letter",
            None,
            date(2024, 2, 29),
            "29/02/2024",
            "Claim No: ABC-99\nCross-page clinical findings.",
            "record.pdf",
            "doc-pdf",
            0,
        ),
        (
            None,
            None,
            None,
            0,
            50,
            "external_markdown",
            None,
            None,
            date(2024, 1, 2),
            "2/1/2024",
            "External clinical note.",
            "note.md",
            "doc-md",
            0,
        ),
        (
            8,
            8,
            8,
            200,
            250,
            "original_pdf",
            "clinical_notes",
            "Dr Source Author",
            None,
            "31/02/2023",
            "31/02/2023\nClinic: Source Named Clinic",
            "record.pdf",
            "doc-invalid",
            0,
        ),
    ]

    events = get_case_timeline("run-medical")

    assert [event["dateNormalized"] for event in events] == [
        "2024-01-02",
        "2024-02-29",
        None,
    ]
    assert events[1]["dateRaw"] == "29/02/2024"
    assert events[1]["pageRange"] == "Pages 2-3"
    assert events[1]["originalFilename"] == "record.pdf"
    assert events[1]["physician"] is None
    assert events[1]["clinic"] is None
    assert events[1]["refNo"] == "Claim No: ABC-99"

    external = events[0]
    assert external["pageRange"] is None
    assert external["pageStart"] is None
    assert external["pageEnd"] is None
    assert external["pageProvenance"] == (
        "No original-PDF page provenance (external Markdown)"
    )

    invalid = events[2]
    assert invalid["dateRaw"] == "31/02/2023"
    assert invalid["dateStatus"] == "unparsed"
    assert invalid["clinic"] == "Source Named Clinic"

    serialized = json.dumps(events)
    for invented in (
        "MedRec-Internal",
        "Treating Practitioner",
        "Medical Clinic / Health Service",
        "Melbourne Orthopaedic Group",
    ):
        assert invented not in serialized


def test_citations_use_exact_ranges_and_mark_external_markdown():
    results = [
        {
            "original_filename": "record.pdf",
            "page_start": 7,
            "page_end": 8,
            "provenance_type": "original_pdf",
            "document_type": "specialist_letter",
            "author": None,
            "date_extracted": "2024-02-29",
            "text": "Accession Number: IMG-77",
        },
        {
            "original_filename": "note.md",
            "page_start": None,
            "page_end": None,
            "provenance_type": "external_markdown",
            "document_type": None,
            "author": None,
            "date_extracted": None,
            "text": "External note.",
        },
    ]

    citation = replace_source_tags_in_string("Finding [Source 1]. Note [Source 2].", results)
    assert "record.pdf" in citation
    assert "pp. 7-8" in citation
    assert "Accession Number: IMG-77" in citation
    assert "note.md" in citation
    assert "external Markdown; no original-PDF page provenance" in citation
    assert "p. 1" not in citation

    context = format_context_for_llm(results)
    assert "Pages: 7-8" in context
    assert "PDF provenance: none (external Markdown)" in context
