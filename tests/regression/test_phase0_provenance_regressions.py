"""Character and page provenance contracts backed by explicit fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.chunker import chunk_document

pytestmark = pytest.mark.phase0_regression

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "provenance_cases.json").read_text(encoding="utf-8")
)
CASES = {case["id"]: case for case in FIXTURES}


def _chunks(case_id: str):
    case = CASES[case_id]
    chunks = chunk_document(
        case["markdown"],
        doc_id=f"doc-{case_id}",
        run_id="run-provenance",
        page_ranges=case["page_ranges"],
        max_chunk_size=case["max_chunk_size"],
        chunk_overlap=case["chunk_overlap"],
    )
    assert chunks
    return case, chunks


def test_leading_whitespace_keeps_absolute_character_and_page_provenance():
    case, chunks = _chunks("leading-whitespace")
    first = chunks[0]

    assert first["char_start"] == case["expected_first_start"]
    assert first["page_number"] == case["expected_first_page"]
    assert case["markdown"][first["char_start"] : first["char_end"]] == first["text"]


def test_paragraph_normalization_preserves_exact_source_slices_without_duplicates():
    case, chunks = _chunks("paragraph-normalization")

    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert len({(chunk["char_start"], chunk["char_end"]) for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        source_slice = case["markdown"][chunk["char_start"] : chunk["char_end"]]
        assert source_slice == chunk["text"]


def test_cross_page_chunk_records_both_page_boundaries():
    case, chunks = _chunks("cross-page-chunk")
    first = chunks[0]

    assert first["page_start"] == case["expected_page_start"]
    assert first["page_end"] == case["expected_page_end"]
