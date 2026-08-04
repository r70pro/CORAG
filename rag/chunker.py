"""
Medicolegal-aware document chunker.

Splits OCR-extracted markdown into semantically meaningful chunks,
preserving clinical note boundaries, letter structures, and
extracting rich metadata (dates, authors, document types, sections).

Designed for:
- GP clinical notes with date-keyed entries
- Specialist referral/report letters
- Physiotherapy progress notes
- Medico-legal assessment reports
- TAC/WorkCover correspondence
"""

import hashlib
import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

# ── Date extraction patterns ──────────────────────────────────
# Covers: DD/MM/YY, DD.MM.YY, DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY,
#         "12 February 2018", "Feb 12, 2018", etc.
DATE_PATTERNS = [
    # DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
    re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b"),
    # "12 February 2018" or "February 12, 2018"
    re.compile(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+"
        r"(\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+"
        r"(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
]

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# ── Author extraction patterns ────────────────────────────────
AUTHOR_PATTERNS = [
    # 1. Signatures at the bottom of letters
    re.compile(
        r"(?:Yours\s+sincerely|Kind\s+regards|Regards|Yours\s+faithfully|Signed|Dictated\s+by|On\s+behalf\s+of)\s*,?\s*\n+"
        r"(?:\([^)]*\)\s*\n+)?"
        r"(?:Dr\.?|A/Prof\.?|Prof\.?|Mr\.?|Ms\.?|Physiotherapist)?\s*"
        r"([A-Z][a-zA-Z\-\'\.]+(?:[ \t]+[A-Z][a-zA-Z\-\'\.]+){0,3})",
        re.IGNORECASE | re.MULTILINE,
    ),
    # 2. Clinical Notes headers / Sender tags
    re.compile(
        r"(?:Clinical\s+Notes\s+of|From:)\s*(?:Dr\.?|A/Prof\.?|Prof\.?|Mr\.?|Ms\.?|Physiotherapist)?\s*"
        r"([A-Z][a-zA-Z\-\'\.]+(?:[ \t]+[A-Z][a-zA-Z\-\'\.]+){0,3})",
        re.IGNORECASE,
    ),
    # 3. Fallback: general Dr/Prof matches but explicitly ignore recipients (Dear Dr...)
    re.compile(
        r"(?<!Dear\s)(?<!Dear\sDr\.\s)(?<!Dear\sDr\s)(?<!Dear\sA/Prof\.\s)(?<!Dear\sA/Prof\s)"
        r"(?:Dr\.?|A/Prof\.?|Prof\.?|Mr\.?|Ms\.?)\s+"
        r"([A-Z][a-zA-Z\-\'\.]+(?:[ \t]+[A-Z][a-zA-Z\-\'\.]+){0,3})",
    ),
]

# ── Document type classification patterns ─────────────────────
DOC_TYPE_PATTERNS = {
    "specialist_letter": re.compile(
        r"(?:Dear\s+(?:Dr|Sybille|Doctor|Prof))|"
        r"(?:Re:\s+.*DOB)|"
        r"(?:Yours\s+sincerely)|"
        r"(?:Kind\s+regards)",
        re.IGNORECASE,
    ),
    "clinical_notes": re.compile(
        r"(?:Clinical\s+Notes)|"
        r"(?:Date\s*\|?\s*Clinical\s*Notes)|"
        r"(?:<th>Date</th>\s*<th>Clinical)",
        re.IGNORECASE,
    ),
    "referral_letter": re.compile(
        r"(?:Thank\s+you\s+for\s+(?:referring|seeing))|"
        r"(?:Please\s+accept\s+this\s+referral)|"
        r"(?:I\s+am\s+referring)",
        re.IGNORECASE,
    ),
    "physiotherapy_report": re.compile(
        r"(?:Physiotherap)|" r"(?:ROM\b)|" r"(?:range\s+of\s+motion)|" r"(?:exercises?\s+program)",
        re.IGNORECASE,
    ),
    "medicolegal_report": re.compile(
        r"(?:medico.?legal)|" r"(?:independent\s+medical)|" r"(?:IME\b)|" r"(?:claim\s+number)",
        re.IGNORECASE,
    ),
    "radiology_report": re.compile(
        r"(?:MR(?:I)?\s+(?:OF\s+|scan|report|findings|both|spine))|"
        r"(?:CT\s+(?:OF\s+|scan))|"
        r"(?:X[ -]?RAY)|"
        r"(?:ultrasound\s+report)|"
        r"(?:imaging\s+findings)|"
        r"(?:Laboratory:\s*I-?MED\s+Radiology)|"
        r"(?:Name of Test:\s*(?:MRI|CT|X[ -]?RAY))",
        re.IGNORECASE,
    ),
}

# ── Section type classification ───────────────────────────────
SECTION_PATTERNS = {
    "clinical_findings": re.compile(
        r"(?:On\s+examination)|"
        r"(?:Clinical\s+findings)|"
        r"(?:Physical\s+examination)|"
        r"(?:range\s+of\s+motion)",
        re.IGNORECASE,
    ),
    "history": re.compile(
        r"(?:Past\s+(?:History|Medical))|"
        r"(?:Current\s+Problems)|"
        r"(?:Presenting\s+complaint)|"
        r"(?:History\s+of\s+(?:present|injury))",
        re.IGNORECASE,
    ),
    "medications": re.compile(
        r"(?:Current\s+Medications?)|" r"(?:Medications?:)|" r"(?:Prescribed)", re.IGNORECASE
    ),
    "diagnosis": re.compile(r"(?:Diagnosis|Impression|Assessment|Conclusion)", re.IGNORECASE),
    "treatment_plan": re.compile(
        r"(?:Treatment\s+Plan)|"
        r"(?:Recommendations?)|"
        r"(?:I\s+have\s+recommended)|"
        r"(?:Management\s+plan)",
        re.IGNORECASE,
    ),
    "allergies": re.compile(r"(?:Allergies?:?\s)|(?:Nil\s+Known)", re.IGNORECASE),
    "correspondence": re.compile(
        r"(?:Dear\s+)|(?:To\s+whom)|(?:ELECTRONIC\s+TRANSMISSION)", re.IGNORECASE
    ),
}

# ── Patient name extraction ───────────────────────────────────
PATIENT_PATTERNS = [
    re.compile(r"Re:\s+([A-Z][a-zA-Z\-\']+(?:[ \t]+[A-Z][a-zA-Z\-\']+){0,3})\s+DOB", re.IGNORECASE),
    re.compile(
        r"Re:\s+(?:Mr|Mrs|Ms|Miss)\.?\s+([A-Z][a-zA-Z\-\']+(?:[ \t]+[A-Z][a-zA-Z\-\']+){0,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"Patient(?:\s+Name)?:\s*([A-Z][a-zA-Z\-\']+(?:[ \t]+[A-Z][a-zA-Z\-\']+){0,3})",
        re.IGNORECASE,
    ),
]

# ── Letter boundary detection ─────────────────────────────────
RADIOLOGY_REPORT_BOUNDARY = re.compile(
    r"^(?:CT\s+OF\s+.+|MR(?:I)?\s+OF\s+.+|MRI\s+BOTH\s+HIPS)$",
    re.MULTILINE | re.IGNORECASE,
)

LETTER_BOUNDARY_PATTERNS = [
    # Primary investigation report titles. Bundled claim files often place the
    # next report's request metadata immediately after the preceding signature;
    # title boundaries keep findings and conclusions attached to the right test.
    RADIOLOGY_REPORT_BOUNDARY,
    # Date + Letter header
    re.compile(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\s+Letter", re.MULTILINE),
    # "Dear Dr..." at start of line
    re.compile(r"^Dear\s+(?:Dr|Prof|Mr|Ms|Mrs)", re.MULTILINE | re.IGNORECASE),
    # Transmission headers
    re.compile(r"^ELECTRONIC\s+TRANSMISSION", re.MULTILINE | re.IGNORECASE),
    # "Re: Patient Name DOB:"
    re.compile(r"^Re:\s+", re.MULTILINE),
    # Clinical Notes header
    re.compile(r"^Clinical\s+Notes\s+of", re.MULTILINE | re.IGNORECASE),
    # Scanned Document marker
    re.compile(r"Scanned\s+Document\s+\(\s*'", re.IGNORECASE),
]


def _first_date_match(text: str):
    matches = [match for pattern in DATE_PATTERNS if (match := pattern.search(text))]
    return min(matches, key=lambda match: match.start()) if matches else None


def _parse_date(text: str) -> str | None:
    """Extract and normalise the first date found in text to ISO-8601 format.

    Assumes the **Australian DD/MM/YYYY** convention used throughout
    medicolegal documents (e.g. ``12/03/2018`` = 12 March 2018). Documents that
    use US ``MM/DD/YYYY`` ordering will be mis-parsed and the date silently
    normalised incorrectly. Returns ``None`` if no recognised date is found.

    Returns:
        ISO date string (YYYY-MM-DD) or None.
    """
    match = _first_date_match(text)
    if match:
        groups = match.groups()
        try:
            if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit():
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                if year < 100:
                    year += 2000 if year < 50 else 1900
                return date(year, month, day).isoformat()
            elif len(groups) == 3:
                # Named month variants
                if groups[0].isdigit():
                    # "12 February 2018"
                    day = int(groups[0])
                    month = MONTH_MAP.get(groups[1].lower(), 0)
                    year = int(groups[2])
                else:
                    # "February 12, 2018"
                    month = MONTH_MAP.get(groups[0].lower(), 0)
                    day = int(groups[1])
                    year = int(groups[2])
                return date(year, month, day).isoformat()
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def _extract_raw_date(text: str) -> str | None:
    """Extract the raw date string as it appears in the text."""
    match = _first_date_match(text)
    return match.group(0) if match else None


def _extract_author(text: str) -> str | None:
    """Extract the most likely author name from text."""
    for pattern in AUTHOR_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _extract_radiology_author(text: str) -> str | None:
    """Prefer the signing radiologist over addressees/referrers."""
    signed = re.findall(
        r"^Dr\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s*\n"
        r"Electronically signed",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if signed:
        return f"Dr {signed[0]}"
    matches = re.findall(
        r"^Dr\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s*$",
        text,
        re.MULTILINE,
    )
    return f"Dr {matches[-1]}" if matches else None


def _extract_radiology_date(text: str) -> tuple[str | None, str | None]:
    """Use report/signature dates, never a patient's DOB, for imaging metadata."""
    candidates = re.findall(
        r"(?:^|\b)(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
        r"(\d{4})\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if not candidates:
        return None, None
    day, month_name, year = candidates[0]
    try:
        month_key = month_name.lower()
        month_number = MONTH_MAP.get(month_key)
        if month_number is None:
            month_number = next(
                value for name, value in MONTH_MAP.items() if name.startswith(month_key)
            )
        normalized = date(int(year), month_number, int(day)).isoformat()
    except (ValueError, KeyError):
        return None, None
    return normalized, f"{day} {month_name} {year}"


def _classify_document_type(text: str) -> str:
    """Classify the document type based on content patterns."""
    # Report title plus findings/conclusion is stronger evidence than the
    # generic "Dear Dr" structure shared by most clinical correspondence.
    if re.search(
        r"\b(?:CT\s+OF|MR(?:I)?\s+OF|MRI\s+BOTH|X[ -]?RAY)\b",
        text,
        re.IGNORECASE,
    ) and re.search(r"\b(?:Findings|Conclusion|Impression)\s*:", text, re.IGNORECASE):
        return "radiology_report"
    scores = {}
    for doc_type, pattern in DOC_TYPE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            scores[doc_type] = len(matches)

    if scores:
        return max(scores, key=scores.get)
    return "unknown"


def _classify_section_type(text: str) -> str:
    """Classify the section type based on content patterns."""
    for section_type, pattern in SECTION_PATTERNS.items():
        if pattern.search(text):
            return section_type
    return "general"


def _extract_patient_name(text: str) -> str | None:
    """Extract the patient name from text."""
    for pattern in PATIENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _find_page_for_position(char_pos: int, page_ranges: list) -> int | None:
    """Determine which page a character position falls on.

    Args:
        char_pos: Character position in the full markdown.
        page_ranges: List of [start, end, page_num] from JSONL attributes.

    Returns:
        Page number (1-indexed) or None.
    """
    for range_info in page_ranges:
        if len(range_info) >= 3:
            start, end, page_num = range_info[0], range_info[1], range_info[2]
            if start <= char_pos < end:
                return page_num
    return None


def _find_pages_for_span(
    source_char_start: int, source_char_end: int, page_ranges: list
) -> tuple[int | None, int | None]:
    """Return the source pages containing both ends of an exclusive span."""
    if source_char_end <= source_char_start:
        return None, None
    page_start = _find_page_for_position(source_char_start, page_ranges)
    page_end = _find_page_for_position(source_char_end - 1, page_ranges)
    return page_start, page_end


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic chunk ID."""
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Chunking configuration constants ──────────────────────────
MIN_SECTION_SIZE_CHARS = 200  # Minimum character count between section boundaries
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64


def _split_into_sections(text: str) -> list[tuple[int, int, str]]:
    """Split text into sections based on letter/document boundaries.

    Returns:
        List of (start_pos, end_pos, section_text) tuples.
    """
    boundary_positions = set()
    boundary_positions.add(0)
    report_positions = {match.start() for match in RADIOLOGY_REPORT_BOUNDARY.finditer(text)}
    report_end_positions = {
        match.end()
        for match in re.finditer(r"^Electronically signed[^\n]*$", text, re.MULTILINE | re.IGNORECASE)
    }
    boundary_positions.update(report_end_positions)

    for pattern in LETTER_BOUNDARY_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            # Don't create tiny sections — minimum MIN_SECTION_SIZE_CHARS from previous
            if pos > 0:
                boundary_positions.add(pos)

    # Sort and filter out boundaries that are too close together
    sorted_positions = sorted(boundary_positions)
    filtered_positions = [sorted_positions[0]]
    for pos in sorted_positions[1:]:
        if pos - filtered_positions[-1] >= MIN_SECTION_SIZE_CHARS:
            filtered_positions.append(pos)
        elif pos in report_positions and filtered_positions[-1] not in report_positions:
            # Prefer the precise report title over a nearby generic "Dear Dr"
            # or "Re:" boundary so the evidence unit starts at its identity.
            filtered_positions[-1] = pos

    sections = []
    for i, start in enumerate(filtered_positions):
        end = filtered_positions[i + 1] if i + 1 < len(filtered_positions) else len(text)
        content_match = re.search(r"\S(?:[\s\S]*\S)?", text[start:end])
        if content_match:
            exact_start = start + content_match.start()
            exact_end = start + content_match.end()
            sections.append((exact_start, exact_end, text[exact_start:exact_end]))

    return sections


def _split_section_into_chunks(
    section_text: str,
    section_start: int,
    max_chunk_size: int = 800,
    overlap: int = 100,
) -> list[tuple[int, int, str]]:
    """Split a section into overlapping chunks at paragraph boundaries.

    Tries to split at double-newlines (paragraph breaks) first,
    then at single newlines, then at sentence boundaries.

    Args:
        section_text: The section text to split.
        section_start: Character offset of section start in full document.
        max_chunk_size: Maximum chunk size in characters (~tokens at 4 chars/token).
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of (global_start, global_end, chunk_text) tuples.
    """
    if len(section_text) <= max_chunk_size:
        return [(section_start, section_start + len(section_text), section_text)]

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")
    overlap = max(0, min(overlap, max_chunk_size - 1))

    chunks = []
    local_start = 0
    text_length = len(section_text)

    while local_start < text_length:
        while local_start < text_length and section_text[local_start].isspace():
            local_start += 1
        if local_start >= text_length:
            break

        limit = min(local_start + max_chunk_size, text_length)
        split_end = limit
        if limit < text_length:
            minimum_break = local_start + max_chunk_size // 2

            paragraph_end = None
            for match in re.finditer(r"\n[ \t]*\n+", section_text[local_start:limit]):
                candidate = local_start + match.start()
                if candidate >= minimum_break:
                    paragraph_end = candidate
            if paragraph_end is not None:
                split_end = paragraph_end
            else:
                sentence_end = section_text.rfind(". ", local_start, limit)
                if sentence_end >= minimum_break:
                    split_end = sentence_end + 1
                else:
                    newline_end = section_text.rfind("\n", local_start, limit)
                    if newline_end >= minimum_break:
                        split_end = newline_end

        exact_end = split_end
        while exact_end > local_start and section_text[exact_end - 1].isspace():
            exact_end -= 1

        if exact_end <= local_start:
            local_start = max(local_start + 1, split_end)
            continue

        global_start = section_start + local_start
        global_end = section_start + exact_end
        chunks.append((global_start, global_end, section_text[local_start:exact_end]))

        if exact_end >= text_length:
            break
        next_start = exact_end - overlap if overlap else split_end
        if next_start <= local_start:
            next_start = local_start + 1
        local_start = next_start

    return chunks


def chunk_document(
    markdown_text: str,
    doc_id: str,
    run_id: str,
    page_ranges: list | None = None,
    max_chunk_size: int = 800,
    chunk_overlap: int = 100,
    original_filename: str | None = None,
    provenance_type: str | None = None,
) -> list[dict]:
    """Chunk a document into semantically meaningful pieces with rich metadata.

    This is the main entry point for the chunker.

    Args:
        markdown_text: Full markdown text extracted by OLMOCR.
        doc_id: Unique document identifier.
        run_id: OCR run identifier (for provenance).
        page_ranges: Optional page boundary data from JSONL attributes.
        max_chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        original_filename: Source filename supported by ingestion metadata.
        provenance_type: ``original_pdf``, ``external_markdown``, or a
            no-page-map Markdown provenance marker.

    Returns:
        List of chunk dicts ready for embedding and database insertion.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    if page_ranges is None:
        page_ranges = []

    # Step 1: Extract document-level metadata
    doc_patient_name = _extract_patient_name(markdown_text)
    doc_type = _classify_document_type(markdown_text)

    # Step 2: Split into sections (letter boundaries, report boundaries)
    sections = _split_into_sections(markdown_text)

    # Step 3: Split sections into overlapping chunks
    all_chunks = []
    chunk_index = 0

    for section_start, _, section_text in sections:
        # Extract section-level metadata
        section_author = _extract_author(section_text)
        section_date = _parse_date(section_text)
        section_date_raw = _extract_raw_date(section_text)
        section_doc_type = _classify_document_type(section_text)
        section_patient = _extract_patient_name(section_text) or doc_patient_name

        # Use section-level doc type if different from document-level
        effective_doc_type = section_doc_type if section_doc_type != "unknown" else doc_type

        # Split the section into chunks
        # A radiology report is the minimum safe evidence unit: its identity,
        # findings, conclusion and signatory must not be separated. Child-size
        # chunks caused title-only hits to displace the actual findings.
        effective_chunk_size = (
            max(max_chunk_size, 6000)
            if effective_doc_type == "radiology_report"
            else max_chunk_size
        )
        chunk_pieces = _split_section_into_chunks(
            section_text, section_start, effective_chunk_size, chunk_overlap
        )

        for chunk_start, chunk_end, chunk_text in chunk_pieces:
            if not chunk_text.strip():
                continue

            page_start, page_end = _find_pages_for_span(chunk_start, chunk_end, page_ranges)

            # Chunk-level metadata extraction (may override section-level)
            chunk_author = _extract_author(chunk_text) or section_author
            chunk_date = _parse_date(chunk_text) or section_date
            chunk_date_raw = _extract_raw_date(chunk_text) or section_date_raw
            chunk_section_type = _classify_section_type(chunk_text)
            if effective_doc_type == "radiology_report":
                chunk_author = _extract_radiology_author(chunk_text) or chunk_author
                radiology_date, radiology_date_raw = _extract_radiology_date(chunk_text)
                chunk_date = radiology_date or chunk_date
                chunk_date_raw = radiology_date_raw or chunk_date_raw

            chunk_id = _make_chunk_id(doc_id, chunk_index)

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "run_id": run_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "char_start": chunk_start,
                    "char_end": chunk_end,
                    "source_char_start": chunk_start,
                    "source_char_end": chunk_end,
                    "page_number": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "original_filename": original_filename,
                    "provenance_type": provenance_type,
                    "document_type": effective_doc_type,
                    "author": chunk_author,
                    "date_extracted": chunk_date,
                    "date_raw": chunk_date_raw,
                    "section_type": chunk_section_type,
                    "patient_name": section_patient,
                    "token_count": len(chunk_text) // 4,  # Rough token estimate
                }
            )
            chunk_index += 1

    return all_chunks


def chunk_documents_from_run(
    run_dir: str,
    run_id: str,
    max_chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> dict[str, list[dict]]:
    """Chunk all documents from a completed OCR run.

    Args:
        run_dir: Path to the run directory.
        run_id: OCR run identifier.
        max_chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Dict mapping doc_id -> list of chunk dicts.
    """
    import json
    from pathlib import Path

    from path_security import (
        PathSecurityError,
        resolve_file_under,
        resolve_run_under,
        resolve_under,
    )
    from settings_manager import WORKSPACE_DIR

    results = {}
    candidate_run = Path(run_dir)
    try:
        safe_run_dir = resolve_run_under(WORKSPACE_DIR, candidate_run.name)
    except PathSecurityError:
        return results
    if candidate_run.resolve() != safe_run_dir:
        return results
    md_inputs_dir = resolve_under(safe_run_dir, "markdown", "inputs")
    results_dir = resolve_under(safe_run_dir, "results")

    if not md_inputs_dir.is_dir():
        return results

    # Load page ranges from JSONL results
    source_metadata_by_markdown = {}
    if results_dir.is_dir():
        for entry in results_dir.iterdir():
            try:
                jsonl_path = resolve_file_under(results_dir, entry.name, {".jsonl"})
            except PathSecurityError:
                continue
            if jsonl_path.is_file():
                try:
                    with jsonl_path.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            source_file = data.get("metadata", {}).get("Source-File", "")
                            page_ranges = data.get("attributes", {}).get("pdf_page_numbers", [])
                            source_basename = Path(str(source_file).replace("\\", "/")).name
                            if Path(source_basename).suffix.lower() == ".pdf":
                                md_name = Path(source_basename).with_suffix(".md").name
                                source_metadata_by_markdown[md_name] = {
                                    "page_ranges": page_ranges,
                                    "original_filename": source_basename,
                                    "provenance_type": (
                                        "original_pdf"
                                        if page_ranges
                                        else "markdown_without_pdf_page_map"
                                    ),
                                }
                except Exception:
                    logger.error("Error reading run JSONL metadata")

    # Process each markdown file
    for entry in sorted(md_inputs_dir.iterdir(), key=lambda path: path.name):
        try:
            md_path = resolve_file_under(md_inputs_dir, entry.name, {".md"})
        except PathSecurityError:
            continue
        if not md_path.is_file():
            continue
        md_file = md_path.name
        try:
            markdown_text = md_path.read_text(encoding="utf-8")
        except OSError:
            logger.error("Error reading run Markdown")
            continue

        # Generate deterministic doc_id from run_id and filename
        doc_id = hashlib.sha256(f"{run_id}:{md_file}".encode()).hexdigest()[:24]

        unprefixed_md_file = re.sub(r"^\d+_", "", md_file)
        source_metadata = source_metadata_by_markdown.get(
            md_file, source_metadata_by_markdown.get(unprefixed_md_file, {})
        )
        page_ranges = source_metadata.get("page_ranges", [])
        original_filename = source_metadata.get("original_filename", md_file)
        provenance_type = source_metadata.get("provenance_type", "markdown_without_pdf_page_map")

        chunks = chunk_document(
            markdown_text=markdown_text,
            doc_id=doc_id,
            run_id=run_id,
            page_ranges=page_ranges,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            original_filename=original_filename,
            provenance_type=provenance_type,
        )

        results[doc_id] = {
            "doc_id": doc_id,
            "md_file": md_file,
            "md_path": str(md_path),
            "markdown_text": markdown_text,
            "page_ranges": page_ranges,
            "original_filename": original_filename,
            "provenance_type": provenance_type,
            "chunks": chunks,
        }

    return results
