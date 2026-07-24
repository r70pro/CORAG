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
        r"(?:MRI\s+(?:scan|report|findings))|"
        r"(?:CT\s+scan)|"
        r"(?:X-ray)|"
        r"(?:ultrasound\s+report)|"
        r"(?:imaging\s+findings)",
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
LETTER_BOUNDARY_PATTERNS = [
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


def _parse_date(text: str) -> str | None:
    """Extract and normalise the first date found in text to ISO-8601 format.

    Assumes the **Australian DD/MM/YYYY** convention used throughout
    medicolegal documents (e.g. ``12/03/2018`` = 12 March 2018). Documents that
    use US ``MM/DD/YYYY`` ordering will be mis-parsed and the date silently
    normalised incorrectly. Returns ``None`` if no recognised date is found.

    Returns:
        ISO date string (YYYY-MM-DD) or None.
    """
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit():
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return f"{year:04d}-{month:02d}-{day:02d}"
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
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, TypeError):
                continue
    return None


def _extract_raw_date(text: str) -> str | None:
    """Extract the raw date string as it appears in the text."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _extract_author(text: str) -> str | None:
    """Extract the most likely author name from text."""
    for pattern in AUTHOR_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _classify_document_type(text: str) -> str:
    """Classify the document type based on content patterns."""
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

    sections = []
    for i, start in enumerate(filtered_positions):
        end = filtered_positions[i + 1] if i + 1 < len(filtered_positions) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((start, end, section_text))

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

    # Find natural break points (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", section_text)

    chunks = []
    current_chunk = ""
    current_start = section_start

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            # Current chunk is full — save it
            if current_chunk:
                chunk_end = current_start + len(current_chunk)
                chunks.append((current_start, chunk_end, current_chunk))
                # Start next chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_start = chunk_end - len(overlap_text)
                current_chunk = overlap_text + ("\n\n" if overlap_text else "") + para
            else:
                current_chunk = para

            # Handle paragraphs that are themselves too large
            while len(current_chunk) > max_chunk_size:
                # Split at sentence boundary
                split_point = max_chunk_size
                sentence_end = current_chunk.rfind(". ", 0, max_chunk_size)
                if sentence_end > max_chunk_size // 2:
                    split_point = sentence_end + 2
                else:
                    newline_pos = current_chunk.rfind("\n", 0, max_chunk_size)
                    if newline_pos > max_chunk_size // 2:
                        split_point = newline_pos + 1

                chunk_text = current_chunk[:split_point].strip()
                # Clamp overlap so the residual slice start never goes negative
                # (which would wrap to the end of the string and duplicate or
                # loop forever when split_point is small).
                actual_overlap = max(0, min(overlap, len(chunk_text) - 1))
                safe_start = len(chunk_text) - actual_overlap
                chunk_end = current_start + len(chunk_text)
                chunks.append((current_start, chunk_end, chunk_text))
                current_start = chunk_end - actual_overlap
                current_chunk = current_chunk[safe_start:].strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append((current_start, current_start + len(current_chunk), current_chunk.strip()))

    return chunks


def chunk_document(
    markdown_text: str,
    doc_id: str,
    run_id: str,
    page_ranges: list | None = None,
    max_chunk_size: int = 800,
    chunk_overlap: int = 100,
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
        chunk_pieces = _split_section_into_chunks(
            section_text, section_start, max_chunk_size, chunk_overlap
        )

        for chunk_start, chunk_end, chunk_text in chunk_pieces:
            if not chunk_text.strip():
                continue

            # Determine which page this chunk falls on
            page_num = _find_page_for_position(chunk_start, page_ranges)

            # Chunk-level metadata extraction (may override section-level)
            chunk_author = _extract_author(chunk_text) or section_author
            chunk_date = _parse_date(chunk_text) or section_date
            chunk_date_raw = _extract_raw_date(chunk_text) or section_date_raw
            chunk_section_type = _classify_section_type(chunk_text)

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
                    "page_number": page_num,
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
    import os

    results = {}
    md_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
    results_dir = os.path.join(run_dir, "results")

    if not os.path.exists(md_inputs_dir):
        return results

    # Load page ranges from JSONL results
    page_ranges_by_source = {}
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".jsonl"):
                jsonl_path = os.path.join(results_dir, f)
                try:
                    with open(jsonl_path, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            source_file = data.get("metadata", {}).get("Source-File", "")
                            page_ranges = data.get("attributes", {}).get("pdf_page_numbers", [])
                            source_basename = os.path.basename(source_file)
                            if source_basename.endswith(".pdf"):
                                md_name = source_basename[:-4] + ".md"
                                page_ranges_by_source[md_name] = page_ranges
                except Exception as e:
                    logger.error(f"Error reading JSONL {jsonl_path}: {e}")

    # Process each markdown file
    for md_file in sorted(os.listdir(md_inputs_dir)):
        if not md_file.endswith(".md"):
            continue

        md_path = os.path.join(md_inputs_dir, md_file)
        try:
            with open(md_path, encoding="utf-8") as f:
                markdown_text = f.read()
        except Exception as e:
            logger.error(f"Error reading {md_path}: {e}")
            continue

        # Generate deterministic doc_id from run_id and filename
        doc_id = hashlib.sha256(f"{run_id}:{md_file}".encode()).hexdigest()[:24]

        page_ranges = page_ranges_by_source.get(md_file, [])

        chunks = chunk_document(
            markdown_text=markdown_text,
            doc_id=doc_id,
            run_id=run_id,
            page_ranges=page_ranges,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

        results[doc_id] = {
            "doc_id": doc_id,
            "md_file": md_file,
            "md_path": md_path,
            "markdown_text": markdown_text,
            "page_ranges": page_ranges,
            "chunks": chunks,
        }

    return results
