import re
from datetime import date
from typing import Any

from rag.db import get_connection

_TIMELINE_DATE_RE = re.compile(
    r"\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
    re.IGNORECASE,
)

_REFERENCE_RE = re.compile(
    r"\b(?:Ref(?:erence)?|Claim|Accession)"
    r"(?:\s*(?:No|Number)\.?)?\s*:\s*[A-Z0-9][A-Z0-9._/\-]*",
    re.IGNORECASE,
)

_CLINIC_RE = re.compile(
    r"^(?:Clinic|Practice|Facility|Hospital|Medical Centre|Health Service)"
    r"\s*:\s*(\S[^\n]{1,120})$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_timeline_date(value: Any) -> str | None:
    """Normalize a source date only when it is a real calendar date."""
    if isinstance(value, date):
        return value.isoformat()
    if not value:
        return None

    raw = str(value).strip()
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        pass

    numeric = re.fullmatch(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", raw)
    if numeric:
        day, month, year = (int(part) for part in numeric.groups())
        if year < 100:
            year += 2000 if year < 50 else 1900
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    named = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if named:
        month_numbers = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        try:
            return date(
                int(named.group(3)),
                month_numbers[named.group(2)[:3].lower()],
                int(named.group(1)),
            ).isoformat()
        except (KeyError, ValueError):
            return None
    return None


def _source_date(date_extracted: Any, date_raw: Any, text: str) -> tuple[str | None, str | None]:
    has_raw_date = bool(date_raw)
    raw = str(date_raw).strip() if has_raw_date else None
    if raw is None and date_extracted:
        raw = str(date_extracted)
    if raw is None:
        match = _TIMELINE_DATE_RE.search(text)
        raw = match.group(1) if match else None
    normalized = (
        _normalize_timeline_date(raw)
        if has_raw_date or not date_extracted
        else _normalize_timeline_date(date_extracted)
    )
    return raw, normalized


def _compile_patterns():
    dob_regex = re.compile(
        r"(?:DOB|Date of Birth)\s*:?\s*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})", re.IGNORECASE
    )
    injury_patterns = [
        re.compile(r"(?:diagnosis|injury|injuries)\s+(?:of|is|was|were)\s+([^.\n]{5,100})", re.I),
        re.compile(r"(?:diagnosed|sustained)\s+(?:with|a)\s+([^.\n]{5,100})", re.I),
        re.compile(r"Re:\s+[^.\n]+?(?:DOB|Date of Birth)[^.\n]*?\n([^.\n]{5,100})", re.I),
        re.compile(r"diagnosis\s*:\s*([^.\n]{5,100})", re.I),
        re.compile(r"injury\s*:\s*([^.\n]{5,100})", re.I),
    ]
    ignore_keywords = [
        "circuit",
        "landing",
        "road",
        "street",
        "highway",
        "avenue",
        "drive",
        "court",
        "tel",
        "phone",
        "ref",
        "dear",
        "patient",
        "client",
        "address",
        "phone",
        "mobile",
        "medicare",
        "age",
        "gentleman",
        "wife",
        "son",
        "letter",
        "referral",
    ]
    return dob_regex, injury_patterns, ignore_keywords


def _build_metadata(names_rows: list[tuple], text_rows: list[tuple]) -> dict[str, Any]:
    """Extract client names, DOB, and injuries from pre-fetched rows."""
    dob_regex, injury_patterns, ignore_keywords = _compile_patterns()

    names = []
    dobs = set()
    unparsed_dobs = set()
    injuries = []

    for (name,) in names_rows:
        clean_name = (name or "").strip()
        if clean_name and len(clean_name) > 2:
            title_name = clean_name.title()
            if not any(k in title_name.lower() for k in ignore_keywords):
                names.append(title_name)

    # Deduplicate and canonicalize names (strip common titles, drop substrings)
    unique_names = list(set(names))
    unique_names.sort(key=len, reverse=True)
    final_names = []
    for n in unique_names:
        stripped_n = re.sub(r"^(mr|mrs|ms|dr|prof)\.?\s+", "", n, flags=re.I).strip()
        if (
            stripped_n
            and stripped_n not in final_names
            and all(stripped_n not in existing for existing in final_names)
        ):
            final_names.append(stripped_n)

    for (text,) in text_rows:
        if not text:
            continue
        m = dob_regex.search(text)
        if m:
            raw_dob = m.group(1).replace(".", "/")
            if _normalize_timeline_date(raw_dob):
                dobs.add(raw_dob)
            else:
                unparsed_dobs.add(raw_dob)

        # Text fallback for patient name if not present in column
        if not names:
            name_matches = [
                re.search(
                    r"Re:\s*(?:Mr|Mrs|Ms|Dr|Prof)?\.?\s*([A-Z][a-zA-Z\-'`]+(?:\s+[A-Z][a-zA-Z\-'`]+){1,3})",
                    text,
                ),
                re.search(r"Client\s+([A-Z][a-zA-Z\-'`]+(?:\s+[A-Z][a-zA-Z\-'`]+){1,3})", text),
                re.search(
                    r"Patient\s*:?\s*([A-Z][a-zA-Z\-'`]+(?:\s+[A-Z][a-zA-Z\-'`]+){1,3})", text
                ),
                re.search(
                    r"([A-Z][a-zA-Z\-'`]+(?:\s+[A-Z][a-zA-Z\-'`]+){1,2})\s+(?:DOB|Date of Birth)",
                    text,
                ),
            ]
            for nm in name_matches:
                if nm:
                    cand = nm.group(1).strip()
                    if (
                        cand
                        and len(cand) > 3
                        and not any(k in cand.lower() for k in ignore_keywords)
                    ):
                        names.append(cand.title())
                        break

        for p in injury_patterns:
            for match in p.finditer(text):
                phrase = re.sub(r"\s+", " ", match.group(1).strip())
                if any(k in phrase.lower() for k in ignore_keywords):
                    continue
                phrase = re.sub(r"^(?:[-*•\s\d]+|[A-Za-z]\)\s*)", "", phrase).strip()
                phrase = phrase.rstrip(",.;:-\"'")

                if 5 < len(phrase) < 90:
                    if not any(phrase.lower() == existing.lower() for existing in injuries):
                        injuries.append(phrase)

        # Specific medicolegal injury pattern matching
        med_injuries = re.findall(
            r"(?:glenohumeral dislocation|bankart lesion|slap repair|slap tear|supraspinatus tear|biceps tendonitis|disc herniation|whiplash|sciatica|abdominal strain|rotator cuff tear)",
            text,
            re.IGNORECASE,
        )
        for mi in med_injuries:
            clean_mi = mi.title()
            if not any(
                clean_mi.lower() in existing.lower() or existing.lower() in clean_mi.lower()
                for existing in injuries
            ):
                injuries.append(clean_mi)

    dob_list = sorted(dobs, key=len, reverse=True)  # Prefer 4-digit year format
    dob_str = dob_list[0] if dob_list else "Not present in source"

    # Deduplicate and canonicalize names
    unique_names = list(set(names))
    unique_names.sort(key=len, reverse=True)
    final_names = []
    for n in unique_names:
        stripped_n = re.sub(r"^(mr|mrs|ms|dr|prof)\.?\s+", "", n, flags=re.I).strip()
        if (
            stripped_n
            and stripped_n not in final_names
            and all(stripped_n not in existing for existing in final_names)
        ):
            final_names.append(stripped_n)

    generic_words = ["injury/condition", "our client", "condition", "the client"]
    clean_injuries = []
    for inj in injuries:
        if inj and not any(gen in inj.lower() for gen in generic_words):
            clean_injuries.append(inj[0].upper() + inj[1:])

    return {
        "names": final_names[:2],
        "dob": dob_str,
        "dob_unparsed_raw": sorted(unparsed_dobs),
        "injuries": clean_injuries[:4],
    }


def get_case_metadata(run_id: str) -> dict[str, Any]:
    """Extract client names, DOB, and injury region/type for a given run.

    Args:
        run_id: The OCR run identifier.

    Returns:
        Dict containing names, dob, and injuries.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT patient_name FROM chunks "
                    "WHERE run_id = %s AND patient_name IS NOT NULL",
                    (run_id,),
                )
                names_rows = cur.fetchall()
                cur.execute("SELECT text FROM chunks WHERE run_id = %s", (run_id,))
                text_rows = cur.fetchall()
        return _build_metadata(names_rows, text_rows)
    except Exception as e:
        return {
            "names": [],
            "dob": "Not present in source",
            "dob_unparsed_raw": [],
            "injuries": [f"Error loading metadata: {e}"],
        }


def get_all_cases_metadata(run_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Batch-fetch metadata for many runs in a single set of queries.

    Avoids the N+1 query problem of calling get_case_metadata() per card when
    rendering the Case Dashboard.

    Args:
        run_ids: List of run identifiers.

    Returns:
        Dict mapping run_id -> metadata dict (empty metadata on per-run error).
    """
    result: dict[str, dict[str, Any]] = {}
    if not run_ids:
        return result

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # One query for all patient names, grouped by run.
                cur.execute(
                    "SELECT run_id, patient_name FROM chunks "
                    "WHERE run_id = ANY(%s) AND patient_name IS NOT NULL",
                    (run_ids,),
                )
                names_by_run: dict[str, list[tuple]] = {rid: [] for rid in run_ids}
                for run_id, name in cur.fetchall():
                    names_by_run.setdefault(run_id, []).append((name,))

                # One query for all chunk text, grouped by run.
                cur.execute("SELECT run_id, text FROM chunks WHERE run_id = ANY(%s)", (run_ids,))
                text_by_run: dict[str, list[tuple]] = {rid: [] for rid in run_ids}
                for run_id, text in cur.fetchall():
                    text_by_run.setdefault(run_id, []).append((text,))
    except Exception as e:
        # On failure return empty metadata for every requested run.
        return {
            rid: {
                "names": [],
                "dob": "Not present in source",
                "dob_unparsed_raw": [],
                "injuries": [f"Error loading metadata: {e}"],
            }
            for rid in run_ids
        }

    for rid in run_ids:
        try:
            result[rid] = _build_metadata(names_by_run.get(rid, []), text_by_run.get(rid, []))
        except Exception:
            result[rid] = {
                "names": [],
                "dob": "Not present in source",
                "dob_unparsed_raw": [],
                "injuries": [],
            }
    return result


def get_case_timeline(run_id: str) -> list[dict[str, Any]]:
    """Extract chronological medicolegal timeline events for a given run ID.

    Conforms to Medical Document Audits and Citations Rule:
    - Never uses raw system source tags (e.g. [Source 26]).
    - Cites exact original-PDF page ranges only when both endpoints are known.
    - Includes physician/clinic, document type, and reference only when the
      source supplies them.

    Args:
        run_id: The OCR run identifier.

    Returns:
        List of dicts representing structured timeline events.
    """
    events = []
    if not run_id:
        return events

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.page_number, c.page_start, c.page_end,
                           c.source_char_start, c.source_char_end,
                           c.provenance_type, c.document_type, c.author,
                           c.date_extracted, c.date_raw, c.text,
                           d.original_filename, c.doc_id, c.chunk_index
                    FROM chunks c
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE c.run_id = %s
                    ORDER BY d.original_filename, c.chunk_index
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()

        if not rows:
            return events

        date_groups: dict[tuple, list[dict[str, Any]]] = {}

        for row in rows:
            if len(row) == 6:
                # Compatibility for callers/tests using the pre-migration shape.
                page_num, doc_type, author, date_ext, date_raw, text = row
                page_start = None
                page_end = None
                source_char_start = source_char_end = None
                provenance_type = original_filename = doc_id = None
                chunk_index = 0
            else:
                (
                    page_num,
                    page_start,
                    page_end,
                    source_char_start,
                    source_char_end,
                    provenance_type,
                    doc_type,
                    author,
                    date_ext,
                    date_raw,
                    text,
                    original_filename,
                    doc_id,
                    chunk_index,
                ) = row
            if not text:
                continue

            date_expression, normalized_date = _source_date(date_ext, date_raw, text)
            if not date_expression:
                continue

            doc_type_disp = (
                str(doc_type).replace("_", " ").title()
                if doc_type and str(doc_type).lower() != "unknown"
                else None
            )
            physician = str(author).strip() if author and len(str(author).strip()) > 2 else None
            clinic_match = _CLINIC_RE.search(text)
            clinic = clinic_match.group(1).strip() if clinic_match else None
            ref_match = _REFERENCE_RE.search(text)
            ref_no = ref_match.group(0).strip().rstrip(",.;") if ref_match else None

            clean_summary = text.split("\n\n")[0].replace("\n", " ").strip()
            clean_summary = re.sub(r"\[Source \d+\]", "", clean_summary).strip()
            if len(clean_summary) > 200:
                clean_summary = clean_summary[:197] + "..."

            group_key = (
                doc_id or original_filename or id(row),
                normalized_date or date_expression,
                physician,
                doc_type_disp,
                ref_no,
            )
            date_groups.setdefault(group_key, []).append(
                {
                    "date": date_expression,
                    "dateNormalized": normalized_date,
                    "pageStart": page_start,
                    "pageEnd": page_end,
                    "sourceCharStart": source_char_start,
                    "sourceCharEnd": source_char_end,
                    "provenanceType": provenance_type,
                    "originalFilename": original_filename,
                    "docType": doc_type_disp,
                    "physician": physician,
                    "clinic": clinic,
                    "refNo": ref_no,
                    "summary": clean_summary,
                }
            )

        for items in date_groups.values():
            first_item = items[0]
            starts = [item["pageStart"] for item in items if item["pageStart"] is not None]
            ends = [item["pageEnd"] for item in items if item["pageEnd"] is not None]
            exact_pages = bool(starts) and len(ends) == len(items)
            range_start = min(starts) if starts else None
            range_end = max(ends) if exact_pages else None

            if first_item["provenanceType"] == "external_markdown":
                page_range = None
                page_provenance = "No original-PDF page provenance (external Markdown)"
            elif range_start is not None and range_end is not None:
                page_range = (
                    f"Page {range_start}"
                    if range_start == range_end
                    else f"Pages {range_start}-{range_end}"
                )
                page_provenance = "Original PDF"
            elif range_start is not None:
                page_range = (
                    f"Starts on page {range_start}; end page not present in source metadata"
                )
                page_provenance = "Incomplete original-PDF page provenance"
            else:
                page_range = None
                page_provenance = "Original-PDF page provenance not present"

            char_starts = [
                item["sourceCharStart"] for item in items if item["sourceCharStart"] is not None
            ]
            char_ends = [
                item["sourceCharEnd"] for item in items if item["sourceCharEnd"] is not None
            ]

            events.append(
                {
                    "date": first_item["date"],
                    "dateRaw": first_item["date"],
                    "dateNormalized": first_item["dateNormalized"],
                    "dateStatus": ("parsed" if first_item["dateNormalized"] else "unparsed"),
                    "title": first_item["docType"],
                    "physician": first_item["physician"],
                    "clinic": first_item["clinic"],
                    "docType": first_item["docType"],
                    "pageRange": page_range,
                    "pageStart": range_start,
                    "pageEnd": range_end,
                    "pageProvenance": page_provenance,
                    "sourceCharStart": min(char_starts) if char_starts else None,
                    "sourceCharEnd": max(char_ends) if char_ends else None,
                    "originalFilename": first_item["originalFilename"],
                    "refNo": first_item["refNo"],
                    "summary": first_item["summary"],
                }
            )

        events.sort(
            key=lambda event: (
                event["dateNormalized"] is None,
                event["dateNormalized"] or "",
                event["dateRaw"] or "",
            )
        )

    except Exception:
        pass

    return events
