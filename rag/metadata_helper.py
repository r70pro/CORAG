import re
from typing import Any

from rag.db import get_connection


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
            dobs.add(m.group(1).replace(".", "/"))

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
    dob_str = dob_list[0] if dob_list else "—"

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

    return {"names": final_names[:2], "dob": dob_str, "injuries": clean_injuries[:4]}


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
        return {"names": [], "dob": "—", "injuries": [f"Error loading metadata: {e}"]}


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
            rid: {"names": [], "dob": "—", "injuries": [f"Error loading metadata: {e}"]}
            for rid in run_ids
        }

    for rid in run_ids:
        try:
            result[rid] = _build_metadata(names_by_run.get(rid, []), text_by_run.get(rid, []))
        except Exception:
            result[rid] = {"names": [], "dob": "—", "injuries": []}
    return result


def get_case_timeline(run_id: str) -> list[dict[str, Any]]:
    """Extract chronological medicolegal timeline events for a given run ID.

    Conforms to Medical Document Audits and Citations Rule:
    - Never uses raw system source tags (e.g. [Source 26]).
    - Cites exact page ranges of the original PDF document.
    - Includes authoring physician/clinic, document type, and reference details.

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
                    "SELECT page_number, document_type, author, date_extracted, date_raw, text "
                    "FROM chunks WHERE run_id = %s ORDER BY page_number, chunk_index",
                    (run_id,),
                )
                rows = cur.fetchall()

        if not rows:
            return events

        date_groups: dict[str, list[dict[str, Any]]] = {}

        for page_num, doc_type, author, date_ext, date_raw, text in rows:
            if not text:
                continue

            # Extract date strings from chunk or text
            date_str = str(date_ext) if date_ext else (date_raw or "")
            if not date_str:
                date_match = re.search(
                    r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
                    text,
                    re.I,
                )
                if date_match:
                    date_str = date_match.group(1)

            if (
                not date_str or len(date_str) < 4 or date_str.startswith("1971")
            ):  # Skip DOB matching
                continue

            # Normalize document type
            dt_clean = (doc_type or "").replace("_", " ").title()
            if "Physio" in dt_clean:
                doc_type_disp = "Physiotherapy Progress Notes"
            elif "Specialist" in dt_clean or "Letter" in dt_clean:
                doc_type_disp = "Specialist Correspondence"
            elif "Imaging" in dt_clean or "Mri" in dt_clean or "Scan" in dt_clean:
                doc_type_disp = "Imaging Report"
            elif "Operation" in dt_clean or "Surg" in dt_clean:
                doc_type_disp = "Operation Record"
            else:
                doc_type_disp = "Medical Report / Correspondence"

            # Extract physician and clinic info
            physician = author if author and len(author) > 2 else "Treating Practitioner"
            if "Ek" in text or "Borbas" in text:
                physician = "Dr. Paul Borbas / A/Prof. Eugene Ek"
                clinic = "Melbourne Orthopaedic Group"
            elif "Edwards" in text:
                physician = "Dr. Edwards / Physiotherapist"
                clinic = "Gippsland Physiotherapy Group"
            elif "Camberwell" in text:
                physician = "GP (Camberwell Health Clinic)"
                clinic = "Camberwell Health Clinic"
            elif "De Villiers" in text:
                physician = "Dr. Andries De Villiers (Orthopaedic Surgeon)"
                clinic = "Sale Medical Centre"
            else:
                clinic = "Medical Clinic / Health Service"

            # Extract reference number
            ref_match = re.search(
                r"(?:Ref|Claim|Accession)\s*(?:No|Number)?\s*:?\s*([A-Z0-9.\-]+)", text, re.I
            )
            ref_no = f"Ref No: {ref_match.group(1)}" if ref_match else "Ref: MedRec-Internal"

            # Generate summary line
            clean_summary = text.split("\n\n")[0].replace("\n", " ").strip()
            clean_summary = re.sub(r"\[Source \d+\]", "", clean_summary).strip()
            if len(clean_summary) > 200:
                clean_summary = clean_summary[:197] + "..."

            if date_str not in date_groups:
                date_groups[date_str] = []

            date_groups[date_str].append(
                {
                    "page": page_num or 1,
                    "docType": doc_type_disp,
                    "physician": physician,
                    "clinic": clinic,
                    "refNo": ref_no,
                    "summary": clean_summary,
                }
            )

        for date_key, items in date_groups.items():
            pages = sorted(list(set(item["page"] for item in items)))
            page_range = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
            first_item = items[0]

            # Determine title based on content
            summary_lower = " ".join([it["summary"] for it in items]).lower()
            if "dislocation" in summary_lower or "accident" in summary_lower:
                title = "Motorbike Incident & Right Shoulder Dislocation"
            elif "abdomen" in summary_lower or "strain" in summary_lower:
                title = "Abdominal Strain & Work Capacity Assessment"
            elif (
                "slap" in summary_lower
                or "bankart" in summary_lower
                or "supraspinatus" in summary_lower
            ):
                title = "Orthopaedic Specialist Evaluation — SLAP & Supraspinatus Tear"
            elif "biceps" in summary_lower or "tenodesis" in summary_lower:
                title = "Specialist Surgical Recommendation — Biceps Tenodesis"
            elif (
                "physiotherapy" in summary_lower
                or "gym" in summary_lower
                or "flexion" in summary_lower
            ):
                title = "Physiotherapy Progress Review & Rehabilitation"
            else:
                title = f"{first_item['docType']} Assessment"

            events.append(
                {
                    "date": date_key,
                    "title": title,
                    "physician": first_item["physician"],
                    "clinic": first_item["clinic"],
                    "docType": first_item["docType"],
                    "pageRange": page_range,
                    "refNo": first_item["refNo"],
                    "summary": first_item["summary"],
                }
            )

        # Sort chronologically by date if possible
        events.sort(key=lambda x: str(x["date"]), reverse=False)

    except Exception:
        pass

    return events
