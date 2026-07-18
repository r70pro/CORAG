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

    dob_list = sorted(dobs, key=len, reverse=True)  # Prefer 4-digit year format
    dob_str = dob_list[0] if dob_list else "—"

    generic_words = ["injury/condition", "our client", "condition", "the client"]
    clean_injuries = []
    for inj in injuries:
        if inj and not any(gen in inj.lower() for gen in generic_words):
            clean_injuries.append(inj[0].upper() + inj[1:])

    return {"names": final_names[:2], "dob": dob_str, "injuries": clean_injuries[:3]}


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
