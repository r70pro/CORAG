import re
from typing import Dict, Any
from rag.db import get_connection

def get_case_metadata(run_id: str) -> Dict[str, Any]:
    """Extract client names, DOB, and injury region/type for a given run from PostgreSQL chunks.

    Args:
        run_id: The OCR run identifier.

    Returns:
        Dict containing names, dob, and injuries.
    """
    dob_regex = re.compile(r'(?:DOB|Date of Birth)\s*:?\s*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})', re.IGNORECASE)
    
    # Heuristics patterns for injury / diagnosis extraction
    patterns = [
        re.compile(r'(?:diagnosis|injury|injuries)\s+(?:of|is|was|were)\s+([^.\n]{5,100})', re.I),
        re.compile(r'(?:diagnosed|sustained)\s+(?:with|a)\s+([^.\n]{5,100})', re.I),
        re.compile(r'Re:\s+[^.\n]+?(?:DOB|Date of Birth)[^.\n]*?\n([^.\n]{5,100})', re.I),
        re.compile(r'diagnosis\s*:\s*([^.\n]{5,100})', re.I),
        re.compile(r'injury\s*:\s*([^.\n]{5,100})', re.I)
    ]
    
    # Generic keywords to ignore (like addresses or generic headings)
    ignore_keywords = [
        "circuit", "landing", "road", "street", "highway", "avenue", "drive", "court",
        "tel", "phone", "ref", "dear", "patient", "client", "address", "phone", "mobile",
        "medicare", "age", "gentleman", "wife", "son", "letter", "referral"
    ]

    names = []
    dobs = set()
    injuries = []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Fetch unique patient names from chunks
                cur.execute("SELECT DISTINCT patient_name FROM chunks WHERE run_id = %s AND patient_name IS NOT NULL", (run_id,))
                for (name,) in cur.fetchall():
                    clean_name = name.strip()
                    if clean_name and len(clean_name) > 2:
                        title_name = clean_name.title()
                        if not any(k in title_name.lower() for k in ignore_keywords):
                            names.append(title_name)

                # Deduplicate and canonicalize names
                unique_names = list(set(names))
                unique_names.sort(key=len, reverse=True)
                final_names = []
                for n in unique_names:
                    # Strip common titles
                    stripped_n = re.sub(r'^(mr|mrs|ms|dr|prof)\.?\s+', '', n, flags=re.I).strip()
                    if stripped_n not in final_names and all(stripped_n not in existing for existing in final_names):
                        final_names.append(stripped_n)

                # 2. Fetch chunk text to search for DOB and injuries
                cur.execute("SELECT text FROM chunks WHERE run_id = %s", (run_id,))
                rows = cur.fetchall()

                for (text,) in rows:
                    # Search DOB
                    m = dob_regex.search(text)
                    if m:
                        dobs.add(m.group(1).replace('.', '/'))

                    # Search Injury/Diagnosis
                    for p in patterns:
                        for match in p.finditer(text):
                            phrase = match.group(1).strip()
                            phrase = re.sub(r'\s+', ' ', phrase)
                            if any(k in phrase.lower() for k in ignore_keywords):
                                continue
                            
                            # Clean leading punctuation, list bullets, etc.
                            phrase = re.sub(r'^(?:[-*•\s\d]+|[A-Za-z]\)\s*)', '', phrase).strip()
                            # Clean trailing punctuation
                            phrase = phrase.rstrip(',.;:-"\'')
                            
                            if len(phrase) > 5 and len(phrase) < 90:
                                if not any(phrase.lower() == existing.lower() for existing in injuries):
                                    injuries.append(phrase)

        # Normalise DOB selection
        dob_list = list(dobs)
        dob_list.sort(key=len, reverse=True) # Prefer 4-digit year format (longer string length)
        dob_str = dob_list[0] if dob_list else "—"

        # Select distinct and clean injury descriptions
        clean_injuries = []
        generic_words = ["injury/condition", "our client", "condition", "the client"]
        for inj in injuries:
            if not any(gen in inj.lower() for gen in generic_words):
                if inj:
                    # Capitalise first letter
                    inj = inj[0].upper() + inj[1:]
                clean_injuries.append(inj)

        return {
            "names": final_names[:2],
            "dob": dob_str,
            "injuries": clean_injuries[:3]
        }
    except Exception as e:
        # Fallback in case of DB or connection issues
        return {
            "names": [],
            "dob": "—",
            "injuries": [f"Error loading metadata: {e}"]
        }
