"""Low-latency quality controls for interactive Free Q&A.

The functions in this module are deliberately deterministic.  They improve
planning, evidence discipline and completion budgeting without adding another
model call to the latency-sensitive Free Q&A path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"
)
_MULTIPART_RE = re.compile(r"(?:^|\s)(?:\([a-z]\)|[a-z]\)|\d+[.)])\s", re.I)
_BROAD_TERMS = (
    "comprehensive",
    "complete",
    "all records",
    "every record",
    "chronology",
    "timeline",
    "entire history",
    "full history",
)
_COMPARISON_TERMS = ("compare", "difference", "conflict", "inconsisten", "versus", " vs ")


@dataclass(frozen=True)
class FreeQAPlan:
    task: str
    broad_scope: bool
    multipart: bool
    evidence_count: int
    dated_evidence_count: int
    requested_output_tokens: int
    compact: bool


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    source_id: int
    document_date: str | None
    document_type: str | None
    author: str | None
    evidence_status: str
    date_conflict: bool
    substantive_clinical_content: bool = False


def _iso_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def build_evidence_ledger(results: Iterable[dict]) -> tuple[EvidenceLedgerEntry, ...]:
    """Summarize provenance and flag obvious index/document-date conflicts."""
    entries = []
    for index, item in enumerate(results, start=1):
        text = str(item.get("text", ""))
        doc_type = item.get("document_type")
        type_text = f"{doc_type or ''} {text[:300]}".lower()
        indirect = any(
            term in type_text for term in ("index", "attachment list", "document review")
        )
        substantive = any(
            term in text.lower()
            for term in (
                "history of injury",
                "presenting complaint",
                "symptom",
                "diagnos",
                "examination",
                "clinical finding",
                "treatment",
                "medication",
                "work capacity",
                "return to work",
                "prognosis",
                "radiology",
                "impression",
                "assessment",
            )
        )
        document_date = item.get("date_extracted") or item.get("document_date")
        parsed_document_date = _iso_date(document_date)
        referenced_dates = []
        for match in _DATE_RE.finditer(text):
            raw = match.group(0).replace("/", "-").replace(".", "-")
            parts = raw.split("-")
            if len(parts) == 3 and len(parts[0]) != 4:
                raw = f"{parts[2]}-{int(parts[1]):02d}-{int(parts[0]):02d}"
            parsed = _iso_date(raw)
            if parsed:
                referenced_dates.append(parsed)
        conflict = bool(
            indirect
            and parsed_document_date
            and any(referenced > parsed_document_date for referenced in referenced_dates)
        )
        entries.append(
            EvidenceLedgerEntry(
                source_id=index,
                document_date=str(document_date) if document_date else None,
                document_type=str(doc_type) if doc_type else None,
                author=str(item.get("author")) if item.get("author") else None,
                evidence_status="indirect index/reference"
                if indirect
                else "retrieved record excerpt",
                date_conflict=conflict,
                substantive_clinical_content=substantive,
            )
        )
    return tuple(entries)


def render_evidence_ledger(entries: Iterable[EvidenceLedgerEntry]) -> str:
    """Render a token-efficient ledger for the model's private working context."""
    rows = []
    for entry in entries:
        fields = [f"Source {entry.source_id}", entry.evidence_status]
        if entry.document_date:
            fields.append(f"document date {entry.document_date}")
        if entry.document_type:
            fields.append(f"type {entry.document_type}")
        if entry.author:
            fields.append(f"author {entry.author}")
        if entry.date_conflict:
            fields.append("DATE CONFLICT: later referenced date appears in an earlier index")
        if entry.substantive_clinical_content:
            fields.append("contains potentially substantive clinical content")
        rows.append("- " + "; ".join(fields))
    return "EVIDENCE LEDGER (provenance aid, not additional evidence):\n" + "\n".join(rows)


def classify_free_qa(query: str, results: Iterable[dict] = ()) -> FreeQAPlan:
    """Classify a request and choose a completion budget in constant local time."""
    lowered = f" {query.lower()} "
    evidence = list(results)
    broad = any(term in lowered for term in _BROAD_TERMS)
    multipart = len(_MULTIPART_RE.findall(query)) >= 2
    if "chronolog" in lowered or "timeline" in lowered:
        task = "chronology"
    elif any(term in lowered for term in _COMPARISON_TERMS):
        task = "comparison"
    elif "summar" in lowered or "history" in lowered:
        task = "summary"
    else:
        task = "factual_qa"

    dated = sum(bool(_DATE_RE.search(str(item.get("text", "")))) for item in evidence)
    complexity = len(evidence) + (dated * 2) + (8 if multipart else 0)
    if broad or complexity >= 40:
        output_tokens = 4096
    elif multipart or complexity >= 20:
        output_tokens = 2560
    else:
        output_tokens = 1536
    return FreeQAPlan(
        task=task,
        broad_scope=broad,
        multipart=multipart,
        evidence_count=len(evidence),
        dated_evidence_count=dated,
        requested_output_tokens=output_tokens,
        compact=broad or complexity >= 32,
    )


def build_quality_instructions(plan: FreeQAPlan) -> str:
    """Return concise task-specific drafting rules for the existing prompt."""
    lines = [
        "FREE Q&A EXECUTION PLAN:",
        f"- Task type: {plan.task}; retrieved excerpts: {plan.evidence_count}.",
        "- Draft a complete answer skeleton before writing, then answer every requested part.",
        "- Distinguish an underlying event date from a document date and from a date merely listed in a later index.",
        "- Label indexed-but-unseen material as indirect evidence; prefer the underlying primary record when supplied.",
        "- Separate documented fact, attributed source opinion, synthesis, and unresolved uncertainty.",
        "- Do not explain a conflicting or malformed date as a template/OCR/metadata error unless a source establishes that explanation.",
        "- Preserve every source date as written. Never replace, repair, or reassign a date merely because it appears implausible; give both values only when internal source text proves a correction.",
        "- Do not characterize the supplied corpus as mostly indexes, administrative material, or clinically insubstantial unless the evidence ledger actually establishes that quantitative description.",
        "- The retrieval set is evidence supplied for this answer, not proof of what the entire indexed corpus does or does not contain. Do not convert retrieval limitations into claims about all available documents.",
        "- Use only supplied [Source N] tags and keep each tag adjacent to the claim it supports.",
        "- Never call the answer comprehensive if the retrieved excerpts or available space do not cover the requested scope.",
    ]
    if plan.task == "chronology":
        lines.extend(
            [
                "- For chronology, prioritize substantive injury, symptoms, diagnosis, investigation, treatment, work-capacity, claim, decision, and outcome events—not a catalogue of filenames.",
                "- Present dated events oldest first, with a separate short section for date conflicts or uncertain dates.",
            ]
        )
    elif plan.task == "comparison":
        lines.append(
            "- Compare the same proposition across sources and state whether any difference is material."
        )
    if plan.compact:
        lines.append(
            "- Use compact entries or a table so full coverage fits; avoid repeating citation metadata already rendered by the application."
        )
    return "\n".join(lines)


def choose_generation_tokens(
    plan: FreeQAPlan,
    remaining_context: int,
    user_max_tokens: int | None,
) -> int:
    """Allocate enough room to finish while respecting user and context limits."""
    desired = user_max_tokens if user_max_tokens is not None else plan.requested_output_tokens
    return max(0, min(desired, remaining_context))


@dataclass(frozen=True)
class QualityFindings:
    truncated: bool
    invalid_source_ids: tuple[int, ...]
    claims_comprehensive_while_truncated: bool
    unsupported_corpus_characterization: bool = False
    speculative_date_error_claim: bool = False
    ungrounded_dates: tuple[str, ...] = ()


def _canonical_dates(value: str) -> set[str]:
    """Return conservative date identities; numeric day-first dates follow AU usage."""
    dates: set[str] = set()
    for match in _DATE_RE.finditer(value):
        raw = match.group(0).replace("/", "-").replace(".", "-")
        parts = raw.split("-")
        if len(parts[0]) == 4:
            candidate = raw
        else:
            candidate = f"{parts[2]}-{int(parts[1]):02d}-{int(parts[0]):02d}"
        if _iso_date(candidate):
            dates.add(candidate)
    return dates


def inspect_response(
    text: str,
    source_count: int,
    results: Iterable[dict] = (),
) -> QualityFindings:
    """Cheap post-generation checks usable by tests, telemetry and non-streaming calls."""
    from rag.analyzer import OUTPUT_LIMIT_WARNING, invalid_source_tag_indices

    truncated = OUTPUT_LIMIT_WARNING in text
    invalid = tuple(sorted(invalid_source_tag_indices(text, source_count)))
    claims_complete = bool(re.search(r"\b(?:comprehensive|complete)\b", text, re.I))
    evidence = list(results)
    ledger = build_evidence_ledger(evidence)
    indirect_count = sum(entry.evidence_status == "indirect index/reference" for entry in ledger)
    corpus_claim = bool(
        re.search(
            r"\b(?:documents?|records?|excerpts?)\b[^.]{0,100}\b"
            r"(?:primarily|mostly|largely)\b[^.]{0,80}\b"
            r"(?:index|indices|administrative|lack(?:ing)? substantive|insubstantial)",
            text,
            re.I,
        )
    )
    unsupported_corpus = bool(corpus_claim and ledger and indirect_count * 2 <= len(ledger))
    speculative_date_error = bool(
        re.search(
            r"\b(?:date|dates|metadata)\b[^.]{0,120}\b"
            r"(?:likely|probably|apparently|appear(?:s)? to be)\b[^.]{0,80}\b"
            r"(?:error|placeholder|ocr|template|default)",
            text,
            re.I,
        )
    )
    grounded_dates: set[str] = set()
    for item in evidence:
        grounded_dates.update(_canonical_dates(str(item.get("text", ""))))
        for key in ("date_extracted", "document_date", "date_raw"):
            grounded_dates.update(_canonical_dates(str(item.get(key, ""))))
    response_dates = _canonical_dates(text)
    ungrounded = tuple(sorted(response_dates - grounded_dates)) if evidence else ()
    return QualityFindings(
        truncated,
        invalid,
        truncated and claims_complete,
        unsupported_corpus,
        speculative_date_error,
        ungrounded,
    )
