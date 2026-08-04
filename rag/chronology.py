"""Complete-case, structured medicolegal chronology generation.

Unlike ordinary RAG, chronology generation may not discard evidence based on
semantic relevance.  This module enumerates the entire selected case, extracts
strict JSON events in bounded batches, validates provenance and dates, merges
only clear duplicates, and renders an auditable chronology.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rag import db as rag_db

logger = logging.getLogger(__name__)

MISSING = "Not documented in the available text"
NOT_APPLICABLE = "Not applicable—administrative event"
MAX_BATCH_CHARACTERS = 48_000
FAST_MAX_BATCH_CHARACTERS = 12_000
FAST_MAX_CHUNKS_PER_BATCH = 3
MAX_EVENTS_PER_BATCH = 100
FAST_EVENTS_PER_PAGE = 12
FAST_MAX_PAGES_PER_BATCH = 50
FAST_DEFAULT_CONCURRENCY = 20
ULTRA_CANDIDATES_PER_REQUEST = 30
CHRONOLOGY_SCHEMA_VERSION = 1
FAST_CHECKPOINT_VERSION = 3
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "workspace" / "runtime" / "chronology"
_ALLOWED_PRECISIONS = {"day", "month", "year", "unknown"}
_UNSUPPORTED_INFERENCE = re.compile(
    r"\b(likely|presumably|apparently|probably|routine|suggests that)\b", re.IGNORECASE
)
_MONTHS = {
    name.lower(): index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1
    )
}
_MONTHS.update({name[:3]: number for name, number in list(_MONTHS.items())})
_MONTH_PATTERN = "|".join(list(_MONTHS) + [name[:3] for name in _MONTHS])
_DOCUMENTED_DATE = re.compile(
    rf"\b(?:"
    rf"(?P<iso>(?:19|20)\d{{2}}-[01]\d-[0-3]\d)|"
    rf"(?P<numeric>[0-3]?\d[./-][01]?\d[./-](?:19|20)\d{{2}})|"
    rf"(?P<dmy>[0-3]?\d(?:st|nd|rd|th)?\s+(?:{_MONTH_PATTERN})\.?\s+(?:19|20)\d{{2}})|"
    rf"(?P<mdy>(?:{_MONTH_PATTERN})\.?\s+[0-3]?\d(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}})|"
    rf"(?P<month>(?:{_MONTH_PATTERN})\.?\s+(?:19|20)\d{{2}})|"
    rf"(?P<year>(?:19|20)\d{{2}})"
    rf")\b",
    re.IGNORECASE,
)
_ULTRA_EVENT_SIGNAL = re.compile(
    r"\b(?:accident|admitted|admission|assessment|attended|capacity|certificate|claim|"
    r"complain(?:ed|s)?|consult(?:ation|ed)?|diagnos(?:is|ed)|discharg(?:e|ed)|doctor|"
    r"examin(?:ation|ed)|fracture|headache|hospital|imaging|impairment|injur(?:y|ed)|"
    r"injection|investigation|medication|mri|operation|pain|physio(?:therapy)?|procedure|"
    r"reported|review(?:ed)?|scan|specialist|surgery|symptom|treatment|unfit|work(?:cover)?|"
    r"x-?ray|ultrasound|prescri(?:be|bed|ption)|referr(?:al|ed))\b",
    re.IGNORECASE,
)
_ULTRA_NON_EVENT = re.compile(
    r"\b(?:date of birth|birthdate|d\.?o\.?b\.?|items? (?:exported|from)|"
    r"subpoena generated|printed on|expiry date|civil procedure act|phone|fax|tel:|"
    r"training/courses?|qualification/course|employer name|job title|driver'?s licence|"
    r"medical records? from|request for (?:a )?copy|following up on (?:our|the) (?:previous )?request|"
    r"consultation notes? [–—-]|workplace injury rehabilitation (?:and compensation )?act|"
    r"entitled to compensation|proceedings for damages)\b",
    re.IGNORECASE,
)
_ULTRA_YEAR_SIGNAL = re.compile(
    r"\b(?:accident|admitted|ceased|commenced|diagnosed|felt|injur(?:y|ed)|pain|"
    r"prescribed|reported|returned|surgery|symptom|treated|treatment|valium|work-related)\b",
    re.IGNORECASE,
)


@dataclass
class ChronologyEvent:
    event_date: str = ""
    date_precision: str = "unknown"
    date_original: str = ""
    event_type: str = "clinical event"
    provider: str = MISSING
    facility: str = MISSING
    presenting_symptoms: str = MISSING
    diagnosis: str = MISSING
    investigations_findings: str = MISSING
    intervention_plan: str = MISSING
    administrative_context: str = ""
    source_ids: list[int] = field(default_factory=list)
    source_quote: str = ""
    warnings: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Any, source_count: int) -> ChronologyEvent | None:
        if not isinstance(value, dict):
            return None
        value = dict(value)
        aliases = {
            "d": "event_date", "p": "date_precision", "od": "date_original",
            "t": "event_type", "pr": "provider", "f": "facility",
            "s": "presenting_symptoms", "dx": "diagnosis",
            "ix": "investigations_findings", "ip": "intervention_plan",
            "a": "administrative_context", "src": "source_ids", "q": "source_quote",
        }
        for short, full in aliases.items():
            if full not in value and short in value:
                value[full] = value[short]

        def clean(key: str, default: str = MISSING) -> str:
            raw = value.get(key)
            text = str(raw).strip() if raw is not None else ""
            return text or default

        raw_ids = value.get("source_ids", [])
        if isinstance(raw_ids, int):
            raw_ids = [raw_ids]
        source_ids: list[int] = []
        if isinstance(raw_ids, list):
            for raw in raw_ids:
                try:
                    source_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if 1 <= source_id <= source_count and source_id not in source_ids:
                    source_ids.append(source_id)
        if not source_ids:
            return None

        precision = clean("date_precision", "unknown").lower()
        if precision not in _ALLOWED_PRECISIONS:
            precision = "unknown"
        event = cls(
            event_date=clean("event_date", ""),
            date_precision=precision,
            date_original=clean("date_original", ""),
            event_type=clean("event_type", "clinical event"),
            provider=clean("provider"),
            facility=clean("facility"),
            presenting_symptoms=clean("presenting_symptoms"),
            diagnosis=clean("diagnosis"),
            investigations_findings=clean("investigations_findings"),
            intervention_plan=clean("intervention_plan"),
            administrative_context=clean("administrative_context", ""),
            source_ids=source_ids,
            source_quote=clean("source_quote", ""),
        )
        event.event_date, event.date_precision, date_warning = normalize_event_date(
            event.event_date, event.date_precision
        )
        if date_warning:
            event.warnings.append(date_warning)
        for field_name in (
            "provider", "facility", "presenting_symptoms", "diagnosis",
            "investigations_findings", "intervention_plan", "administrative_context",
        ):
            text = getattr(event, field_name)
            match = _UNSUPPORTED_INFERENCE.search(text) if text else None
            if match and match.group(0).lower() not in event.source_quote.lower():
                setattr(event, field_name, MISSING)
                event.warnings.append(
                    f"Removed potentially unsupported inference from {field_name}"
                )
        return event


@dataclass
class ChronologyAudit:
    documents_total: int
    documents_audited: int
    chunks_total: int
    batches_total: int
    profile: str = "thorough"
    batches_processed: int = 0
    batches_failed: int = 0
    pages_completed: int = 0
    chunks_completed: int = 0
    events_extracted: int = 0
    events_rendered: int = 0
    rejected_without_source: int = 0
    warnings: list[str] = field(default_factory=list)


def normalize_event_date(value: str, precision: str) -> tuple[str, str, str | None]:
    """Validate an ISO extraction date without inventing missing components."""
    raw = str(value or "").strip()
    if not raw or precision == "unknown":
        return "", "unknown", "Event date is not available as a normalised value"
    patterns = {
        "day": (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        "month": (r"^\d{4}-\d{2}$", "%Y-%m"),
        "year": (r"^\d{4}$", "%Y"),
    }
    pattern, fmt = patterns[precision]
    if not re.fullmatch(pattern, raw):
        return "", "unknown", f"Invalid {precision}-precision event date: {raw}"
    try:
        datetime.strptime(raw, fmt)
    except ValueError:
        return "", "unknown", f"Invalid calendar date: {raw}"
    return raw, precision, None


def display_date(event: ChronologyEvent) -> str:
    if event.date_precision == "day" and event.event_date:
        return datetime.strptime(event.event_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    if event.date_precision == "month" and event.event_date:
        return datetime.strptime(event.event_date, "%Y-%m").strftime("%m/%Y") + " [Date Incomplete]"
    if event.date_precision == "year" and event.event_date:
        return event.event_date + " [Date Incomplete]"
    original = event.date_original.strip()
    return f"{original} [Date Incomplete]" if original else "[Date Incomplete]"


def _date_sort_key(event: ChronologyEvent) -> tuple:
    if event.date_precision == "day":
        parts = [int(part) for part in event.event_date.split("-")]
        return (*parts, 0, event.provider.lower(), event.event_type.lower())
    if event.date_precision == "month":
        year, month = (int(part) for part in event.event_date.split("-"))
        return (year, month, 1, 1, event.provider.lower(), event.event_type.lower())
    if event.date_precision == "year":
        return (int(event.event_date), 1, 1, 2, event.provider.lower(), event.event_type.lower())
    return (9999, 12, 31, 3, event.provider.lower(), event.event_type.lower())


def _source_identity(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or "")


def prepare_complete_case(run_id: str) -> tuple[list[dict[str, Any]], int, int]:
    chunks = rag_db.get_chunks_for_run(run_id)
    documents = rag_db.get_documents_for_run(run_id)
    # Exact repeated chunks arise in some re-indexed/bundled records.  Retain
    # separate documents but remove an accidental duplicate within one source.
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        key = (str(chunk.get("doc_id") or ""), digest)
        if key in seen:
            continue
        seen.add(key)
        item = dict(chunk)
        if prepared and prepared[-1].get("doc_id") == item.get("doc_id"):
            previous_end = prepared[-1].get("char_end")
            current_start = item.get("char_start")
            if isinstance(previous_end, int) and isinstance(current_start, int):
                overlap = max(0, previous_end - current_start)
                if 0 < overlap < len(text):
                    item["text"] = text[overlap:]
                    item["char_start"] = current_start + overlap
        prepared.append(item)
    documents_with_chunks = {str(chunk.get("doc_id") or "") for chunk in prepared}
    return prepared, len(documents), len(documents_with_chunks)


def build_batches(
    chunks: list[dict[str, Any]],
    max_characters: int = MAX_BATCH_CHARACTERS,
    max_chunks: int | None = None,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for chunk in chunks:
        chunk_size = len(str(chunk.get("text") or "")) + 600
        if current and (
            size + chunk_size > max_characters
            or (max_chunks is not None and len(current) >= max_chunks)
        ):
            batches.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += chunk_size
    if current:
        batches.append(current)
    return batches


def chronology_checkpoint_dir(run_id: str) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))[:128]
    return CHECKPOINT_ROOT / safe_run_id


def _batch_checkpoint_path(
    directory: Path,
    batch: list[dict[str, Any]],
    detail: str = "thorough",
    page: int | None = None,
) -> Path:
    digest = hashlib.sha256()
    digest.update(f"chronology-schema-{CHRONOLOGY_SCHEMA_VERSION}".encode())
    if detail != "thorough":
        digest.update(f"-profile-{detail}-{FAST_CHECKPOINT_VERSION}".encode())
    for chunk in batch:
        digest.update(str(chunk.get("chunk_id") or "").encode())
        digest.update(str(chunk.get("text") or "").encode())
    suffix = f"-page-{page:04d}" if page is not None else ""
    return directory / f"batch-{digest.hexdigest()}{suffix}.json"


def _load_checkpoint(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        response = value.get("response") if isinstance(value, dict) else None
        return response if isinstance(response, str) else None
    except (OSError, ValueError):
        return None


def _save_checkpoint(path: Path, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".chronology-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": CHRONOLOGY_SCHEMA_VERSION, "response": response},
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fast_response_format() -> dict[str, Any]:
    """OpenAI-compatible strict schema used to constrain every fast page."""
    missing_or_text = {"type": "string"}
    event = {
        "type": "object",
        "additionalProperties": False,
        "required": ["d", "p", "od", "t", "pr", "s", "dx", "ip", "src", "q"],
        "properties": {
            "d": {"type": "string"},
            "p": {"type": "string", "enum": ["day", "month", "year", "unknown"]},
            "od": {"type": "string"},
            "t": missing_or_text,
            "pr": missing_or_text,
            "s": missing_or_text,
            "dx": missing_or_text,
            "ip": missing_or_text,
            "src": {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "integer"}},
            "q": {"type": "string"},
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["events", "complete"],
        "properties": {
            "events": {
                "type": "array",
                "maxItems": FAST_EVENTS_PER_PAGE,
                "items": event,
            },
            "complete": {"type": "boolean"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "chronology_page", "strict": True, "schema": schema},
    }


def _call_llm(
    llm: Callable[..., str],
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
) -> str:
    """Pass generation constraints when supported; retain simple test/custom callables."""
    if response_format is not None:
        try:
            signature = inspect.signature(llm)
            accepts_keyword = "response_format" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_keyword = True
        if accepts_keyword:
            return llm(messages, response_format=response_format)
    return llm(messages)


def _extract_validated_batch(
    batch: list[dict[str, Any]],
    llm: Callable[[list[dict[str, str]]], str],
    checkpoint_dir: Path | None,
    repair_callback: Callable[[], None] | None = None,
    detail: str = "thorough",
) -> tuple[list[ChronologyEvent], int, bool]:
    """Extract one batch, retrying malformed structured output exactly once."""
    checkpoint_path = (
        _batch_checkpoint_path(checkpoint_dir, batch, detail) if checkpoint_dir else None
    )
    response = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    if response is None:
        response = llm(extraction_messages(batch, detail))
    events, rejected = parse_batch_response(response, batch)
    structured = _extract_json(response)
    truncated = "Incomplete response" in response
    needs_repair = (
        not structured
        or not isinstance(structured.get("events"), list)
        or rejected > 0
    )
    if needs_repair and not truncated:
        if repair_callback:
            repair_callback()
        repair = extraction_messages(batch, detail)
        repair.append({
            "role": "user",
            "content": (
                f"Your previous response was invalid or contained {rejected} rejected event(s). "
                "Return one corrected JSON object with an events array only. Preserve every "
                "documented event. Every event must use exactly one source ID and one short, "
                "contiguous verbatim quotation copied from that same source. Never join quotation "
                "fragments with ellipses. Use the exact required missing-value wording. Do not add Markdown."
            ),
        })
        response = llm(repair)
        events, rejected = parse_batch_response(response, batch)
        structured = _extract_json(response)
    valid = bool(
        structured
        and isinstance(structured.get("events"), list)
        and rejected == 0
    )
    if checkpoint_path and valid:
        _save_checkpoint(checkpoint_path, response)
    return events, rejected, valid


def _fast_page_messages(
    batch: list[dict[str, Any]],
    page: int,
    previous_events: list[ChronologyEvent],
) -> list[dict[str, str]]:
    messages = extraction_messages(batch, "fast")
    prior = [
        {
            "d": event.event_date,
            "t": event.event_type,
            "src": event.source_ids,
            "q": event.source_quote,
        }
        for event in previous_events
    ]
    messages[-1]["content"] += (
        f"\n\nPAGE REQUEST {page}. Return at most {FAST_EVENTS_PER_PAGE} events. "
        "Set complete=true only after every distinct event in these sources has been returned. "
        "If more remain, set complete=false; the next request will use the next page number. "
        "Order events deterministically by source ID, then textual occurrence, then event date."
    )
    if prior:
        messages[-1]["content"] += (
            "\nEvents already checkpointed on earlier pages; continue strictly after these and do not repeat them:\n"
            + json.dumps(prior, ensure_ascii=False)
        )
    return messages


def _extract_fast_pages(
    batch: list[dict[str, Any]],
    llm: Callable[..., str],
    checkpoint_dir: Path | None,
    page_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[ChronologyEvent], int, int, bool]:
    """Generate, validate, and immediately checkpoint deterministic fast pages."""
    accepted: list[ChronologyEvent] = []
    rejected_total = 0
    pages_completed = 0
    for page in range(1, FAST_MAX_PAGES_PER_BATCH + 1):
        checkpoint_path = (
            _batch_checkpoint_path(checkpoint_dir, batch, "fast", page)
            if checkpoint_dir else None
        )
        response = _load_checkpoint(checkpoint_path) if checkpoint_path else None
        if response is None:
            response = _call_llm(
                llm,
                _fast_page_messages(batch, page, accepted),
                fast_response_format(),
            )
        # Output-limit responses are intentionally not retried: the page is
        # invalid and the already checkpointed pages remain safely resumable.
        if "Incomplete response" in response:
            return accepted, rejected_total, pages_completed, False
        payload = _extract_json(response)
        raw_events = payload.get("events") if payload else None
        if (
            not isinstance(raw_events, list)
            or len(raw_events) > FAST_EVENTS_PER_PAGE
            or not isinstance(payload.get("complete"), bool)
        ):
            return accepted, rejected_total, pages_completed, False
        events, rejected = parse_batch_response(response, batch)
        rejected_total += rejected
        if rejected or len(events) != len(raw_events):
            return accepted, rejected_total, pages_completed, False
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, response)
        accepted.extend(events)
        pages_completed += 1
        if page_callback:
            page_callback(pages_completed, len(accepted))
        if payload["complete"]:
            return accepted, rejected_total, pages_completed, True
        if not events:
            return accepted, rejected_total, pages_completed, False
    return accepted, rejected_total, pages_completed, False


def _batch_context(batch: list[dict[str, Any]]) -> str:
    blocks = []
    for index, chunk in enumerate(batch, 1):
        metadata = {
            "source_id": index,
            "document": chunk.get("original_filename"),
            "document_type": chunk.get("document_type"),
            "author": chunk.get("author"),
            "metadata_date": str(chunk.get("date_extracted") or ""),
            "metadata_date_raw": chunk.get("date_raw"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "source_char_start": chunk.get("source_char_start"),
            "source_char_end": chunk.get("source_char_end"),
        }
        blocks.append(f"SOURCE {index} METADATA\n{json.dumps(metadata, ensure_ascii=False)}\nTEXT\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def extraction_messages(
    batch: list[dict[str, Any]], detail: str = "thorough"
) -> list[dict[str, str]]:
    thorough_schema = {
        "events": [{
            "event_date": "YYYY-MM-DD, YYYY-MM, YYYY, or empty",
            "date_precision": "day | month | year | unknown",
            "date_original": "verbatim date expression",
            "event_type": "consultation/procedure/investigation/etc",
            "provider": MISSING,
            "facility": MISSING,
            "presenting_symptoms": MISSING,
            "diagnosis": MISSING,
            "investigations_findings": MISSING,
            "intervention_plan": MISSING,
            "administrative_context": "",
            "source_ids": [1],
            "source_quote": "short verbatim supporting excerpt",
        }]
    }
    fast_schema = {
        "events": [{
            "d": "YYYY-MM-DD, YYYY-MM, YYYY, or empty",
            "p": "day | month | year | unknown",
            "od": "verbatim date",
            "t": "short event type",
            "pr": f"specific documented provider/facility or exactly: {MISSING}",
            "s": f"brief symptoms/context or exactly: {MISSING}",
            "dx": f"brief diagnosis/finding or exactly: {MISSING}",
            "ip": f"specific documented action/plan or exactly: {MISSING}",
            "src": [1],
            "q": "one short contiguous verbatim quotation from the single cited source",
        }],
        "complete": "boolean",
    }
    schema = fast_schema if detail == "fast" else thorough_schema
    profile_instruction = (
        "FAST PROFILE: Be extremely concise. Use the short JSON keys exactly as shown. "
        f"Limit each descriptive value to 18 words and each page to {FAST_EVENTS_PER_PAGE} events."
        if detail == "fast"
        else "THOROUGH PROFILE: Populate every separate clinical field with all material source-supported detail."
    )
    system = f"""You are a forensic medical-record event extractor. Extract every clinical event and every material administrative event documented in the supplied sources. Return JSON only, matching this schema: {json.dumps(schema)}

Mandatory rules:
 - {profile_instruction}
- Treat source metadata dates as metadata only. Use an event date only when the TEXT supports it.
- Make a separate event for each distinct date or encounter. Separate an order from a later result.
- Never infer. Do not use words such as likely, routine, apparently, presumably or probably unless quoted.
- Keep presenting symptoms, diagnosis, investigations/findings and intervention/plan separate.
- For a missing field write exactly: {MISSING!r}.
- For an administrative-only event, clinical fields may use exactly: {NOT_APPLICABLE!r}.
- A referenced report title without substantive contents is administrative context, not proof of its clinical findings.
- Preserve partial or ambiguous dates with date_precision and date_original; never invent a day or month.
- Each event must cite at least one valid source_ids value and include a short supporting source_quote.
- Every event must cite exactly one source ID. Its quotation must be copied contiguously from that source only.
- Never combine quotation fragments, insert an ellipsis, or paraphrase inside the quotation field.
- When the text names a provider, facility, investigation, treatment or action, use that specific wording; do not replace it with generic labels such as "doctor", "consultation", "test" or "treatment".
- Do not add a summary or Markdown."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _batch_context(batch)},
    ]


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError):
        return None


def parse_batch_response(text: str, batch: list[dict[str, Any]]) -> tuple[list[ChronologyEvent], int]:
    payload = _extract_json(text)
    if not payload or not isinstance(payload.get("events"), list):
        return [], 0
    events: list[ChronologyEvent] = []
    rejected = 0
    for raw in payload["events"][:MAX_EVENTS_PER_BATCH]:
        if isinstance(raw, dict) and any(
            isinstance(value, str)
            and value.strip().lower() in {"missing-value phrase", "missing value phrase"}
            for value in raw.values()
        ):
            rejected += 1
            continue
        event = ChronologyEvent.from_mapping(raw, len(batch))
        if event is None:
            rejected += 1
            continue
        event.sources = [batch[index - 1] for index in event.source_ids]
        if len(event.source_ids) != 1 or "..." in event.source_quote or "…" in event.source_quote:
            rejected += 1
            continue
        quote = " ".join(event.source_quote.split()).lower()
        if not quote or not any(
            quote in " ".join(str(source.get("text") or "").split()).lower()
            for source in event.sources
        ):
            rejected += 1
            continue
        events.append(event)
    return events, rejected


def _event_fingerprint(event: ChronologyEvent) -> tuple[str, ...]:
    def normalise(value: str) -> str:
        return re.sub(r"\W+", " ", value.lower()).strip()

    return (
        event.event_date,
        event.date_precision,
        normalise(event.provider),
        normalise(event.event_type),
        normalise(event.presenting_symptoms),
        normalise(event.diagnosis),
        normalise(event.intervention_plan),
    )


def deduplicate_events(events: Iterable[ChronologyEvent]) -> list[ChronologyEvent]:
    merged: dict[tuple[str, ...], ChronologyEvent] = {}
    for event in events:
        key = _event_fingerprint(event)
        existing = merged.get(key)
        if existing is None:
            merged[key] = event
            continue
        known = {_source_identity(source) for source in existing.sources}
        existing.sources.extend(
            source for source in event.sources if _source_identity(source) not in known
        )
        existing.warnings.extend(w for w in event.warnings if w not in existing.warnings)
    return sorted(merged.values(), key=_date_sort_key)


def _escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _source_label(source: dict[str, Any]) -> str:
    parts = [str(source.get("original_filename") or "Indexed case record")]
    start, end = source.get("page_start"), source.get("page_end")
    if start is not None and end is not None:
        parts.append(f"p. {start}" if start == end else f"pp. {start}–{end}")
    elif source.get("source_char_start") is not None and source.get("source_char_end") is not None:
        parts.append(f"chars {source['source_char_start']}–{source['source_char_end']}")
    return ", ".join(parts)


def render_chronology(events: list[ChronologyEvent], audit: ChronologyAudit) -> str:
    if audit.profile in {"fast", "ultra_fast"}:
        lines = [
            f"## Comprehensive Medicolegal Chronology — {'Ultra-Fast' if audit.profile == 'ultra_fast' else 'Fast'} Profile",
            "",
            "| Date | Provider/Facility | Event Summary | Intervention/Plan | Source |",
            "|---|---|---|---|---|",
        ]
        if audit.profile == "ultra_fast":
            lines.insert(1, "")
            lines.insert(2, "> ⚡ Whole-case dated-text index with bounded semantic filtering. It may not resolve implicit dates or separate multiple events described in one excerpt.")
        for event in events:
            provider = event.provider
            if event.facility not in {MISSING, NOT_APPLICABLE, ""}:
                provider = f"{provider}; {event.facility}" if provider != MISSING else event.facility
            summary_parts = [event.event_type]
            summary_parts.extend(
                value for value in (
                    event.presenting_symptoms,
                    event.diagnosis,
                    event.investigations_findings,
                )
                if value not in {MISSING, NOT_APPLICABLE, ""}
            )
            sources = "; ".join(_source_label(source) for source in event.sources)
            lines.append(
                "| " + " | ".join(_escape(value) for value in (
                    display_date(event), provider, "; ".join(summary_parts),
                    event.intervention_plan, sources,
                )) + " |"
            )
    else:
        lines = [
        "## Comprehensive Medicolegal Chronology",
        "",
        "| Date | Provider/Facility | Presenting Symptoms | Diagnosis | Investigations/Findings | Intervention/Plan | Source |",
        "|---|---|---|---|---|---|---|",
        ]
        for event in events:
            provider = event.provider
            if event.facility not in {MISSING, NOT_APPLICABLE, ""}:
                provider = f"{provider}; {event.facility}" if provider != MISSING else event.facility
            sources = "; ".join(_source_label(source) for source in event.sources)
            lines.append(
                "| " + " | ".join(_escape(value) for value in (
                    display_date(event), provider, event.presenting_symptoms,
                    event.diagnosis, event.investigations_findings,
                    event.intervention_plan, sources,
                )) + " |"
            )
    lines.extend([
        "", "### Completeness and data-quality audit", "",
        f"- Chronology profile: **{audit.profile.title()}**",
        f"- Documents audited: **{audit.documents_audited}/{audit.documents_total}**",
        f"- Source chunks audited: **{audit.chunks_total}/{audit.chunks_total}**",
        f"- Extraction batches processed: **{audit.batches_processed}/{audit.batches_total}**",
        f"- Valid extraction pages checkpointed: **{audit.pages_completed}**",
        f"- Source chunks completed: **{audit.chunks_completed}/{audit.chunks_total}**",
        f"- Events extracted before deduplication: **{audit.events_extracted}**",
        f"- Chronology events rendered: **{audit.events_rendered}**",
        f"- Events rejected for missing/invalid provenance or source quotation: **{audit.rejected_without_source}**",
        f"- Failed extraction batches: **{audit.batches_failed}**",
    ])
    warning_count = sum(len(event.warnings) for event in events)
    lines.append(f"- Entries carrying validation warnings: **{warning_count}**")
    if audit.documents_audited < audit.documents_total:
        lines.extend(["", "> ⚠️ **Incomplete chronology:** One or more indexed documents contained no auditable source chunks."])
    if audit.batches_failed:
        lines.extend(["", "> ⚠️ **Incomplete chronology:** One or more source batches could not be extracted. Do not rely on this output as a complete record audit."])
    if warning_count:
        lines.extend(["", "> ⚠️ Entries with incomplete or invalid dates, or potentially inferential language, require source review."])
    return "\n".join(lines)


def _parse_documented_date(match: re.Match[str]) -> tuple[str, str]:
    raw = match.group(0).strip()
    try:
        if match.lastgroup == "iso":
            return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d"), "day"
        if match.lastgroup == "numeric":
            parts = re.split(r"[./-]", raw)
            parsed = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
            return parsed.strftime("%Y-%m-%d"), "day"
        if match.lastgroup in {"dmy", "mdy"}:
            cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b|[,.]", "", raw, flags=re.IGNORECASE)
            tokens = cleaned.split()
            if match.lastgroup == "dmy":
                day, month_text, year = tokens
            else:
                month_text, day, year = tokens
            month = _MONTHS[month_text.lower()]
            return datetime(int(year), month, int(day)).strftime("%Y-%m-%d"), "day"
        if match.lastgroup == "month":
            month_text, year = raw.replace(".", "").split()
            month = _MONTHS[month_text.lower()]
            return f"{int(year):04d}-{month:02d}", "month"
        return raw, "year"
    except (KeyError, ValueError):
        return "", "unknown"


def _ultra_fast_events(
    chunks: list[dict[str, Any]],
    progress_callback: Callable[[float, str], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
) -> tuple[list[ChronologyEvent], int]:
    """Build a bounded, provenance-linked dated-text index without LLM calls."""
    events: list[ChronologyEvent] = []
    rejected = 0
    for chunk_index, chunk in enumerate(chunks):
        if cancellation_callback and cancellation_callback():
            break
        text = html.unescape(re.sub(r"<[^>]+>", " ", str(chunk.get("text") or "")))
        text = " ".join(text.split())
        seen: set[tuple[str, str]] = set()
        for match in _DOCUMENTED_DATE.finditer(text):
            event_date, precision = _parse_documented_date(match)
            if not event_date:
                rejected += 1
                continue
            left = max(text.rfind(". ", 0, match.start()), text.rfind("; ", 0, match.start()))
            right_candidates = [position for position in (text.find(". ", match.end()), text.find("; ", match.end())) if position >= 0]
            start = max(left + 2, match.start() - 140)
            end = min(min(right_candidates) + 1 if right_candidates else len(text), match.end() + 220)
            quote = text[start:end].strip()
            if len(quote) > 260:
                relative = match.start() - start
                clip_start = max(0, relative - 90)
                quote = quote[clip_start : clip_start + 260].strip(" ,;:-")
            key = (event_date, quote.lower())
            if not quote or key in seen:
                continue
            seen.add(key)
            lower = quote.lower()
            local_prefix = text[max(0, match.start() - 35) : match.start()]
            local_context = text[max(0, match.start() - 100) : min(len(text), match.end() + 150)]
            if re.search(r"(?:date of birth|birthdate|d\.?o\.?b\.?)\s*:?[\s\w,/-]*$", local_prefix, re.IGNORECASE):
                continue
            if _ULTRA_NON_EVENT.search(quote) or not _ULTRA_EVENT_SIGNAL.search(local_context):
                continue
            if len(list(_DOCUMENTED_DATE.finditer(quote))) >= 3:
                # Flattened problem lists and document inventories otherwise
                # create one clipped pseudo-event for every date in the row.
                continue
            if precision == "year":
                if (
                    not _ULTRA_YEAR_SIGNAL.search(local_context)
                    or len(re.findall(r"[A-Za-z]{3,}", quote)) < 5
                    or re.search(r"(?:\+61|@|\b(?:act|section|division|course|certificate)\b)", quote, re.IGNORECASE)
                ):
                    continue
            if any(word in lower for word in ("mri", "x-ray", "xray", "ct ", "ultrasound", "scan", "imaging")):
                event_type = "Investigation/imaging"
            elif any(word in lower for word in ("surgery", "operation", "procedure", "injection")):
                event_type = "Procedure/treatment"
            elif any(word in lower for word in ("claim", "insurer", "workcover", "compensation", "capacity")):
                event_type = "Administrative/work capacity"
            else:
                event_type = "Dated record event"
            provider = str(chunk.get("author") or MISSING).strip()
            if re.search(r"^(?:electronically approved by|unknown)|\bvisit$", provider, re.IGNORECASE):
                provider = MISSING
            events.append(ChronologyEvent(
                event_date=event_date,
                date_precision=precision,
                date_original=match.group(0),
                event_type=event_type,
                provider=provider,
                presenting_symptoms=quote,
                source_ids=[1],
                source_quote=quote,
                sources=[chunk],
            ))
        if progress_callback and (chunk_index + 1) % 100 == 0:
            progress_callback(
                0.08 + 0.72 * (chunk_index + 1) / len(chunks),
                f"Ultra-Fast scan: {chunk_index + 1}/{len(chunks)} chunks and {len(events)} dated excerpts completed...",
            )
    return events, rejected


def _deduplicate_ultra_events(events: list[ChronologyEvent]) -> list[ChronologyEvent]:
    """Remove only highly similar repeated excerpts on the same date."""
    accepted: list[ChronologyEvent] = []
    token_sets: dict[tuple[str, str], list[set[str]]] = {}
    for event in deduplicate_events(events):
        key = (event.event_date, event.event_type)
        tokens = set(re.findall(r"[a-z0-9]{3,}", event.source_quote.lower()))
        duplicates = token_sets.setdefault(key, [])
        if tokens and any(
            len(tokens & prior) / max(len(tokens | prior), 1) >= 0.92
            for prior in duplicates
        ):
            continue
        duplicates.append(tokens)
        accepted.append(event)
    return accepted


def _adjudicate_ultra_events(
    events: list[ChronologyEvent],
    llm: Callable[..., str],
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[ChronologyEvent]:
    """Use one bounded concurrent wave to reject deterministic false positives."""
    groups = [events[i:i + ULTRA_CANDIDATES_PER_REQUEST] for i in range(0, len(events), ULTRA_CANDIDATES_PER_REQUEST)]
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "timeline_candidate_filter", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": ["keep"],
            "properties": {"keep": {"type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": ULTRA_CANDIDATES_PER_REQUEST}}},
        }},
    }

    def adjudicate(index: int, group: list[ChronologyEvent]):
        candidates = [{"id": i, "date": e.event_date, "excerpt": e.source_quote} for i, e in enumerate(group, 1)]
        messages = [{"role": "system", "content": (
            "You are filtering a medicolegal chronology. Keep only substantive patient-specific clinical, "
            "treatment, investigation, injury, work-capacity, or claim events. Reject DOB/header repetitions, "
            "document inventories, legal boilerplate, contact details, form metadata, education/employment lists, "
            "and fragments that do not describe an event. Return JSON only."
        )}, {"role": "user", "content": json.dumps(candidates, ensure_ascii=False)}]
        response = _call_llm(llm, messages, schema)
        payload = _extract_json(response) or {}
        keep = payload.get("keep")
        if not isinstance(keep, list):
            logger.warning("Ultra-Fast adjudication group %s returned invalid JSON", index + 1)
            return index, []
        selected = {value for value in keep if isinstance(value, int) and 1 <= value <= len(group)}
        return index, [event for i, event in enumerate(group, 1) if i in selected]

    results: dict[int, list[ChronologyEvent]] = {}
    workers = min(FAST_DEFAULT_CONCURRENCY, len(groups))
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="ultra-chronology") as pool:
        futures = [pool.submit(adjudicate, index, group) for index, group in enumerate(groups)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, selected = future.result()
            results[index] = selected
            if progress_callback:
                progress_callback(0.80 + 0.12 * completed / len(groups), f"Ultra-Fast semantic filter: {completed}/{len(groups)} candidate groups completed...")
    return [event for index in range(len(groups)) for event in results[index]]


def generate_comprehensive_chronology(
    run_id: str,
    llm: Callable[[list[dict[str, str]]], str],
    progress_callback: Callable[[float, str], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
    checkpoint_dir: Path | None = None,
    detail: str = "thorough",
) -> str:
    detail = detail if detail in {"ultra_fast", "fast", "thorough"} else "fast"
    chunks, document_count, audited_document_count = prepare_complete_case(run_id)
    if not chunks:
        return "No indexed source chunks were found for the selected case."
    batches = (
        build_batches(chunks, FAST_MAX_BATCH_CHARACTERS, FAST_MAX_CHUNKS_PER_BATCH)
        if detail == "fast"
        else build_batches(chunks)
    )
    audit = ChronologyAudit(
        document_count, audited_document_count, len(chunks), len(batches), profile=detail
    )
    extracted: list[ChronologyEvent] = []
    if progress_callback:
        profile_description = (
            "compact complete-event extraction" if detail == "fast" else "detailed clinical-field extraction"
        )
        progress_callback(
            0.05,
            f"Enumerated {document_count} documents and {len(chunks)} source chunks for complete chronology audit. {detail.title()} profile: {profile_description}.",
        )
    cancelled = False

    if detail == "ultra_fast":
        extracted, rejected = _ultra_fast_events(chunks, progress_callback, cancellation_callback)
        extracted = _deduplicate_ultra_events(extracted)
        extracted = _adjudicate_ultra_events(extracted, llm, progress_callback)
        audit.batches_processed = len(batches)
        audit.pages_completed = 0
        audit.chunks_completed = len(chunks)
        audit.rejected_without_source = rejected
        audit.events_extracted = len(extracted)
        events = deduplicate_events(extracted)
        audit.events_rendered = len(events)
        if progress_callback:
            progress_callback(0.94, f"Rendering {len(events)} deterministic dated excerpts with provenance...")
        return render_chronology(events, audit)

    if detail == "fast":
        # vLLM continuously batches concurrent requests. Serial extraction made
        # a large case take hours even though every source batch is independent.
        # Keep result assembly ordered so concurrency cannot change the output.
        try:
            configured_workers = int(
                os.environ.get("KIRAG_FAST_TIMELINE_CONCURRENCY", FAST_DEFAULT_CONCURRENCY)
            )
        except ValueError:
            configured_workers = FAST_DEFAULT_CONCURRENCY
        worker_count = max(1, min(configured_workers, len(batches)))
        progress_lock = threading.Lock()
        completed_pages = 0
        completed_events = 0
        completed_chunks = 0

        def extract_fast_batch(index: int, batch: list[dict[str, Any]]):
            nonlocal completed_pages, completed_events
            previous_batch_events = 0

            def report_page(pages: int, batch_events: int) -> None:
                nonlocal completed_pages, completed_events, previous_batch_events
                with progress_lock:
                    # A callback reports cumulative values for its own batch.
                    # Publish monotonic global progress using per-page deltas.
                    completed_pages += 1
                    completed_events += max(0, batch_events - previous_batch_events)
                    previous_batch_events = batch_events
                    if progress_callback:
                        progress_callback(
                            0.08 + 0.72 * completed_chunks / len(chunks),
                            f"Checkpointed {completed_pages} new valid pages; "
                            f"{completed_chunks}/{len(chunks)} chunks completed; "
                            f"at least {completed_events} events completed.",
                        )

            result = _extract_fast_pages(batch, llm, checkpoint_dir, report_page)
            return index, batch, result

        results: dict[int, tuple[list[dict[str, Any]], tuple[list[ChronologyEvent], int, int, bool]]] = {}
        if progress_callback:
            progress_callback(
                0.07,
                f"Fast timeline dispatching {len(batches)} source batches with up to "
                f"{worker_count} concurrent model requests...",
            )
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="chronology") as pool:
            futures = {
                pool.submit(extract_fast_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index, batch, result = future.result()
                results[index] = (batch, result)
                events, rejected, pages, valid = result
                with progress_lock:
                    audit.rejected_without_source += rejected
                    audit.pages_completed += pages
                    if valid:
                        audit.batches_processed += 1
                        audit.chunks_completed += len(batch)
                        completed_chunks = audit.chunks_completed
                    else:
                        audit.batches_failed += 1
                        audit.warnings.append(
                            f"Fast extraction batch {index + 1} stopped after {pages} valid checkpointed page(s)"
                        )
                if cancellation_callback and cancellation_callback():
                    cancelled = True
                    for pending in futures:
                        pending.cancel()
                    break

        if cancelled:
            return "Chronology generation cancelled."
        for index in range(len(batches)):
            extracted.extend(results[index][1][0])
        audit.events_extracted = len(extracted)
        if progress_callback:
            progress_callback(0.84, f"Validating and deduplicating {len(extracted)} extracted events...")
        events = deduplicate_events(extracted)
        audit.events_rendered = len(events)
        if progress_callback:
            progress_callback(0.94, f"Rendering {len(events)} validated chronology events with provenance...")
        return render_chronology(events, audit)

    def process_batch(batch: list[dict[str, Any]], label: str, progress: float, depth: int = 0):
        nonlocal cancelled
        if cancellation_callback and cancellation_callback():
            cancelled = True
            return
        if progress_callback:
            progress_callback(
                progress,
                (
                    f"Extracting next chronology page; {audit.chunks_completed}/{audit.chunks_total} "
                    f"chunks and {len(extracted)} events completed..."
                    if detail == "fast"
                    else f"Extracting chronology events from source batch {label}..."
                ),
            )
        if detail == "fast":
            def report_page(pages: int, batch_events: int) -> None:
                if progress_callback:
                    progress_callback(
                        progress,
                        f"Checkpointed {audit.pages_completed + pages} valid pages; "
                        f"{audit.chunks_completed}/{audit.chunks_total} chunks completed; "
                        f"{len(extracted) + batch_events} events completed.",
                    )

            events, rejected, pages, valid = _extract_fast_pages(
                batch, llm, checkpoint_dir, report_page
            )
            audit.rejected_without_source += rejected
            audit.pages_completed += pages
            extracted.extend(events)
            if valid:
                audit.batches_processed += 1
                audit.chunks_completed += len(batch)
            else:
                audit.batches_failed += 1
                audit.warnings.append(
                    f"Fast extraction stopped after {pages} valid checkpointed page(s)"
                )
            return
        events, rejected, valid = _extract_validated_batch(
            batch,
            llm,
            checkpoint_dir,
            (
                lambda: progress_callback(
                    progress,
                    f"Source batch {label} returned invalid JSON; running one repair attempt...",
                )
                if progress_callback
                else None
            ),
            detail,
        )
        audit.rejected_without_source += rejected
        if valid:
            audit.batches_processed += 1
            audit.pages_completed += 1
            audit.chunks_completed += len(batch)
            extracted.extend(events)
            return
        if len(batch) > 1 and depth < 8:
            midpoint = len(batch) // 2
            subdivisions = (batch[:midpoint], batch[midpoint:])
            audit.batches_total += 1
            if progress_callback:
                progress_callback(
                    progress,
                    f"Source batch {label} exceeded structured output limits; splitting recursively...",
                )
            for sub_index, subdivision in enumerate(subdivisions, 1):
                process_batch(subdivision, f"{label}.{sub_index}", progress, depth + 1)
                if cancelled:
                    return
            return
        audit.batches_failed += 1
        audit.warnings.append(f"Batch {label} remained invalid at minimum subdivision")

    for index, batch in enumerate(batches, 1):
        process_batch(
            batch,
            f"{index} of {len(batches)}",
            0.08 + 0.72 * (index - 1) / len(batches),
        )
        if cancelled:
            return "Chronology generation cancelled."
    audit.events_extracted = len(extracted)
    if progress_callback:
        progress_callback(0.84, f"Validating and deduplicating {len(extracted)} extracted events...")
    events = deduplicate_events(extracted)
    audit.events_rendered = len(events)
    if progress_callback:
        progress_callback(0.94, f"Rendering {len(events)} validated chronology events with provenance...")
    return render_chronology(events, audit)
