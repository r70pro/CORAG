import json
import threading
import time
from unittest.mock import patch

from rag.analyzer import analyze
from rag.chronology import (
    MISSING,
    ChronologyAudit,
    ChronologyEvent,
    _batch_checkpoint_path,
    build_batches,
    deduplicate_events,
    display_date,
    generate_comprehensive_chronology,
    normalize_event_date,
    parse_batch_response,
    render_chronology,
)


def source(chunk_id="c1", doc_id="d1", text="14 February 2022 consultation with sudden sharp stabbing pain"):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": 0,
        "text": text,
        "original_filename": "record.pdf",
        "page_start": 3,
        "page_end": 3,
        "source_char_start": 10,
        "source_char_end": 80,
    }


def event_mapping(**overrides):
    value = {
        "event_date": "2022-02-14",
        "date_precision": "day",
        "date_original": "14 February 2022",
        "event_type": "GP consultation",
        "provider": "Dr Tong",
        "facility": "The Local Doctor",
        "presenting_symptoms": "Sharp low-back pain and inability to sit",
        "diagnosis": "Acute low-back injury",
        "investigations_findings": MISSING,
        "intervention_plan": "Ibuprofen prescribed",
        "administrative_context": "",
        "source_ids": [1],
        "source_quote": "sudden sharp stabbing pain",
    }
    value.update(overrides)
    return value


def test_normalizes_and_displays_complete_and_partial_australian_dates():
    assert normalize_event_date("2022-02-14", "day") == ("2022-02-14", "day", None)
    assert normalize_event_date("2022-02", "month") == ("2022-02", "month", None)
    assert normalize_event_date("2022-02-31", "day")[1] == "unknown"

    complete = ChronologyEvent(event_date="2022-02-14", date_precision="day")
    partial = ChronologyEvent(event_date="2022-02", date_precision="month")
    assert display_date(complete) == "14/02/2022"
    assert display_date(partial) == "02/2022 [Date Incomplete]"


def test_parse_rejects_unprovenanced_events_and_separates_required_fields():
    batch = [source()]
    payload = {"events": [event_mapping(), event_mapping(source_ids=[99])]}
    events, rejected = parse_batch_response(json.dumps(payload), batch)

    assert rejected == 1
    assert len(events) == 1
    assert events[0].presenting_symptoms.startswith("Sharp")
    assert events[0].diagnosis == "Acute low-back injury"
    assert events[0].intervention_plan == "Ibuprofen prescribed"
    assert events[0].sources[0]["chunk_id"] == "c1"


def test_parse_rejects_invented_source_quote():
    events, rejected = parse_batch_response(
        json.dumps({"events": [event_mapping(source_quote="words absent from source")]}),
        [source()],
    )
    assert events == []
    assert rejected == 1


def test_parse_rejects_placeholder_leak_and_non_contiguous_multi_source_quote():
    batch = [source("c1"), source("c2", text="a second documented passage")]
    payload = {"events": [
        event_mapping(provider="missing-value phrase"),
        event_mapping(source_ids=[1, 2], source_quote="sudden sharp...second documented"),
    ]}

    events, rejected = parse_batch_response(json.dumps(payload), batch)

    assert events == []
    assert rejected == 2


def test_potential_inference_is_removed_unless_supported_by_quote():
    event = ChronologyEvent.from_mapping(
        event_mapping(diagnosis="Likely disc injury", source_quote="sudden sharp stabbing pain"),
        1,
    )
    assert event is not None
    assert event.diagnosis == MISSING
    assert any("unsupported inference" in warning for warning in event.warnings)


def test_deduplicates_same_event_but_preserves_distinct_same_day_encounters():
    first = ChronologyEvent.from_mapping(event_mapping(), 1)
    duplicate = ChronologyEvent.from_mapping(event_mapping(), 1)
    distinct = ChronologyEvent.from_mapping(
        event_mapping(event_type="Imaging", diagnosis="No fracture", intervention_plan=MISSING), 1
    )
    assert first and duplicate and distinct
    first.sources = [source("c1")]
    duplicate.sources = [source("c2")]
    distinct.sources = [source("c3")]

    events = deduplicate_events([distinct, duplicate, first])

    assert len(events) == 2
    consultation = next(event for event in events if event.event_type == "GP consultation")
    assert {item["chunk_id"] for item in consultation.sources} == {"c1", "c2"}


def test_render_has_required_columns_sorted_dates_and_completeness_statement():
    later = ChronologyEvent.from_mapping(event_mapping(event_date="2023-01-01"), 1)
    earlier = ChronologyEvent.from_mapping(event_mapping(), 1)
    assert later and earlier
    later.sources = earlier.sources = [source()]
    events = deduplicate_events([later, earlier])
    audit = ChronologyAudit(1, 1, 1, 1, batches_processed=1, events_extracted=2, events_rendered=2)

    rendered = render_chronology(events, audit)

    assert "Presenting Symptoms | Diagnosis | Investigations/Findings | Intervention/Plan" in rendered
    assert rendered.index("14/02/2022") < rendered.index("01/01/2023")
    assert "Documents audited: **1/1**" in rendered
    assert "record.pdf, p. 3" in rendered


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}, {"doc_id": "d2"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_complete_pipeline_accounts_for_every_chunk_and_reports_failed_batches(mock_chunks, _documents):
    mock_chunks.return_value = [
        source("c1", text="sudden sharp stabbing pain " + "A" * 30_000),
        source("c2", text="B" * 30_000),
    ]
    responses = iter([
        json.dumps({"events": [event_mapping()]}),
        "not json",
        "still not json",
    ])
    progress = []

    result = generate_comprehensive_chronology(
        "case-1",
        lambda _messages: next(responses),
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    # Force one source per batch for this test without coupling production's
    # batch size to the fixture.
    assert "Source chunks audited: **2/2**" in result
    assert "14/02/2022" in result
    assert "Failed extraction batches: **1**" in result
    assert progress[0][1].startswith("Enumerated 2 documents")


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_valid_batches_resume_from_private_checkpoint(mock_chunks, _documents, tmp_path):
    mock_chunks.return_value = [source()]
    calls = []

    def llm(_messages):
        calls.append(True)
        payload = {"events": [event_mapping()]}
        if len(calls) == 2:
            payload["complete"] = True
        return json.dumps(payload)

    first = generate_comprehensive_chronology("case-1", llm, checkpoint_dir=tmp_path)
    second = generate_comprehensive_chronology("case-1", llm, checkpoint_dir=tmp_path)

    assert "14/02/2022" in first and "14/02/2022" in second
    assert len(calls) == 1
    checkpoint = next(tmp_path.glob("batch-*.json"))
    assert checkpoint.stat().st_mode & 0o777 == 0o600


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_truncated_dense_batch_is_adaptively_split(mock_chunks, _documents):
    mock_chunks.return_value = [source("c1"), source("c2", text="sudden sharp stabbing pain second record")]
    responses = iter([
        "truncated",
        "still truncated",
        json.dumps({"events": [event_mapping()]}),
        json.dumps({"events": [event_mapping()]}),
    ])
    progress = []

    result = generate_comprehensive_chronology(
        "case-1",
        lambda _messages: next(responses),
        progress_callback=lambda value, message: progress.append(message),
    )

    assert "Failed extraction batches: **0**" in result
    assert "Extraction batches processed: **2/2**" in result
    assert any("splitting recursively" in message for message in progress)


def test_fast_profile_accepts_compact_schema_and_renders_compact_table():
    payload = {"events": [{
        "d": "2022-02-14", "p": "day", "od": "14 February 2022",
        "t": "GP consultation", "pr": "Dr Tong", "s": "Sharp low-back pain",
        "dx": "Acute low-back injury", "ip": "Ibuprofen prescribed",
        "src": [1], "q": "sudden sharp stabbing pain",
    }]}
    events, rejected = parse_batch_response(json.dumps(payload), [source()])
    audit = ChronologyAudit(1, 1, 1, 1, profile="fast", batches_processed=1)

    rendered = render_chronology(events, audit)

    assert rejected == 0
    assert "Fast Profile" in rendered
    assert "| Date | Provider/Facility | Event Summary | Intervention/Plan | Source |" in rendered
    assert "Presenting Symptoms | Diagnosis" not in rendered


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_ultra_fast_scans_all_chunks_and_semantically_filters_candidates(mock_chunks, _documents):
    mock_chunks.return_value = [
        source("c1", text="On 14 February 2022 the worker reported sudden sharp stabbing pain."),
        source("c2", text="MRI performed 2022-03-04 showed no fracture."),
        source("c3", text="A work capacity certificate was issued in July 2023."),
    ]

    calls = []
    def llm(_messages, response_format=None):
        calls.append(response_format)
        return json.dumps({"keep": [1, 2, 3]})

    result = generate_comprehensive_chronology(
        "case-1",
        llm,
        detail="ultra_fast",
    )

    assert "Ultra-Fast Profile" in result
    assert "14/02/2022" in result
    assert "04/03/2022" in result
    assert "07/2023 [Date Incomplete]" in result
    assert "Source chunks completed: **3/3**" in result
    assert "bounded semantic filtering" in result.lower()
    assert len(calls) == 1


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_recursive_splitting_continues_beyond_one_level(mock_chunks, _documents):
    mock_chunks.return_value = [
        source(f"c{index}", text=f"sudden sharp stabbing pain source {index}")
        for index in range(4)
    ]
    valid = json.dumps({"events": [event_mapping()]})
    responses = iter([
        "⚠️ Incomplete response",
        "⚠️ Incomplete response",
        valid,
        valid,
        valid,
    ])

    result = generate_comprehensive_chronology("case-1", lambda _messages: next(responses))

    assert "Extraction batches processed: **3/3**" in result
    assert "Failed extraction batches: **0**" in result


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_fast_and_thorough_checkpoints_are_isolated(mock_chunks, _documents, tmp_path):
    mock_chunks.return_value = [source()]
    calls = []

    def llm(_messages):
        calls.append(True)
        payload = {"events": [event_mapping()]}
        if len(calls) == 2:
            payload["complete"] = True
        return json.dumps(payload)

    generate_comprehensive_chronology("case-1", llm, checkpoint_dir=tmp_path, detail="thorough")
    generate_comprehensive_chronology("case-1", llm, checkpoint_dir=tmp_path, detail="fast")

    assert len(calls) == 2
    assert len(list(tmp_path.glob("batch-*.json"))) == 2


def test_fast_checkpoint_v2_does_not_reuse_v1_path(tmp_path):
    batch = [source()]
    with patch("rag.chronology.FAST_CHECKPOINT_VERSION", 1):
        old_path = _batch_checkpoint_path(tmp_path, batch, "fast")
    new_path = _batch_checkpoint_path(tmp_path, batch, "fast")

    assert old_path != new_path


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_rejected_fast_page_is_not_repaired_or_checkpointed(mock_chunks, _documents, tmp_path):
    mock_chunks.return_value = [source()]
    invalid = event_mapping(provider="missing-value phrase")
    calls = []

    def llm(_messages):
        calls.append(True)
        return json.dumps({"events": [invalid], "complete": True})

    result = generate_comprehensive_chronology(
        "case-1", llm, checkpoint_dir=tmp_path, detail="fast"
    )

    assert len(calls) == 1
    assert "Failed extraction batches: **1**" in result
    assert "missing-value phrase" not in result
    assert len(list(tmp_path.glob("batch-*.json"))) == 0


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_fast_mode_checkpoints_each_bounded_continuation_page(mock_chunks, _documents, tmp_path):
    mock_chunks.return_value = [source()]
    responses = iter([
        {"events": [event_mapping()], "complete": False},
        {"events": [event_mapping(event_type="Follow-up")], "complete": True},
    ])
    formats = []

    def llm(_messages, response_format=None):
        formats.append(response_format)
        return json.dumps(next(responses))

    result = generate_comprehensive_chronology(
        "case-1", llm, checkpoint_dir=tmp_path, detail="fast"
    )

    assert len(formats) == 2
    assert all(item["json_schema"]["schema"]["properties"]["events"]["maxItems"] == 12 for item in formats)
    assert len(list(tmp_path.glob("batch-*-page-*.json"))) == 2
    assert "Valid extraction pages checkpointed: **2**" in result
    assert "Source chunks completed: **1/1**" in result


def test_build_batches_never_drops_oversized_source():
    chunks = [source("c1", text="A" * 100), source("c2", text="B" * 100)]
    batches = build_batches(chunks, max_characters=50)
    assert [item[0]["chunk_id"] for item in batches] == ["c1", "c2"]


def test_build_batches_honours_small_chunk_cap():
    chunks = [source(f"c{index}") for index in range(7)]
    batches = build_batches(chunks, max_characters=1_000_000, max_chunks=3)

    assert [len(batch) for batch in batches] == [3, 3, 1]


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_fast_batches_run_concurrently_and_render_deterministically(mock_chunks, _documents):
    mock_chunks.return_value = [
        source(f"c{index}", doc_id=f"d{index}", text=f"documented pain source {index}")
        for index in range(6)
    ]
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def llm(messages, response_format=None):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        text = messages[-1]["content"]
        quote = next(line for line in text.splitlines() if line.startswith("documented pain"))
        return json.dumps({
            "events": [event_mapping(source_quote=quote, source_ids=[1])],
            "complete": True,
        })

    with patch("rag.chronology.FAST_MAX_CHUNKS_PER_BATCH", 1):
        result = generate_comprehensive_chronology("case-1", llm, detail="fast")

    assert peak_active > 1
    assert "Source chunks completed: **6/6**" in result


@patch("rag.chronology.rag_db.get_documents_for_run", return_value=[{"doc_id": "d1"}])
@patch("rag.chronology.rag_db.get_chunks_for_run")
def test_fast_output_limit_is_not_retried(mock_chunks, _documents, tmp_path):
    mock_chunks.return_value = [source()]
    calls = []

    def llm(_messages, response_format=None):
        calls.append(response_format)
        return '{"events": []\n\n⚠️ **Incomplete response**'

    result = generate_comprehensive_chronology(
        "case-1", llm, checkpoint_dir=tmp_path, detail="fast"
    )

    assert len(calls) == 1
    assert "Failed extraction batches: **1**" in result
    assert not list(tmp_path.glob("batch-*.json"))


@patch("rag.chronology.generate_comprehensive_chronology", return_value="complete chronology")
def test_selected_case_timeline_uses_complete_case_pipeline(mock_generate):
    output = list(analyze("Audit every event", mode="timeline", run_id_filter="case-1"))

    assert output == ["complete chronology"]
    assert mock_generate.call_args.args[0] == "case-1"
