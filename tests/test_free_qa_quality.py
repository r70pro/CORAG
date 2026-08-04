from rag.free_qa_quality import (
    build_evidence_ledger,
    build_quality_instructions,
    choose_generation_tokens,
    classify_free_qa,
    inspect_response,
)


def _result(text: str) -> dict:
    return {"text": text}


def test_classifies_comprehensive_chronology_and_allocates_completion_room():
    results = [_result(f"Consultation dated 2024-01-{day:02d}") for day in range(1, 13)]

    plan = classify_free_qa("Provide a comprehensive chronology of all records", results)

    assert plan.task == "chronology"
    assert plan.broad_scope is True
    assert plan.dated_evidence_count == 12
    assert plan.requested_output_tokens == 4096
    assert plan.compact is True


def test_simple_question_keeps_fast_compact_budget():
    plan = classify_free_qa("Who authored the report?", [_result("Report by Dr Smith")])

    assert plan.task == "factual_qa"
    assert plan.requested_output_tokens == 1536
    assert plan.compact is False


def test_generation_budget_respects_context_and_explicit_user_limit():
    plan = classify_free_qa("Provide a complete history", [])

    assert choose_generation_tokens(plan, 3000, None) == 3000
    assert choose_generation_tokens(plan, 8000, 2200) == 2200


def test_chronology_guidance_distinguishes_event_and_document_dates():
    plan = classify_free_qa("Provide a chronology", [_result("2024-01-01")])
    guidance = build_quality_instructions(plan)

    assert "underlying event date" in guidance
    assert "indexed-but-unseen" in guidance
    assert "substantive injury" in guidance
    assert "oldest first" in guidance


def test_evidence_ledger_marks_indexes_and_future_reference_conflicts():
    ledger = build_evidence_ledger([{
        "text": "Document Review Index: treating report dated 2025-02-10",
        "date_extracted": "2024-12-02",
        "document_type": "document_index",
        "author": "Reviewer",
    }])

    assert ledger[0].evidence_status == "indirect index/reference"
    assert ledger[0].date_conflict is True


def test_response_inspection_detects_truncation_and_invalid_sources():
    from rag.analyzer import OUTPUT_LIMIT_WARNING

    findings = inspect_response(
        "A comprehensive answer [Source 3]." + OUTPUT_LIMIT_WARNING,
        source_count=2,
    )

    assert findings.truncated is True
    assert findings.invalid_source_ids == (3,)
    assert findings.claims_comprehensive_while_truncated is True


def test_response_inspection_accepts_complete_valid_answer():
    findings = inspect_response("The scan was normal [Source 1].", source_count=1)

    assert findings.truncated is False
    assert findings.invalid_source_ids == ()
    assert findings.claims_comprehensive_while_truncated is False


def test_rejects_failure_pattern_of_false_index_claim_and_speculative_date_errors():
    results = [{
        "text": "History of injury, symptoms, diagnosis and treatment on 2022-04-07.",
        "date_extracted": "1952-02-01",
        "document_type": "physiotherapy_report",
    }, {
        "text": "Clinical examination and return to work assessment dated 1997-07-25.",
        "document_type": "clinical_record",
    }]
    response = (
        "The available documents are primarily index pages or administrative correspondence. "
        "Several document dates appear to be metadata errors or placeholders. "
        "The report occurred on 2024-02-01."
    )

    findings = inspect_response(response, 2, results)

    assert findings.unsupported_corpus_characterization is True
    assert findings.speculative_date_error_claim is True
    assert findings.ungrounded_dates == ("2024-02-01",)


def test_evidence_ledger_identifies_potentially_substantive_clinical_text():
    ledger = build_evidence_ledger([{
        "text": "History of injury followed by examination, diagnosis and treatment.",
        "document_type": "physiotherapy_report",
    }])

    assert ledger[0].substantive_clinical_content is True
