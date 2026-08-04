from unittest.mock import patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import has_admin_access
from rag.analysis_policy import get_analysis_policy
from rag.analyzer import ContextWindowError, _validate_managed_context_invariant, analyze
from rag.retriever import (
    EXPERT_ANALYTICAL_QUERY_FACETS,
    JUDGE_ANALYTICAL_QUERY_FACETS,
    _apply_cross_encoder_rerank,
    search_comprehensive,
)


def test_free_qa_is_bounded_interactive_analysis():
    policy = get_analysis_policy("free_qa")
    assert policy.enable_thinking is False
    assert policy.comprehensive_retrieval is True
    assert policy.min_top_k == 16
    assert policy.score_threshold == 0.15


@patch("rag.analyzer.search_similar")
def test_unscoped_timeline_refuses_relevance_limited_generation(mock_search):
    output = list(analyze("Build timeline", mode="timeline", use_reranker=True))

    assert "requires exactly one selected case" in output[0]
    mock_search.assert_not_called()


def test_deep_modes_enable_thinking():
    for mode in (
        "expert_analysis",
        "judge_analysis",
        "causation",
        "prognosis",
        "work_capacity",
        "treatment_planning",
    ):
        policy = get_analysis_policy(mode)
        assert policy.enable_thinking is True
        assert policy.comprehensive_retrieval is True


def test_radiology_report_is_kept_as_complete_evidence_unit():
    from rag.chunker import chunk_document

    report = """1 April 2022\nDear Dr Tong\nMR OF THORACOLUMBAR AND SACRAL SPINE\nFindings:\n""" + ("Minor finding. " * 100) + "\nConclusion:\n1. Minor lumbar degenerative disease.\nDr Nicholas Gelber"
    chunks = chunk_document(report, "doc", "run", max_chunk_size=800)

    assert len(chunks) == 1
    assert chunks[0]["document_type"] == "radiology_report"
    assert "MR OF THORACOLUMBAR" in chunks[0]["text"]
    assert "Conclusion:" in chunks[0]["text"]


def test_expert_and_judge_modes_require_high_assurance_verification():
    assert get_analysis_policy("expert_analysis").high_assurance is True
    assert get_analysis_policy("judge_analysis").high_assurance is True
    assert get_analysis_policy("causation").high_assurance is False


def test_extraction_modes_disable_thinking():
    for mode in ("timeline", "injury_summary", "inconsistency_finder", "medication_tracker"):
        policy = get_analysis_policy(mode)
        assert policy.enable_thinking is False
        assert policy.comprehensive_retrieval is False


def test_general_knowledge_disables_retrieval_and_enables_thinking():
    policy = get_analysis_policy("general_knowledge")
    assert policy.enable_thinking is True
    assert policy.uses_retrieval is False
    assert policy.comprehensive_retrieval is False
    assert policy.min_top_k == 0
    assert policy.score_threshold == 0.0


def test_comprehensive_retrieval_deduplicates_and_preserves_facets():
    def fake_search(query, **kwargs):
        return [{"chunk_id": "same", "doc_id": "d1", "score": 0.8, "text": query}]

    results = search_comprehensive("causation", top_k=50, search_function=fake_search)
    assert len(results) == 1
    assert len(results[0]["retrieval_facets"]) == 8


def test_radiology_question_injects_named_primary_reports_before_generation():
    def fake_vector(_query, **_kwargs):
        return [{"chunk_id": "summary", "doc_id": "d1", "score": 0.8, "text": "report list"}]

    def fake_keyword(terms, **kwargs):
        assert "MR OF THORACOLUMBAR AND SACRAL SPINE" in terms
        assert kwargs["run_id"] == "case-1"
        return [{
            "chunk_id": "primary",
            "doc_id": "d1",
            "score": 0.5,
            "text": "MR OF THORACOLUMBAR AND SACRAL SPINE\nFindings\nConclusion",
            "document_type": "radiology_report",
        }]

    results = search_comprehensive(
        "What do the radiological reports show?",
        top_k=8,
        analytical_facets=(),
        search_function=fake_vector,
        keyword_search_function=fake_keyword,
        run_id_filter="case-1",
        use_reranker=False,
    )

    assert results[0]["chunk_id"] == "primary"
    assert results[0]["primary_evidence"] is True


def test_expert_retrieval_uses_every_requested_evidence_facet():
    def fake_search(query, **kwargs):
        return [{"chunk_id": query, "doc_id": query, "score": 0.8, "text": query}]

    results = search_comprehensive(
        "expert causation question",
        top_k=50,
        analytical_facets=EXPERT_ANALYTICAL_QUERY_FACETS,
        search_function=fake_search,
    )

    assert len(results) == len(EXPERT_ANALYTICAL_QUERY_FACETS) + 1
    retrieved_facets = {result["retrieval_facets"][0] for result in results}
    assert "primary question" in retrieved_facets
    assert set(EXPERT_ANALYTICAL_QUERY_FACETS) <= retrieved_facets


def test_judge_retrieval_uses_legal_and_evidentiary_facets():
    def fake_search(query, **kwargs):
        return [{"chunk_id": query, "doc_id": query, "score": 0.8, "text": query}]

    results = search_comprehensive(
        "legal question",
        top_k=50,
        analytical_facets=JUDGE_ANALYTICAL_QUERY_FACETS,
        search_function=fake_search,
    )

    retrieved_facets = {result["retrieval_facets"][0] for result in results}
    assert set(JUDGE_ANALYTICAL_QUERY_FACETS) <= retrieved_facets


def test_specialized_facets_survive_sparse_overlapping_results():
    calls = 0

    def fake_search(query, **kwargs):
        nonlocal calls
        calls += 1
        if calls in {1, 2, 5}:
            return [
                {
                    "chunk_id": "overlap",
                    "doc_id": "d1",
                    "score": 0.7 + calls / 100,
                    "text": query,
                }
            ]
        return []

    results = search_comprehensive(
        "expert question",
        top_k=50,
        analytical_facets=EXPERT_ANALYTICAL_QUERY_FACETS,
        search_function=fake_search,
    )

    assert calls == len(EXPERT_ANALYTICAL_QUERY_FACETS) + 1
    assert len(results) == 1
    assert results[0]["retrieval_facets"] == [
        "primary question",
        EXPERT_ANALYTICAL_QUERY_FACETS[0],
        EXPERT_ANALYTICAL_QUERY_FACETS[3],
    ]


@patch("rag.embedding.load_reranker_model")
def test_comprehensive_retrieval_reranks_only_once_after_facets(mock_load_reranker):
    reranker = mock_load_reranker.return_value
    reranker.predict.return_value = [0.1, 0.9]
    search_calls = []

    def fake_search(query, **kwargs):
        search_calls.append(kwargs)
        index = len(search_calls)
        return [{"chunk_id": str(index), "doc_id": "d1", "score": 1 / index, "text": query}]

    results = search_comprehensive(
        "causation",
        top_k=2,
        analytical_facets=("chronology",),
        search_function=fake_search,
        use_reranker=True,
        reranker_model="reranker",
        reranker_device="cpu",
    )

    assert all(call["use_reranker"] is False for call in search_calls)
    reranker.predict.assert_called_once()
    assert [result["chunk_id"] for result in results] == ["2", "1"]


@patch("rag.embedding.load_reranker_model")
def test_comprehensive_reranking_is_batched_and_cancellable(mock_load_reranker):
    reranker = mock_load_reranker.return_value
    reranker.predict.side_effect = [[0.1] * 8, [0.2] * 8]
    cancelled = False

    def is_cancelled():
        return cancelled

    def report_progress(_progress, message):
        nonlocal cancelled
        if message.startswith("Reranked 8 of"):
            cancelled = True

    results = [{"text": f"excerpt {index}", "score": 0.5} for index in range(16)]
    completed = _apply_cross_encoder_rerank(
        results,
        "question",
        "reranker",
        "cpu",
        progress_callback=report_progress,
        cancellation_callback=is_cancelled,
    )

    assert completed is False
    reranker.predict.assert_called_once()


@patch.dict("os.environ", {"TESTING": "false"})
@patch("vllm_lifecycle.status", return_value={"ready": True, "active_role": "ocr"})
def test_ocr_active_rejects_analysis(_status):
    with pytest.raises(ContextWindowError, match="active role is 'ocr'"):
        _validate_managed_context_invariant("http://localhost:8002/v1", 262144, 262144)


@patch.dict("os.environ", {"TESTING": "false"})
@patch("vllm_lifecycle.status", return_value={"ready": True, "active_role": "analysis"})
def test_analysis_requires_full_context(_status):
    with pytest.raises(ContextWindowError, match="expected 262,144"):
        _validate_managed_context_invariant("http://localhost:8002/v1", 32768, 262144)


@patch.dict("os.environ", {"KIRAG_ADMIN_API_KEY": "admin-secret"}, clear=False)
def test_reasoning_admin_status_comes_from_verified_credential():
    assert has_admin_access(admin_from_header="admin-secret", key_from_header=None, bearer=None)
    assert not has_admin_access(
        admin_from_header=None,
        key_from_header="regular-key",
        bearer=HTTPAuthorizationCredentials(scheme="Bearer", credentials="regular-key"),
    )
