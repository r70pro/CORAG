from unittest.mock import patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import has_admin_access
from rag.analysis_policy import get_analysis_policy
from rag.analyzer import ContextWindowError, _validate_managed_context_invariant
from rag.retriever import (
    EXPERT_ANALYTICAL_QUERY_FACETS,
    JUDGE_ANALYTICAL_QUERY_FACETS,
    search_comprehensive,
)


def test_free_qa_and_deep_modes_enable_thinking():
    for mode in (
        "free_qa",
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


@patch.dict("os.environ", {"TESTING": "false"})
@patch("docker_manager.get_docker_status", return_value="running")
def test_ocr_active_requires_32k_context(_status):
    with pytest.raises(ContextWindowError, match="expected 32,768"):
        _validate_managed_context_invariant("http://localhost:8002/v1", 262144, 262144)


@patch.dict("os.environ", {"TESTING": "false"})
@patch("docker_manager.get_docker_status", return_value="exited")
def test_ocr_inactive_requires_full_context(_status):
    with pytest.raises(ContextWindowError, match="full 262,144"):
        _validate_managed_context_invariant("http://localhost:8002/v1", 32768, 262144)


@patch.dict("os.environ", {"KIRAG_ADMIN_API_KEY": "admin-secret"}, clear=False)
def test_reasoning_admin_status_comes_from_verified_credential():
    assert has_admin_access(admin_from_header="admin-secret", key_from_header=None, bearer=None)
    assert not has_admin_access(
        admin_from_header=None,
        key_from_header="regular-key",
        bearer=HTTPAuthorizationCredentials(scheme="Bearer", credentials="regular-key"),
    )
