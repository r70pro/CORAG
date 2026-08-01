"""Central policy for RAG retrieval and Qwen reasoning behaviour."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisPolicy:
    mode: str
    enable_thinking: bool
    comprehensive_retrieval: bool
    min_top_k: int
    score_threshold: float


_EXTRACTION_MODES = {
    "timeline",
    "injury_summary",
    "inconsistency_finder",
    "medication_tracker",
}
_ANALYTICAL_MODES = {
    "free_qa",
    "causation",
    "prognosis",
    "work_capacity",
    "treatment_planning",
}


def get_analysis_policy(mode: str) -> AnalysisPolicy:
    """Return the authoritative server-side policy for an analysis mode."""
    normalized = mode if mode in _EXTRACTION_MODES | _ANALYTICAL_MODES else "free_qa"
    if normalized in _EXTRACTION_MODES:
        return AnalysisPolicy(normalized, False, False, 50, 0.05)
    return AnalysisPolicy(normalized, True, True, 50, 0.05)


def is_analytical_mode(mode: str) -> bool:
    return get_analysis_policy(mode).comprehensive_retrieval
