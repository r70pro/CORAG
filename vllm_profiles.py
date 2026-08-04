"""Immutable runtime profiles for KIRAG's single vLLM inference slot."""

from __future__ import annotations

import os
from typing import Any

from analysis_profiles import ANALYSIS_PROFILES

OCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
OCR_PORT = 8000
ANALYSIS_PORT = 8002


def profile_for(role: str, analysis_model: str | None = None) -> dict[str, Any]:
    """Return a validated, complete profile suitable for Compose."""
    if role == "ocr":
        revision = os.environ.get("KIRAG_OCR_MODEL_REVISION", "").strip()
        if not revision:
            raise ValueError("KIRAG_OCR_MODEL_REVISION is required")
        return {
            "key": "ocr",
            "role": "ocr",
            "model": OCR_MODEL,
            "revision": revision,
            "host_port": OCR_PORT,
            "gpu_memory_utilization": os.environ.get("KIRAG_OCR_GPU_MEMORY_UTILIZATION", "0.85"),
            "max_model_len": int(os.environ.get("KIRAG_OCR_MAX_MODEL_LEN", "15360")),
            "max_batched_tokens": int(os.environ.get("KIRAG_OCR_MAX_BATCHED_TOKENS", "4096")),
            "estimated_load_seconds": 300,
        }
    if role != "analysis":
        raise ValueError(f"Unsupported vLLM role: {role}")
    model = analysis_model or "Qwen/Qwen3.6-35B-A3B"
    source = ANALYSIS_PROFILES.get(model)
    if not source:
        raise ValueError("Model is not an approved analysis profile")
    return {
        "key": f"analysis:{model}",
        "role": "analysis",
        "model": model,
        "revision": source["revision"],
        "host_port": ANALYSIS_PORT,
        "gpu_memory_utilization": os.environ.get("KIRAG_ANALYSIS_GPU_MEMORY_UTILIZATION", "0.85"),
        "max_model_len": int(source["context_length"]),
        "max_batched_tokens": int(os.environ.get("KIRAG_ANALYSIS_MAX_BATCHED_TOKENS", "8192")),
        "estimated_load_seconds": int(source["estimated_load_seconds"]),
    }


def compose_environment(profile: dict[str, Any]) -> dict[str, str]:
    return {
        "KIRAG_VLLM_ROLE": str(profile["role"]),
        "KIRAG_VLLM_MODEL": str(profile["model"]),
        "KIRAG_VLLM_MODEL_REVISION": str(profile["revision"]),
        "KIRAG_VLLM_HOST_PORT": str(profile["host_port"]),
        "KIRAG_VLLM_GPU_MEMORY_UTILIZATION": str(profile["gpu_memory_utilization"]),
        "KIRAG_VLLM_MAX_MODEL_LEN": str(profile["max_model_len"]),
        "KIRAG_VLLM_MAX_BATCHED_TOKENS": str(profile["max_batched_tokens"]),
    }
