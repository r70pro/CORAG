#!/usr/bin/env python3
"""Start core services, then restore the user's selected inference profile."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_profiles import ANALYSIS_PROFILES  # noqa: E402
from settings_manager import MODEL_MAX_CONTENT_LENGTHS, load_settings  # noqa: E402

COMPOSE = [
    "docker", "compose", "--project-directory", str(ROOT),
    "-f", str(ROOT / "docker-compose.rag.yml"),
    "-f", str(ROOT / "docker-compose.production.yml"),
]


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run([*COMPOSE, *args], check=True, cwd=ROOT, env=env)


def main() -> None:
    settings = load_settings()
    mode = os.environ.get("KIRAG_STARTUP_MODE", settings.get("startup_mode", "analysis_262k"))

    # Make databases available independently of multi-minute model loading.
    run("up", "--detach", "--wait", "--wait-timeout", "300", "postgres", "redis", "minio", "qdrant")
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "cli.py"), "rag", "infra", "init"], check=True, cwd=ROOT)

    if mode == "dual_32k":
        run("up", "--detach", "--wait", "--wait-timeout", "3600", "vllm", "vllm-analysis")
        return

    if mode == "ocr_only":
        subprocess.run(["docker", "stop", "kirag_vllm_analysis"], check=False, capture_output=True)
        run("up", "--detach", "--wait", "--wait-timeout", "3600", "vllm")
        return

    # Default: dedicate the GPU to full-context analysis and leave OCR stopped.
    subprocess.run(["docker", "stop", "olmocr"], check=False, capture_output=True)
    analysis_model = settings.get("analysis_model_name", "Qwen/Qwen3.6-35B-A3B")
    profile_env = os.environ.copy()
    if analysis_model not in ANALYSIS_PROFILES:
        raise RuntimeError(f"Unsupported production analysis profile: {analysis_model}")
    profile_env["KIRAG_ANALYSIS_MODEL"] = analysis_model
    profile_env["KIRAG_ANALYSIS_MODEL_REVISION"] = ANALYSIS_PROFILES[analysis_model]["revision"]
    profile_env["KIRAG_ANALYSIS_MAX_MODEL_LEN"] = str(MODEL_MAX_CONTENT_LENGTHS.get(analysis_model, 262144))
    profile_env["KIRAG_ANALYSIS_GPU_MEMORY_UTILIZATION"] = "0.85"
    run("up", "--detach", "--no-deps", "--force-recreate", "vllm-analysis", env=profile_env)


if __name__ == "__main__":
    main()
