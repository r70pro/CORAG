#!/usr/bin/env python3
"""Start core services, then restore the user's selected inference profile."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from settings_manager import load_settings  # noqa: E402
from vllm_lifecycle import stop_vllm, switch_vllm  # noqa: E402

COMPOSE = [
    "docker",
    "compose",
    "--project-directory",
    str(ROOT),
    "-f",
    str(ROOT / "docker-compose.rag.yml"),
    "-f",
    str(ROOT / "docker-compose.production.yml"),
]


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run([*COMPOSE, *args], check=True, cwd=ROOT, env=env)


def main() -> None:
    settings = load_settings()
    mode = os.environ.get("KIRAG_STARTUP_MODE", settings.get("startup_mode", "analysis"))

    # Make databases available independently of multi-minute model loading.
    run("up", "--detach", "--wait", "--wait-timeout", "300", "postgres", "redis", "minio", "qdrant")
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "cli.py"), "rag", "infra", "init"],
        check=True,
        cwd=ROOT,
    )

    # Migrate legacy persisted modes without ever restoring simultaneous models.
    mode = {"analysis_262k": "analysis", "ocr_only": "ocr", "dual_32k": "stopped"}.get(mode, mode)
    if mode == "ocr":
        switch_vllm("ocr")
    elif mode == "analysis":
        switch_vllm("analysis", settings.get("analysis_model_name", "Qwen/Qwen3.6-35B-A3B"))
    else:
        stop_vllm()


if __name__ == "__main__":
    main()
