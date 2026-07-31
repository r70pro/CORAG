#!/usr/bin/env python3
"""Fail-fast validation for an unattended single-machine deployment."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def read_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    env_file = args.env_file.resolve()
    values = read_environment(env_file)
    failures: list[str] = []

    mode = env_file.stat().st_mode & 0o777
    if mode & 0o077:
        failures.append(f"{env_file} must not be readable or writable by group/other")

    required = {
        "OLMOCR_PG_PASS",
        "OLMOCR_MINIO_ACCESS_KEY",
        "OLMOCR_MINIO_SECRET_KEY",
        "OLMOCR_VLLM_IMAGE",
        "KIRAG_OCR_MODEL",
        "KIRAG_OCR_MODEL_REVISION",
        "KIRAG_OCR_SERVER_URL",
        "KIRAG_ANALYSIS_MODEL",
        "KIRAG_ANALYSIS_MODEL_REVISION",
        "KIRAG_ANALYSIS_SERVER_URL",
        "KIRAG_HF_HOME",
        "KIRAG_API_KEY",
        "KIRAG_ADMIN_API_KEY",
    }
    for key in sorted(required):
        value = values.get(key, "")
        if not value or "change_me" in value or "replace_with" in value:
            failures.append(f"{key} is missing or still uses a placeholder")

    image = values.get("OLMOCR_VLLM_IMAGE", "")
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        failures.append("OLMOCR_VLLM_IMAGE must be pinned by sha256 digest")
    for key in ("KIRAG_OCR_MODEL_REVISION", "KIRAG_ANALYSIS_MODEL_REVISION"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", values.get(key, "")):
            failures.append(f"{key} must be a 40-character commit SHA")
    for key in ("KIRAG_API_KEY", "KIRAG_ADMIN_API_KEY", "OLMOCR_PG_PASS"):
        if len(values.get(key, "")) < 24:
            failures.append(f"{key} must contain at least 24 characters")
    if values.get("KIRAG_API_KEY") == values.get("KIRAG_ADMIN_API_KEY"):
        failures.append("KIRAG_API_KEY and KIRAG_ADMIN_API_KEY must differ")

    try:
        tensor_parallel = int(values.get("KIRAG_VLLM_TENSOR_PARALLEL_SIZE", "1"))
        role_memory: dict[str, float] = {}
        role_ports: dict[str, int] = {}
        for role, default_port, default_memory, default_length in (
            ("OCR", "8000", "0.28", "15360"),
            ("ANALYSIS", "8002", "0.57", "32768"),
        ):
            port = int(values.get(f"KIRAG_{role}_VLLM_PORT", default_port))
            gpu_utilization = float(
                values.get(f"KIRAG_{role}_GPU_MEMORY_UTILIZATION", default_memory)
            )
            max_model_len = int(values.get(f"KIRAG_{role}_MAX_MODEL_LEN", default_length))
            max_batched = int(values.get(f"KIRAG_{role}_MAX_BATCHED_TOKENS", default_length))
            if not 1 <= port <= 65535:
                failures.append(f"KIRAG_{role}_VLLM_PORT must be between 1 and 65535")
            if not 0 < gpu_utilization <= 1:
                failures.append(f"KIRAG_{role}_GPU_MEMORY_UTILIZATION must be in (0, 1]")
            if max_model_len < 1024 or max_batched < 1024:
                failures.append(f"{role} model and batched token limits must be at least 1024")
            role_memory[role] = gpu_utilization
            role_ports[role] = port
        if role_ports.get("OCR") == role_ports.get("ANALYSIS"):
            failures.append("OCR and analysis vLLM ports must differ")
        if role_memory.get("OCR", 1) >= role_memory.get("ANALYSIS", 0):
            failures.append("analysis memory high-water mark must exceed the OCR high-water mark")
        if role_memory.get("ANALYSIS", 1) > 0.92:
            failures.append("analysis memory high-water mark must not exceed 0.92")
        if tensor_parallel < 1:
            failures.append("KIRAG_VLLM_TENSOR_PARALLEL_SIZE must be positive")
    except ValueError:
        failures.append("vLLM port, memory, context, batching, and parallel values must be numeric")

    hf_home = Path(values.get("KIRAG_HF_HOME", "/missing")).expanduser()
    if not hf_home.is_absolute() or not hf_home.is_dir():
        failures.append("KIRAG_HF_HOME must be an existing absolute directory")
    else:
        hub_cache = hf_home / "hub"
        for role in ("OCR", "ANALYSIS"):
            model = values.get(f"KIRAG_{role}_MODEL", "")
            revision = values.get(f"KIRAG_{role}_MODEL_REVISION", "")
            repo_cache = hub_cache / ("models--" + model.replace("/", "--"))
            snapshot = repo_cache / "snapshots" / revision
            weight_files = [
                *snapshot.glob("*.safetensors"),
                *snapshot.glob("*.bin"),
                *snapshot.glob("*.gguf"),
            ]
            incomplete = list(repo_cache.rglob("*.incomplete")) if repo_cache.exists() else []
            if (
                not snapshot.is_dir()
                or not (snapshot / "config.json").is_file()
                or not weight_files
                or incomplete
            ):
                failures.append(
                    f"{role} immutable model snapshot is missing or incomplete under {hub_cache}"
                )
    log_dir = Path(values.get("KIRAG_LOG_DIR", root / "logs")).expanduser().resolve()
    if log_dir != (root / "logs").resolve() or not os.access(log_dir, os.W_OK):
        failures.append("KIRAG_LOG_DIR must be the writable <root>/logs directory")
    if not (root / "frontend" / ".next" / "BUILD_ID").is_file():
        failures.append("frontend production build is missing; run npm ci && npm run build")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    environment = os.environ.copy()
    environment.update(values)
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(root / "docker-compose.rag.yml"),
            "-f",
            str(root / "docker-compose.production.yml"),
            "config",
            "--quiet",
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, check=True)
    print("Production preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
