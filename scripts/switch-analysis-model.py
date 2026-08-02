#!/usr/bin/env python3
"""Atomically switch the offline production analysis model with rollback."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
MODEL_REVISIONS = {
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
}
MODEL_CONTEXTS = {model: 262144 for model in MODEL_REVISIONS}
COMPOSE = [
    "docker", "compose", "--project-directory", str(ROOT),
    "-f", str(ROOT / "docker-compose.rag.yml"),
    "-f", str(ROOT / "docker-compose.production.yml"),
]


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def replace_env_values(updates: dict[str, str]) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    remaining = dict(updates)
    output: list[str] = []
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)
    output.extend(f"{key}={value}" for key, value in remaining.items())
    fd, temporary = tempfile.mkstemp(prefix=".env.", dir=ROOT, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, ENV_FILE.stat().st_mode & 0o777)
        os.replace(temporary, ENV_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recreate(model: str, revision: str) -> None:
    env = os.environ.copy()
    env.update(read_env())
    env.update(
        KIRAG_ANALYSIS_MODEL=model,
        KIRAG_ANALYSIS_MODEL_REVISION=revision,
        KIRAG_ANALYSIS_MAX_MODEL_LEN=str(MODEL_CONTEXTS[model]),
        KIRAG_ANALYSIS_GPU_MEMORY_UTILIZATION="0.85",
    )
    subprocess.run(
        [*COMPOSE, "up", "-d", "--no-deps", "--force-recreate", "vllm-analysis"],
        check=True, cwd=ROOT, env=env,
    )


def wait_and_smoke(model: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "endpoint did not respond"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8002/v1/models", timeout=5) as response:
                served = json.load(response)["data"][0]["id"]
            if served != model:
                raise RuntimeError(f"endpoint serves {served}, expected {model}")
            break
        except Exception as exc:  # model loading is expected to take minutes
            last_error = str(exc)
            time.sleep(5)
    else:
        raise TimeoutError(last_error)

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly READY"}],
            "temperature": 0,
            "max_tokens": 64,
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8002/v1/chat/completions", payload,
        {"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    choice = result.get("choices", [{}])[0]
    if not choice.get("message") or choice.get("finish_reason") not in {"stop", "length"}:
        raise RuntimeError(f"invalid inference response: {result}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_REVISIONS))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    target_revision = MODEL_REVISIONS[args.model]
    current = read_env()
    previous_model = current.get("KIRAG_ANALYSIS_MODEL", "Qwen/Qwen3.6-35B-A3B")
    previous_revision = current.get(
        "KIRAG_ANALYSIS_MODEL_REVISION", MODEL_REVISIONS.get(previous_model, "")
    )
    cache_home = current.get("KIRAG_HF_HOME", str(ROOT / "workspace/huggingface"))

    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/prepare-production-model.py"),
         args.model, "--revision", target_revision, "--cache-dir", cache_home, "--offline-check"],
        check=True, cwd=ROOT,
    )
    updates = {
        "KIRAG_ANALYSIS_MODEL": args.model,
        "KIRAG_ANALYSIS_MODEL_REVISION": target_revision,
    }
    replace_env_values(updates)
    try:
        recreate(args.model, target_revision)
        wait_and_smoke(args.model, args.timeout)
    except Exception:
        replace_env_values(
            {"KIRAG_ANALYSIS_MODEL": previous_model,
             "KIRAG_ANALYSIS_MODEL_REVISION": previous_revision}
        )
        recreate(previous_model, previous_revision)
        wait_and_smoke(previous_model, args.timeout)
        raise

    # Reload environment overrides used by the API and its bound frontend.
    subprocess.run(["systemctl", "--user", "restart", "kirag-api.service"], check=False)
    subprocess.run(["systemctl", "--user", "restart", "kirag-frontend.service"], check=False)
    print(f"Analysis model switched to {args.model}@{target_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
