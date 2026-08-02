"""Verified production analysis profiles and guarded runtime switching."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "workspace" / "runtime"
PROFILE_STATE = RUNTIME_DIR / "analysis-profile.json"
OPERATIONS_DIR = RUNTIME_DIR / "analysis-operations"
SWITCH_LOCK = RUNTIME_DIR / "analysis-switch.lock"
ANALYSIS_URL = "http://127.0.0.1:8002/v1"
TERMINAL_STATES = {"completed", "rolled_back", "failed"}
logger = logging.getLogger(__name__)

ANALYSIS_PROFILES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3.6-35B-A3B": {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "display_name": "Qwen 3.6 35B A3B",
        "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "context_length": 262144,
        "dtype": "bfloat16",
        "quantization": "none",
        "reasoning_parser": "qwen3",
        "estimated_load_seconds": 300,
    },
    "google/gemma-4-31B-it": {
        "model": "google/gemma-4-31B-it",
        "display_name": "Gemma 4 31B IT",
        "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "context_length": 262144,
        "dtype": "bfloat16",
        "quantization": "none",
        "reasoning_parser": None,
        "estimated_load_seconds": 420,
    },
}

_state_lock = threading.RLock()
_active_thread: threading.Thread | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_runtime_profile() -> dict[str, Any] | None:
    try:
        value = json.loads(PROFILE_STATE.read_text(encoding="utf-8"))
        model = value.get("model")
        if model in ANALYSIS_PROFILES and value.get("revision") == ANALYSIS_PROFILES[model]["revision"]:
            return value
    except (OSError, ValueError, TypeError):
        pass
    return None


def configured_profile() -> dict[str, Any]:
    state = read_runtime_profile()
    if state:
        return dict(ANALYSIS_PROFILES[state["model"]])
    model = os.environ.get("KIRAG_ANALYSIS_MODEL", "Qwen/Qwen3.6-35B-A3B")
    return dict(ANALYSIS_PROFILES.get(model, ANALYSIS_PROFILES["Qwen/Qwen3.6-35B-A3B"]))


def validate_cached_profile(model: str) -> tuple[bool, str, str]:
    profile = ANALYSIS_PROFILES.get(model)
    if not profile:
        return False, "Model is not an approved analysis profile", ""
    hf_home = Path(os.environ.get("KIRAG_HF_HOME", ROOT / "workspace/huggingface")).resolve()
    folder = "models--" + model.replace("/", "--", 1)
    repository = hf_home / "hub" / folder
    revision = profile["revision"]
    snapshot = repository / "snapshots" / revision
    errors: list[str] = []
    ref = repository / "refs" / "main"
    if not ref.is_file() or ref.read_text(encoding="utf-8").strip() != revision:
        errors.append("refs/main does not point to the verified revision")
    if not snapshot.is_dir():
        errors.append("verified snapshot is missing")
    else:
        broken = [path for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
        if broken:
            errors.append(f"{len(broken)} broken snapshot link(s)")
        indexes = list(snapshot.glob("*.safetensors.index.json"))
        for index in indexes:
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
                missing = {name for name in weight_map.values() if not (snapshot / name).is_file()}
                if missing:
                    errors.append(f"{len(missing)} indexed weight shard(s) missing")
            except (OSError, ValueError, AttributeError):
                errors.append(f"invalid weight index {index.name}")
        if not list(snapshot.glob("*.safetensors")):
            errors.append("snapshot has no safetensors weights")
    incomplete = [path for path in (repository / "blobs").glob("*.incomplete") if path.stat().st_size]
    if incomplete:
        errors.append(f"{len(incomplete)} unfinished download(s)")
    return not errors, "; ".join(errors), str(snapshot)


def _live_model() -> str:
    try:
        with urllib.request.urlopen(f"{ANALYSIS_URL}/models", timeout=3) as response:
            return str(json.load(response).get("data", [{}])[0].get("id", ""))
    except (OSError, ValueError, IndexError, TypeError):
        return ""


def analysis_status() -> dict[str, Any]:
    resume_pending_switch()
    runtime = read_runtime_profile()
    configured = configured_profile()
    live = _live_model()
    operations = list_operations(limit=1)
    profile_rows = []
    for model, profile in ANALYSIS_PROFILES.items():
        complete, error, snapshot = validate_cached_profile(model)
        profile_rows.append({**profile, "cache_complete": complete, "cache_error": error, "snapshot": snapshot})
    return {
        "configured_model": configured["model"],
        "served_model": live,
        "configuration_matches_runtime": live == configured["model"],
        "runtime_state": runtime,
        "profiles": profile_rows,
        "operation": operations[0] if operations and operations[0].get("state") not in TERMINAL_STATES else None,
    }


def _operation_path(operation_id: str) -> Path:
    return OPERATIONS_DIR / f"{operation_id}.json"


def _save_operation(operation: dict[str, Any]) -> None:
    operation["updated_at"] = _now()
    _atomic_json(_operation_path(operation["id"]), operation)


def get_operation(operation_id: str) -> dict[str, Any] | None:
    if not operation_id or any(char not in "0123456789abcdef" for char in operation_id):
        return None
    try:
        return json.loads(_operation_path(operation_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_operations(limit: int = 20) -> list[dict[str, Any]]:
    if not OPERATIONS_DIR.is_dir():
        return []
    rows = []
    for path in sorted(OPERATIONS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if len(rows) >= limit:
            break
    return rows


def switch_in_progress() -> bool:
    resume_pending_switch()
    operations = list_operations(limit=1)
    return bool(operations and operations[0].get("state") not in TERMINAL_STATES)


def resume_pending_switch() -> bool:
    """Resume a durable non-terminal operation after an API process restart."""
    global _active_thread
    operations = list_operations(limit=1)
    if not operations or operations[0].get("state") in TERMINAL_STATES:
        return False
    with _state_lock:
        if _active_thread and _active_thread.is_alive():
            return False
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        lock_handle = open(SWITCH_LOCK, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_handle.close()
            return False
        operation = operations[0]
        operation.setdefault("events", []).append({
            "at": _now(), "state": operation.get("state", "queued"),
            "message": "Resuming guarded switch after API restart",
            "progress": operation.get("progress", 0),
        })
        _save_operation(operation)
        _active_thread = threading.Thread(
            target=_run_switch, args=(operation, lock_handle),
            name=f"analysis-switch-{operation['id'][:8]}", daemon=True,
        )
        _active_thread.start()
        return True


def _update(operation: dict[str, Any], state: str, message: str, progress: int) -> None:
    operation.update(state=state, message=message, progress=progress)
    operation.setdefault("events", []).append(
        {"at": _now(), "state": state, "message": message, "progress": progress}
    )
    _save_operation(operation)


def _compose_recreate(model: str) -> None:
    from settings_manager import load_settings

    profile = ANALYSIS_PROFILES[model]
    settings = load_settings()
    extended = settings.get("startup_mode", "analysis_262k") == "analysis_262k"
    env = os.environ.copy()
    env.update(
        KIRAG_ANALYSIS_MODEL=model,
        KIRAG_ANALYSIS_MODEL_REVISION=profile["revision"],
        KIRAG_ANALYSIS_MAX_MODEL_LEN=str(profile["context_length"] if extended else 32768),
        KIRAG_ANALYSIS_GPU_MEMORY_UTILIZATION="0.85" if extended else "0.57",
    )
    command = [
        "docker", "compose", "--project-directory", str(ROOT),
        "-f", str(ROOT / "docker-compose.rag.yml"),
        "-f", str(ROOT / "docker-compose.production.yml"),
        "up", "-d", "--no-deps", "--force-recreate", "vllm-analysis",
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def _wait_for_model(operation: dict[str, Any], model: str, timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = _live_model()
        if live == model:
            return
        elapsed = int(timeout - max(0, deadline - time.monotonic()))
        estimate = max(1, int(ANALYSIS_PROFILES[model]["estimated_load_seconds"]))
        progress = min(82, 25 + int(55 * elapsed / estimate))
        operation.update(message=f"Loading {ANALYSIS_PROFILES[model]['display_name']}…", progress=progress)
        _save_operation(operation)
        time.sleep(5)
    raise TimeoutError(f"analysis endpoint did not serve {model} within {timeout} seconds")


def _smoke(model: str) -> None:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly READY"}],
        "temperature": 0,
        "max_tokens": 64,
    }).encode()
    request = urllib.request.Request(
        f"{ANALYSIS_URL}/chat/completions", payload,
        {"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if not result.get("choices") or not result["choices"][0].get("message"):
        raise RuntimeError("analysis smoke test returned no assistant message")


def _activate(model: str) -> None:
    profile = ANALYSIS_PROFILES[model]
    state = {"model": model, "revision": profile["revision"], "updated_at": _now(), "last_verified_at": _now()}
    _atomic_json(PROFILE_STATE, state)
    from settings_manager import load_settings, save_settings
    settings = load_settings()
    settings["analysis_model_name"] = model
    settings["analysis_server_url"] = ANALYSIS_URL
    save_settings(settings)
    try:
        from rag.analyzer import invalidate_model_cache
        invalidate_model_cache()
    except Exception:
        pass


def _run_switch(operation: dict[str, Any], lock_handle) -> None:
    global _active_thread
    target = operation["target_model"]
    previous = operation["previous_model"]
    try:
        _update(operation, "validating_cache", "Validating immutable offline snapshot…", 8)
        complete, error, _ = validate_cached_profile(target)
        if not complete:
            raise RuntimeError(error)
        _update(operation, "creating_container", "Recreating the analysis role…", 20)
        _compose_recreate(target)
        _update(operation, "loading_weights", f"Loading {ANALYSIS_PROFILES[target]['display_name']}…", 25)
        _wait_for_model(operation, target)
        _update(operation, "running_smoke_test", "Running live chat-completion smoke test…", 88)
        _smoke(target)
        _update(operation, "activating", "Persisting verified analysis profile…", 96)
        _activate(target)
        operation["completed_at"] = _now()
        _update(operation, "completed", f"Analysis model switched to {target}", 100)
    except Exception as exc:
        operation["error"] = str(exc)
        logger.exception("Analysis switch %s failed", operation["id"])
        if previous in ANALYSIS_PROFILES and previous != target:
            try:
                _update(operation, "rolling_back", f"Switch failed; restoring {previous}…", 90)
                _compose_recreate(previous)
                _wait_for_model(operation, previous)
                _smoke(previous)
                _activate(previous)
                operation["completed_at"] = _now()
                _update(operation, "rolled_back", f"Switch failed and {previous} was restored: {exc}", 100)
            except Exception as rollback_exc:
                operation["rollback_error"] = str(rollback_exc)
                _update(operation, "failed", f"Switch and rollback failed: {exc}; {rollback_exc}", 100)
        else:
            _update(operation, "failed", f"Analysis switch failed: {exc}", 100)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
        with _state_lock:
            _active_thread = None


def start_switch(target_model: str) -> dict[str, Any]:
    global _active_thread
    if target_model not in ANALYSIS_PROFILES:
        raise ValueError("Model is not an approved analysis profile")
    if _live_model() == target_model:
        raise ValueError(f"{target_model} is already serving")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = open(SWITCH_LOCK, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError("Another analysis model switch is already running") from exc
    with _state_lock:
        if _active_thread and _active_thread.is_alive():
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
            raise RuntimeError("Another analysis model switch is already running")
        previous = _live_model() or configured_profile()["model"]
        operation = {
            "id": uuid.uuid4().hex,
            "state": "queued",
            "message": "Analysis switch queued",
            "progress": 0,
            "target_model": target_model,
            "previous_model": previous,
            "created_at": _now(),
            "updated_at": _now(),
            "events": [],
            "error": "",
            "rollback_error": "",
        }
        _save_operation(operation)
        _active_thread = threading.Thread(
            target=_run_switch, args=(operation, lock_handle),
            name=f"analysis-switch-{operation['id'][:8]}", daemon=True,
        )
        _active_thread.start()
        return operation
