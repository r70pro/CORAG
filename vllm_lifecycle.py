"""Crash-recoverable, mutually exclusive lifecycle for the vLLM inference slot."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_log import audit_event
from vllm_profiles import ANALYSIS_PORT, OCR_PORT, compose_environment, profile_for

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "workspace" / "runtime" / "vllm"
STATE_FILE = RUNTIME_DIR / "state.json"
OPERATION_FILE = RUNTIME_DIR / "operation.json"
LIFECYCLE_LOCK = RUNTIME_DIR / "lifecycle.lock"
LEASE_LOCK = RUNTIME_DIR / "inference.lock"
CONTAINER_NAME = "kirag_vllm"
LEGACY_CONTAINERS = ("olmocr", "kirag_vllm_analysis")
COMPOSE = [
    "docker", "compose", "--project-directory", str(ROOT),
    "-f", str(ROOT / "docker-compose.rag.yml"),
    "-f", str(ROOT / "docker-compose.production.yml"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def read_state() -> dict[str, Any]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError):
        pass
    return {"schema_version": 2, "desired_role": "stopped", "active_role": "stopped", "state": "stopped"}


def read_operation() -> dict[str, Any] | None:
    try:
        value = json.loads(OPERATION_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _container_status(name: str = CONTAINER_NAME) -> str:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip().lower()
    if "no such" in result.stderr.lower():
        return "not_found"
    return "error"


def _is_managed(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", '{{index .Config.Labels "com.kirag.managed"}}', name],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _quarantine_legacy_containers() -> None:
    """Ensure an upgraded installation cannot resurrect either old GPU owner."""
    for name in LEGACY_CONTAINERS:
        status = _container_status(name)
        if status == "not_found":
            continue
        if not _is_managed(name):
            raise RuntimeError(f"Legacy container name '{name}' is occupied by an unmanaged container")
        subprocess.run(["docker", "update", "--restart=no", name], check=True, capture_output=True, text=True)
        if status in {"running", "restarting", "paused"}:
            subprocess.run(["docker", "stop", "--time", "120", name], check=True, capture_output=True, text=True)


def _validate_profile_cache(profile: dict[str, Any]) -> None:
    hf_home = Path(os.environ.get("KIRAG_HF_HOME", ROOT / "workspace" / "huggingface")).resolve()
    repository = hf_home / "hub" / ("models--" + str(profile["model"]).replace("/", "--", 1))
    snapshot = repository / "snapshots" / str(profile["revision"])
    if not snapshot.is_dir():
        raise RuntimeError(f"Verified offline snapshot is missing: {snapshot}")
    broken = [path for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    if broken:
        raise RuntimeError(f"Verified offline snapshot has {len(broken)} broken link(s)")
    if not [*snapshot.glob("*.safetensors"), *snapshot.glob("*.bin")]:
        raise RuntimeError("Verified offline snapshot contains no model weights")


def _served_model(port: int, timeout: float = 3) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
            return str(json.load(response).get("data", [{}])[0].get("id", ""))
    except (OSError, ValueError, IndexError, TypeError):
        return ""


def _update_operation(operation: dict[str, Any], state: str, message: str, progress: int) -> None:
    operation.update(state=state, message=message, progress=progress, updated_at=_now())
    operation.setdefault("events", []).append(
        {"at": operation["updated_at"], "state": state, "message": message, "progress": progress}
    )
    _atomic_json(OPERATION_FILE, operation)


@contextlib.contextmanager
def inference_lease(role: str) -> Iterator[None]:
    """Hold a shared cross-process lease for the complete inference operation."""
    state = read_state()
    if state.get("state") != "ready" or state.get("active_role") != role:
        raise RuntimeError(f"{role.upper()} inference is unavailable; active role is {state.get('active_role', 'stopped')}")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(LEASE_LOCK, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_SH)
        state = read_state()
        if state.get("state") != "ready" or state.get("active_role") != role:
            raise RuntimeError(f"{role.upper()} inference began switching")
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _wait_for_model(profile: dict[str, Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _container_status() == "restarting":
            raise RuntimeError("vLLM entered a restart loop")
        if _served_model(int(profile["host_port"])) == profile["model"]:
            return
        time.sleep(5)
    raise TimeoutError(f"vLLM did not serve {profile['model']} within {timeout} seconds")


def _smoke(profile: dict[str, Any]) -> None:
    if profile["role"] == "ocr":
        # Model identity is the dependable lightweight readiness test; the OCR
        # pipeline performs the first image request under an inference lease.
        if _served_model(OCR_PORT, timeout=10) != profile["model"]:
            raise RuntimeError("OCR model identity verification failed")
        return
    payload = json.dumps({
        "model": profile["model"],
        "messages": [{"role": "user", "content": "Reply with exactly READY"}],
        "temperature": 0,
        "max_tokens": 32,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ANALYSIS_PORT}/v1/chat/completions", payload,
        {"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if not result.get("choices") or not result["choices"][0].get("message"):
        raise RuntimeError("analysis smoke test returned no assistant message")


def _stop_and_remove() -> None:
    status = _container_status()
    if status == "not_found":
        return
    if status == "error":
        raise RuntimeError("unable to inspect the managed vLLM container")
    if status in {"running", "restarting", "paused"}:
        subprocess.run(["docker", "stop", "--time", "120", CONTAINER_NAME], check=True, capture_output=True, text=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _container_status() not in {"exited", "created", "not_found"}:
        time.sleep(1)
    if _container_status() not in {"exited", "created", "not_found"}:
        raise RuntimeError("old vLLM process did not exit; refusing to start another model")
    if _container_status() != "not_found":
        subprocess.run(["docker", "rm", CONTAINER_NAME], check=True, capture_output=True, text=True)


def _create(profile: dict[str, Any]) -> None:
    env = os.environ.copy()
    env.update(compose_environment(profile))
    subprocess.run(
        [*COMPOSE, "up", "-d", "--no-deps", "--force-recreate", "vllm"],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=300,
    )


def _activate(profile: dict[str, Any], desired_role: str) -> None:
    previous = read_state()
    state = {
        "schema_version": 2,
        "desired_role": desired_role,
        "active_role": profile["role"],
        "active_model": profile["model"],
        "model_revision": profile["revision"],
        "host_port": profile["host_port"],
        "state": "ready",
        "generation": int(previous.get("generation", 0)) + 1,
        "updated_at": _now(),
    }
    _atomic_json(STATE_FILE, state)


def switch_vllm(role: str, analysis_model: str | None = None, *, drain_timeout: int = 600) -> dict[str, Any]:
    """Synchronously switch the single inference slot and roll back on failure."""
    target = profile_for(role, analysis_model)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lifecycle = open(LIFECYCLE_LOCK, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lifecycle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another vLLM lifecycle operation is already running") from exc
        previous_state = read_state()
        operation = {
            "id": uuid.uuid4().hex, "target_role": role, "target_model": target["model"],
            "previous_role": previous_state.get("active_role", "stopped"),
            "previous_model": previous_state.get("active_model"),
            "created_at": _now(), "events": [],
        }
        _validate_profile_cache(target)
        _update_operation(operation, "validating", "Validated immutable target profile", 5)
        _atomic_json(STATE_FILE, {**previous_state, "desired_role": role, "state": "draining", "updated_at": _now()})
        lease = open(LEASE_LOCK, "a+", encoding="utf-8")
        try:
            deadline = time.monotonic() + drain_timeout
            while True:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("active inference did not drain before the switch timeout")
                    time.sleep(1)
            _update_operation(operation, "stopping", "Stopping the previous inference profile", 15)
            _quarantine_legacy_containers()
            _stop_and_remove()
            _update_operation(operation, "creating", f"Creating {role} profile on port {target['host_port']}", 30)
            _create(target)
            _update_operation(operation, "loading", f"Loading {target['model']}", 45)
            _wait_for_model(target, max(300, int(target["estimated_load_seconds"]) * 3))
            _update_operation(operation, "smoke_testing", "Verifying the live model", 90)
            _smoke(target)
            _activate(target, role)
            _update_operation(operation, "completed", f"{role.upper()} inference is ready on port {target['host_port']}", 100)
            audit_event("vllm_switch", "success", role=role, model=target["model"], port=target["host_port"])
            return operation
        except Exception as exc:
            operation["error"] = str(exc)
            previous_role = previous_state.get("active_role")
            previous_model = previous_state.get("active_model")
            if previous_role in {"ocr", "analysis"}:
                try:
                    _update_operation(operation, "rolling_back", "Restoring the previous verified profile", 92)
                    _stop_and_remove()
                    previous = profile_for(previous_role, previous_model if previous_role == "analysis" else None)
                    _create(previous)
                    _wait_for_model(previous, max(300, int(previous["estimated_load_seconds"]) * 3))
                    _smoke(previous)
                    _activate(previous, str(previous_state.get("desired_role", previous_role)))
                    _update_operation(operation, "rolled_back", f"Switch failed; restored {previous_role}: {exc}", 100)
                except Exception as rollback_exc:
                    operation["rollback_error"] = str(rollback_exc)
                    _stop_and_remove()
                    _atomic_json(STATE_FILE, {**previous_state, "active_role": "stopped", "state": "failed", "updated_at": _now()})
                    _update_operation(operation, "failed", f"Switch and rollback failed: {exc}; {rollback_exc}", 100)
            else:
                _stop_and_remove()
                _atomic_json(STATE_FILE, {**previous_state, "active_role": "stopped", "state": "failed", "updated_at": _now()})
                _update_operation(operation, "failed", f"Switch failed: {exc}", 100)
            audit_event("vllm_switch", "failure", role=role, model=target["model"], error=str(exc))
            raise
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()
    finally:
        fcntl.flock(lifecycle, fcntl.LOCK_UN)
        lifecycle.close()


def stop_vllm(*, drain_timeout: int = 600) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lifecycle = open(LIFECYCLE_LOCK, "a+", encoding="utf-8")
    lease = None
    try:
        fcntl.flock(lifecycle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_state()
        _atomic_json(STATE_FILE, {**state, "desired_role": "stopped", "state": "draining", "updated_at": _now()})
        lease = open(LEASE_LOCK, "a+", encoding="utf-8")
        deadline = time.monotonic() + drain_timeout
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("active inference did not drain before stop timeout")
                time.sleep(1)
        _quarantine_legacy_containers()
        _stop_and_remove()
        _atomic_json(STATE_FILE, {"schema_version": 2, "desired_role": "stopped", "active_role": "stopped", "state": "stopped", "updated_at": _now()})
    finally:
        if lease is not None:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()
        fcntl.flock(lifecycle, fcntl.LOCK_UN)
        lifecycle.close()


def status() -> dict[str, Any]:
    state = read_state()
    container_status = _container_status()
    active_role = state.get("active_role", "stopped")
    port = OCR_PORT if active_role == "ocr" else ANALYSIS_PORT if active_role == "analysis" else None
    served = _served_model(port) if port and container_status == "running" else ""
    ready = state.get("state") == "ready" and served == state.get("active_model")
    return {
        **state,
        "container": CONTAINER_NAME,
        "container_status": container_status,
        "served_model": served,
        "ready": ready,
        "ocr": {"endpoint": f"http://127.0.0.1:{OCR_PORT}/v1", "available": ready and active_role == "ocr"},
        "analysis": {"endpoint": f"http://127.0.0.1:{ANALYSIS_PORT}/v1", "available": ready and active_role == "analysis"},
        "operation": read_operation(),
    }
