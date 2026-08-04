from __future__ import annotations

import json
from pathlib import Path

import pytest

import vllm_lifecycle as lifecycle


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(lifecycle, "STATE_FILE", runtime_dir / "state.json")
    monkeypatch.setattr(lifecycle, "OPERATION_FILE", runtime_dir / "operation.json")
    monkeypatch.setattr(lifecycle, "LIFECYCLE_LOCK", runtime_dir / "lifecycle.lock")
    monkeypatch.setattr(lifecycle, "LEASE_LOCK", runtime_dir / "inference.lock")
    return runtime_dir


def test_inference_lease_rejects_inactive_role(runtime):
    lifecycle._atomic_json(
        lifecycle.STATE_FILE,
        {"state": "ready", "active_role": "ocr", "active_model": "ocr"},
    )
    with pytest.raises(RuntimeError, match="active role is ocr"):
        with lifecycle.inference_lease("analysis"):
            pass


def test_switch_orders_stop_before_create_and_publishes_one_profile(runtime, monkeypatch):
    events: list[str] = []
    profile = {
        "role": "analysis", "model": "gemma", "revision": "a" * 40,
        "host_port": 8002, "estimated_load_seconds": 1,
    }
    monkeypatch.setattr(lifecycle, "profile_for", lambda role, model=None: dict(profile))
    monkeypatch.setattr(lifecycle, "_validate_profile_cache", lambda value: events.append("validate"))
    monkeypatch.setattr(lifecycle, "_quarantine_legacy_containers", lambda: events.append("legacy"))
    monkeypatch.setattr(lifecycle, "_stop_and_remove", lambda: events.append("stop"))
    monkeypatch.setattr(lifecycle, "_create", lambda value: events.append("create"))
    monkeypatch.setattr(lifecycle, "_wait_for_model", lambda value, timeout: events.append("ready"))
    monkeypatch.setattr(lifecycle, "_smoke", lambda value: events.append("smoke"))

    operation = lifecycle.switch_vllm("analysis", "gemma")

    assert operation["state"] == "completed"
    assert events == ["validate", "legacy", "stop", "create", "ready", "smoke"]
    state = json.loads(lifecycle.STATE_FILE.read_text())
    assert state["active_role"] == "analysis"
    assert state["host_port"] == 8002


def test_failed_switch_rolls_back_previous_verified_profile(runtime, monkeypatch):
    lifecycle._atomic_json(lifecycle.STATE_FILE, {
        "state": "ready", "active_role": "ocr", "desired_role": "ocr",
        "active_model": "ocr", "generation": 2,
    })
    profiles = {
        "analysis": {"role": "analysis", "model": "gemma", "revision": "a" * 40, "host_port": 8002, "estimated_load_seconds": 1},
        "ocr": {"role": "ocr", "model": "ocr", "revision": "b" * 40, "host_port": 8000, "estimated_load_seconds": 1},
    }
    monkeypatch.setattr(lifecycle, "profile_for", lambda role, model=None: dict(profiles[role]))
    monkeypatch.setattr(lifecycle, "_validate_profile_cache", lambda value: None)
    monkeypatch.setattr(lifecycle, "_quarantine_legacy_containers", lambda: None)
    monkeypatch.setattr(lifecycle, "_stop_and_remove", lambda: None)
    created: list[str] = []
    monkeypatch.setattr(lifecycle, "_create", lambda value: created.append(value["role"]))
    def wait(value, timeout):
        if value["role"] == "analysis":
            raise RuntimeError("target failed")
    monkeypatch.setattr(lifecycle, "_wait_for_model", wait)
    monkeypatch.setattr(lifecycle, "_smoke", lambda value: None)

    with pytest.raises(RuntimeError, match="target failed"):
        lifecycle.switch_vllm("analysis", "gemma")

    assert created == ["analysis", "ocr"]
    state = lifecycle.read_state()
    assert state["active_role"] == "ocr"
    assert lifecycle.read_operation()["state"] == "rolled_back"
