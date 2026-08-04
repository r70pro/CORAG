import json
from pathlib import Path
from unittest.mock import patch

import analysis_profiles


def _temporary_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_profiles, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(analysis_profiles, "PROFILE_STATE", tmp_path / "analysis-profile.json")
    monkeypatch.setattr(analysis_profiles, "OPERATIONS_DIR", tmp_path / "operations")
    monkeypatch.setattr(analysis_profiles, "SWITCH_LOCK", tmp_path / "switch.lock")


def test_runtime_profile_requires_verified_revision(tmp_path, monkeypatch):
    _temporary_runtime(tmp_path, monkeypatch)
    profile = analysis_profiles.ANALYSIS_PROFILES["google/gemma-4-31B-it"]
    analysis_profiles._atomic_json(
        analysis_profiles.PROFILE_STATE,
        {"model": profile["model"], "revision": profile["revision"]},
    )
    assert analysis_profiles.read_runtime_profile()["model"] == profile["model"]
    analysis_profiles._atomic_json(
        analysis_profiles.PROFILE_STATE,
        {"model": profile["model"], "revision": "0" * 40},
    )
    assert analysis_profiles.read_runtime_profile() is None


def test_cache_validation_checks_revision_and_indexed_shards(tmp_path, monkeypatch):
    model = "google/gemma-4-31B-it"
    profile = analysis_profiles.ANALYSIS_PROFILES[model]
    monkeypatch.setenv("KIRAG_HF_HOME", str(tmp_path))
    repository = tmp_path / "hub" / "models--google--gemma-4-31B-it"
    snapshot = repository / "snapshots" / profile["revision"]
    (repository / "refs").mkdir(parents=True)
    (repository / "blobs").mkdir()
    snapshot.mkdir(parents=True)
    (repository / "refs" / "main").write_text(profile["revision"], encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model.safetensors"}}), encoding="utf-8"
    )
    complete, error, resolved = analysis_profiles.validate_cached_profile(model)
    assert complete is True
    assert error == ""
    assert resolved == str(snapshot)
    (snapshot / "model.safetensors").unlink()
    complete, error, _ = analysis_profiles.validate_cached_profile(model)
    assert complete is False
    assert "indexed weight shard" in error


def test_switch_operation_completes_and_persists_progress(tmp_path, monkeypatch):
    _temporary_runtime(tmp_path, monkeypatch)
    operation = {
        "id": "a" * 32,
        "target_model": "google/gemma-4-31B-it",
        "previous_model": "Qwen/Qwen3.6-35B-A3B",
        "events": [],
    }
    lock_path = tmp_path / "held.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    with (
        patch("analysis_profiles.validate_cached_profile", return_value=(True, "", "/snapshot")),
        patch("analysis_profiles._compose_recreate"),
        patch("analysis_profiles._wait_for_model"),
        patch("analysis_profiles._smoke"),
        patch("analysis_profiles._activate") as activate,
    ):
        analysis_profiles._run_switch(operation, lock_handle)
    stored = analysis_profiles.get_operation(operation["id"])
    assert stored["state"] == "completed"
    assert stored["progress"] == 100
    activate.assert_called_once_with("google/gemma-4-31B-it")


def test_switch_failure_rolls_back_previous_profile(tmp_path, monkeypatch):
    _temporary_runtime(tmp_path, monkeypatch)
    operation = {
        "id": "b" * 32,
        "target_model": "google/gemma-4-31B-it",
        "previous_model": "Qwen/Qwen3.6-35B-A3B",
        "events": [],
    }
    lock_handle = (tmp_path / "held.lock").open("a+")
    with (
        patch("analysis_profiles.validate_cached_profile", return_value=(True, "", "/snapshot")),
        patch("analysis_profiles._compose_recreate", side_effect=RuntimeError("load failed")),
        patch("vllm_lifecycle.status", return_value={
            "ready": True, "active_model": "Qwen/Qwen3.6-35B-A3B"
        }),
        patch("analysis_profiles._wait_for_model"),
        patch("analysis_profiles._smoke"),
        patch("analysis_profiles._activate") as activate,
    ):
        analysis_profiles._run_switch(operation, lock_handle)
    stored = analysis_profiles.get_operation(operation["id"])
    assert stored["state"] == "rolled_back"
    assert "load failed" in stored["message"]
    activate.assert_not_called()


def test_pending_operation_is_resumed_after_process_restart(tmp_path, monkeypatch):
    _temporary_runtime(tmp_path, monkeypatch)
    operation = {
        "id": "c" * 32,
        "state": "loading_weights",
        "message": "loading",
        "progress": 40,
        "target_model": "google/gemma-4-31B-it",
        "previous_model": "Qwen/Qwen3.6-35B-A3B",
        "events": [],
    }
    analysis_profiles._save_operation(operation)
    monkeypatch.setattr(analysis_profiles, "_active_thread", None)
    with patch("analysis_profiles.threading.Thread") as thread:
        thread.return_value.is_alive.return_value = False
        assert analysis_profiles.resume_pending_switch() is True
        thread.return_value.start.assert_called_once()
        lock_handle = thread.call_args.kwargs["args"][1]
        import fcntl
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    stored = analysis_profiles.get_operation(operation["id"])
    assert any("Resuming guarded switch" in event["message"] for event in stored["events"])
