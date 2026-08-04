import re
from pathlib import Path
from unittest.mock import patch

import yaml

import system_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[1]
DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def test_every_compose_service_is_labeled_and_digest_pinned():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.rag.yml").read_text())

    for service_name, service in compose["services"].items():
        assert service["labels"]["com.kirag.managed"] == "true", service_name
        assert DIGEST_IMAGE_RE.fullmatch(service["image"]), service_name


def test_integration_compose_uses_only_disposable_storage():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.integration.yml").read_text())

    for service_name, service in compose["services"].items():
        assert DIGEST_IMAGE_RE.fullmatch(service["image"]), service_name
        assert "container_name" not in service, service_name
        assert not service.get("volumes"), service_name
        assert service.get("tmpfs"), service_name
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1::"), service_name


def test_production_compose_supervises_offline_vllm():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.production.yml").read_text())
    assert "vllm-analysis" not in compose["services"]
    vllm = compose["services"]["vllm"]
    assert vllm["container_name"] == "kirag_vllm"
    assert vllm["restart"] == "unless-stopped"
    assert vllm["init"] is True
    assert vllm["healthcheck"]["start_period"]
    assert vllm["environment"]["HF_HUB_OFFLINE"] == "1"
    assert vllm["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "KIRAG_VLLM_ROLE" in vllm["environment"]
    assert "KIRAG_VLLM_MODEL" in vllm["environment"]
    assert "KIRAG_VLLM_HOST_PORT" in str(vllm["ports"])
    launcher = (REPO_ROOT / "scripts/vllm-entrypoint.sh").read_text()
    assert "--language-model-only" in launcher
    assert "--reasoning-parser qwen3" in launcher
    assert "Invalid OCR model" in launcher


def test_analysis_switcher_pins_both_verified_models():
    switcher = (REPO_ROOT / "scripts/switch-analysis-model.py").read_text()
    assert "ANALYSIS_PROFILES" in switcher
    assert "start_switch" in switcher
    assert "get_operation" in switcher
    assert "TERMINAL_STATES" in switcher


def test_model_deletion_emits_audit_event(tmp_path):
    model_dir = tmp_path / "models--example--unused"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"weights")
    model_info = {
        "id": "example/unused",
        "name": "unused",
        "folder": model_dir.name,
        "path": str(model_dir),
        "size_bytes": 7,
        "is_active": False,
    }

    with (
        patch(
            "system_diagnostics.get_installed_models_data",
            return_value={"models": [model_info]},
        ),
        patch("system_diagnostics.audit_event") as audit,
    ):
        success, _, deleted, reclaimed = system_diagnostics.delete_installed_models(
            ["example/unused"]
        )

    assert success is True
    assert deleted == ["example/unused"]
    assert reclaimed == 7
    assert not model_dir.exists()
    audit.assert_any_call(
        "model_delete",
        "success",
        model="example/unused",
        reclaimed_bytes=7,
    )
