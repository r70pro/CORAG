"""Focused coverage for filesystem browsing and privileged container routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pypdf import PdfWriter

from api.models import (
    AnalysisContextModeRequest,
    AnalysisSwitchRequest,
    DockerCreateRequest,
    EmbeddingConfigRequest,
    IndexRunRequest,
    StartupModeRequest,
)
from api.routes import docker, documents, rag


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    run = workspace / "run_case"
    markdown = run / "markdown" / "inputs"
    inputs = run / "inputs"
    results = run / "results"
    markdown.mkdir(parents=True)
    inputs.mkdir()
    results.mkdir()
    (markdown / "record.md").write_text("first page\nsecond page", encoding="utf-8")
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with (inputs / "record.pdf").open("wb") as pdf:
        writer.write(pdf)
    metadata = {
        "metadata": {"Source-File": "record.pdf"},
        "attributes": {"pdf_page_numbers": [[0, 10, 1], [11, 22, 2]]},
    }
    (results / "record.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(
        "settings_manager.get_available_runs",
        lambda: [("run_case", str(run))],
    )
    return run


def test_document_routes_success_and_boundaries(completed_run: Path):
    runs = documents.list_runs()
    assert runs[0]["run_name"] == "run_case"
    assert runs[0]["has_pdf"] is True
    assert documents.list_run_files("run_case") == ["record.md"]
    assert documents.get_markdown("run_case", "record.md").body == b"first page\nsecond page"

    archive = documents.download_run_markdown("run_case")
    assert archive.media_type == "application/zip"
    pdf = documents.get_run_pdf("run_case")
    assert str(pdf.path).endswith("record.pdf")

    info = documents.get_run_doc_info("run_case", "record.md")
    assert info["total_pages"] == 1
    assert info["pages_markdown"] == {"1": "first page", "2": "second page"}

    for operation in (
        lambda: documents.list_run_files("../escape"),
        lambda: documents.get_markdown("run_case", "../record.md"),
    ):
        with pytest.raises(HTTPException) as error:
            operation()
        assert error.value.status_code == 400


def test_document_routes_missing_and_fallbacks(completed_run: Path):
    (completed_run / "markdown" / "inputs" / "record.md").unlink()
    assert documents.list_run_files("run_case") == []
    with pytest.raises(HTTPException) as error:
        documents.download_run_markdown("run_case")
    assert error.value.status_code == 404

    (completed_run / "inputs" / "record.pdf").unlink()
    with pytest.raises(HTTPException) as error:
        documents.get_run_pdf("run_case")
    assert error.value.status_code == 404

    info = documents.get_run_doc_info("run_case")
    assert info["total_pages"] == 1
    assert info["pages_markdown"] == {}


def test_docker_read_and_switch_routes(monkeypatch: pytest.MonkeyPatch):
    status = {"served_model": "model"}
    monkeypatch.setattr("analysis_profiles.analysis_status", lambda: status)
    monkeypatch.setattr("analysis_profiles.start_switch", lambda model: {"target": model})
    monkeypatch.setattr("analysis_profiles.get_operation", lambda operation: {"id": operation})
    monkeypatch.setattr(
        "docker_manager.get_cached_models_info", lambda: (["model"], {"model": 4096})
    )
    monkeypatch.setattr("settings_manager.load_settings", lambda **kwargs: {"docker_port": 8123})
    monkeypatch.setattr("vllm_lifecycle.status", lambda: {
        "active_role": "ocr", "ready": True,
        "ocr": {"available": True}, "analysis": {"available": False},
    })
    monkeypatch.setattr("docker_manager.get_docker_logs", lambda **kwargs: "logs")
    monkeypatch.setattr("docker_manager.get_docker_status", lambda container: "running")

    assert docker.get_analysis_profiles() == status
    assert docker.get_analysis_model_status() == status
    request = AnalysisSwitchRequest(target_model="model", confirmation="SWITCH")
    assert docker.switch_analysis_model(request) == {"target": "model"}
    assert docker.get_analysis_switch_operation("operation") == {"id": "operation"}
    assert docker.get_models().models == ["model"]
    assert docker.get_status("ocr").status == "ready"
    assert docker.get_status("analysis").status == "stopped"
    assert docker.get_logs(10, "analysis").logs == "logs"

    monkeypatch.setattr("analysis_profiles.get_operation", lambda operation: None)
    with pytest.raises(HTTPException) as error:
        docker.get_analysis_switch_operation("missing")
    assert error.value.status_code == 404


def test_docker_privileged_success_routes(monkeypatch: pytest.MonkeyPatch):
    saved = Mock(return_value="saved")
    settings = {"model_name": "allenai/olmOCR-2-7B-1025-FP8"}
    monkeypatch.setattr("analysis_profiles.switch_in_progress", lambda: False)
    monkeypatch.setattr("settings_manager.load_settings", lambda **kwargs: settings.copy())
    monkeypatch.setattr("settings_manager.save_settings", saved)
    monkeypatch.setattr("docker_manager.set_extended_analysis_context", lambda enabled: (True, "ok"))
    monkeypatch.setattr("docker_manager.set_vllm_role_running", lambda role, running: (True, "ok"))
    monkeypatch.setattr("docker_manager.start_docker_container", lambda: (True, "started"))
    monkeypatch.setattr("docker_manager.stop_docker_container", lambda: (True, "stopped"))
    monkeypatch.setattr("docker_manager.shutdown_docker_container", lambda: (True, "removed"))
    monkeypatch.setattr("docker_manager.create_docker_container", lambda *args: (True, "created"))
    monkeypatch.setattr("rag.analyzer.invalidate_model_cache", lambda: None)
    monkeypatch.setattr("vllm_lifecycle.switch_vllm", lambda *args: {"state": "completed"})
    monkeypatch.setattr("vllm_lifecycle.stop_vllm", lambda: None)

    run = asyncio.run
    with pytest.raises(HTTPException) as retired:
        run(docker.set_analysis_context_mode(AnalysisContextModeRequest(extended=True)))
    assert retired.value.status_code == 410
    for mode in ("analysis", "ocr", "stopped"):
        assert run(docker.set_startup_mode(StartupModeRequest(mode=mode))).success
    assert run(docker.start_container()).success
    assert run(docker.start_role_container("analysis")).success
    assert run(docker.stop_role_container("ocr")).success
    assert run(docker.stop_container()).success
    assert run(docker.shutdown_container()).success
    created = run(docker.create_container(DockerCreateRequest(hf_token="token")))
    assert created.success
    assert saved.called


def test_docker_privileged_failure_routes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("analysis_profiles.switch_in_progress", lambda: True)
    with pytest.raises(HTTPException) as error:
        asyncio.run(docker.start_role_container("analysis"))
    assert error.value.status_code == 409
    with pytest.raises(HTTPException):
        asyncio.run(docker.set_analysis_context_mode(AnalysisContextModeRequest(extended=False)))

    monkeypatch.setattr("analysis_profiles.switch_in_progress", lambda: False)
    monkeypatch.setattr("vllm_lifecycle.switch_vllm", lambda *args: (_ for _ in ()).throw(RuntimeError("failed")))
    monkeypatch.setattr("vllm_lifecycle.stop_vllm", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(HTTPException):
        asyncio.run(docker.start_container())
    with pytest.raises(HTTPException):
        asyncio.run(docker.stop_container())
    with pytest.raises(HTTPException):
        asyncio.run(docker.shutdown_container())


def test_rag_management_and_corpus_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("rag.cache.clear_chat_history", lambda session: None)
    assert rag.delete_chat_history("session_1").success
    with pytest.raises(HTTPException):
        rag.delete_chat_history("bad/session")

    monkeypatch.setattr(
        "embedding_pipeline_ui.save_embedding_pipeline_settings", lambda *args: "✅ saved"
    )
    monkeypatch.setattr("embedding_pipeline_ui.purge_embedding_cache", lambda: "✅ purged")
    config = EmbeddingConfigRequest(
        embedding_model="model",
        embedding_device="cpu",
        chunk_size=800,
        chunk_overlap=100,
        embedding_batch_size=8,
    )
    assert rag.save_embedding_config(config).success
    assert rag.purge_cache().success

    monkeypatch.setattr(
        "rag_infra_manager.get_rag_service_status",
        lambda: {name: "healthy" for name in ("postgres", "redis", "minio", "qdrant")},
    )
    assert rag.infra_status().postgres == "healthy"

    monkeypatch.setattr(
        "rag.db.get_corpus_stats",
        lambda: {
            "indexed_runs": 1,
            "indexed_documents": 2,
            "total_chunks": 3,
            "unique_authors": 4,
            "earliest_date": "2020-01-01",
            "latest_date": "2021-01-01",
        },
    )
    monkeypatch.setattr("rag.embedding.get_collection_info", lambda *args: {"points_count": 3})
    monkeypatch.setattr(
        "rag.db.get_indexed_runs",
        lambda: [{"run_id": "run_case", "display_name": "Case"}],
    )
    assert rag.corpus_stats().vectors_count == 3
    assert rag.list_cases()[0].label == "Case"

    workspace = tmp_path / "workspace"
    run_dir = workspace / "run_case"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr("settings_manager.WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(
        "indexing_service.CorpusIndexingService.index_run",
        lambda path, force: iter(["✅ indexed"]),
    )
    monkeypatch.setattr(
        "indexing_service.CorpusIndexingService.index_all_runs",
        lambda force: iter(["Done"]),
    )
    request = IndexRunRequest(run_dir="run_case")
    assert rag.index_run(request).success
    assert rag.index_all_runs().success
    assert rag.stream_index_run(request).media_type == "text/event-stream"
    assert rag.stream_index_all_runs().media_type == "text/event-stream"

    success_events = list(rag._indexing_event_stream(["working", "✅ done"]))
    assert success_events[-1] == "data: [DONE]\n\n"
    failure_events = list(rag._indexing_event_stream(["failed"]))
    assert "indexing_failed" in failure_events[-1]


def test_rag_management_failure_routes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "embedding_pipeline_ui.save_embedding_pipeline_settings", lambda *args: "failed"
    )
    monkeypatch.setattr("embedding_pipeline_ui.purge_embedding_cache", lambda: "failed")
    config = EmbeddingConfigRequest(
        embedding_model="model",
        embedding_device="cpu",
        chunk_size=800,
        chunk_overlap=100,
        embedding_batch_size=8,
    )
    with pytest.raises(HTTPException):
        rag.save_embedding_config(config)
    with pytest.raises(HTTPException):
        rag.purge_cache()

    monkeypatch.setattr("rag.db.get_corpus_stats", Mock(side_effect=RuntimeError("down")))
    monkeypatch.setattr("rag.db.get_indexed_runs", Mock(side_effect=RuntimeError("down")))
    with pytest.raises(HTTPException):
        rag.corpus_stats()
    with pytest.raises(HTTPException):
        rag.list_cases()
