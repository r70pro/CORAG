"""Regression tests for filesystem disclosure, Gradio serving, and API auth."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from gradio.route_utils import file_fetch
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.auth import (
    authentication_configured,
    requested_api_bind_host,
    require_safe_bind,
    verify_admin_key,
    verify_api_key,
)
from api.main import require_authentication
from api.models import DeleteCasesRequest, IndexRunRequest, PipelineStartRequest
from api.routes import documents
from api.routes.pipeline import start_pipeline
from api.routes.rag import delete_cases, index_run
from gradio_security import get_gradio_path_config
from path_security import (
    PathSecurityError,
    require_approved_file,
    resolve_file_under,
    resolve_run_under,
    resolve_under,
    validate_filename,
    validate_run_name,
)


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "run_case" / "markdown" / "inputs").mkdir(parents=True)
    (workspace / "run_case" / "inputs").mkdir()
    return workspace


@pytest.mark.parametrize(
    "run_name,filename",
    [
        ("../run_case", "document.md"),
        ("run_case", "../secret.md"),
        ("run_case", "/etc/passwd"),
        ("run_case", "document.pdf"),
        ("case", "document.md"),
        ("run_case", "bad\x00.md"),
    ],
)
def test_document_path_inputs_are_rejected(tmp_path, run_name, filename):
    workspace = _workspace(tmp_path)
    with pytest.raises(PathSecurityError):
        run_dir = resolve_run_under(workspace, run_name)
        resolve_file_under(run_dir / "markdown" / "inputs", filename, {".md"})


def test_document_reader_rejects_symlink_escape_without_disclosure(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    secret = tmp_path / "secret.md"
    secret.write_text("DO_NOT_DISCLOSE", encoding="utf-8")
    (workspace / "run_case" / "markdown" / "inputs" / "escape.md").symlink_to(secret)
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))

    with pytest.raises(HTTPException) as exc:
        documents.get_markdown("run_case", "escape.md")

    assert exc.value.status_code in {400, 403}
    assert str(tmp_path) not in str(exc.value.detail)
    assert "DO_NOT_DISCLOSE" not in str(exc.value.detail)


def test_pipeline_and_indexing_reject_absolute_paths():
    with pytest.raises(HTTPException) as exc:
        start_pipeline(PipelineStartRequest(file_paths=["/etc/passwd"]))
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        index_run(IndexRunRequest(run_dir="/tmp/run_case"))
    assert exc.value.status_code == 400


def test_deletion_rejects_path_like_run_identifier():
    with pytest.raises(HTTPException) as exc:
        delete_cases(DeleteCasesRequest(run_ids=["../../run_case"]))
    assert exc.value.status_code == 400


def test_gradio_request_handler_denies_dotenv(tmp_path):
    project = tmp_path / "project"
    workspace = project / "workspace"
    project.mkdir()
    dotenv = project / ".env"
    dotenv.write_text("KIRAG_API_KEY=DO_NOT_DISCLOSE", encoding="utf-8")
    allowed, blocked = get_gradio_path_config(workspace, project, tmp_path / "home")
    blocks = SimpleNamespace(allowed_paths=allowed, blocked_paths=blocked)
    request = SimpleNamespace(headers={})

    with pytest.raises(StarletteHTTPException) as exc:
        file_fetch(str(dotenv), request, blocks, str(workspace / "uploads"))

    assert exc.value.status_code == 403


def test_api_authentication_fails_closed_and_uses_admin_key(monkeypatch):
    monkeypatch.delenv("KIRAG_API_KEY", raising=False)
    monkeypatch.delenv("KIRAG_ADMIN_API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc:
        verify_api_key(key_from_header=None, bearer=None)
    assert exc.value.status_code == 401

    monkeypatch.setenv("KIRAG_API_KEY", "api-secret")
    with pytest.raises(HTTPException) as exc:
        verify_admin_key(admin_from_header=None, key_from_header="api-secret", bearer=None)
    assert exc.value.status_code == 403

    monkeypatch.setenv("KIRAG_ADMIN_API_KEY", "admin-secret")
    assert (
        verify_admin_key(
            admin_from_header="admin-secret", key_from_header=None, bearer=None
        )
        == "admin-secret"
    )


def test_unauthenticated_mutation_request_is_rejected(monkeypatch):
    monkeypatch.delenv("KIRAG_API_KEY", raising=False)
    monkeypatch.delenv("KIRAG_ADMIN_API_KEY", raising=False)
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "DELETE",
        "scheme": "http",
        "path": "/api/rag/cases/delete",
        "raw_path": b"/api/rag/cases/delete",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8001),
    }
    called = False

    async def call_next(_request):
        nonlocal called
        called = True
        return JSONResponse({"unexpected": True})

    response = asyncio.run(require_authentication(Request(scope), call_next))
    assert response.status_code == 401
    assert not called


def test_non_loopback_binding_requires_authentication(monkeypatch):
    monkeypatch.delenv("KIRAG_API_KEY", raising=False)
    monkeypatch.delenv("KIRAG_ADMIN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_safe_bind("0.0.0.0")
    assert require_safe_bind("127.0.0.1") == "127.0.0.1"

    monkeypatch.setenv("KIRAG_API_KEY", "configured")
    assert require_safe_bind("0.0.0.0") == "0.0.0.0"


def test_uvicorn_non_loopback_argument_is_detected(monkeypatch):
    monkeypatch.delenv("KIRAG_API_HOST", raising=False)
    assert requested_api_bind_host(["api.main:app"]) == "127.0.0.1"
    assert requested_api_bind_host(["api.main:app", "--host", "0.0.0.0"]) == "0.0.0.0"
    assert requested_api_bind_host(["api.main:app", "--host=::"]) == "::"


def test_authorization_helper_branches(monkeypatch):
    monkeypatch.setenv("KIRAG_API_KEY", "api-secret")
    monkeypatch.setenv("KIRAG_ADMIN_API_KEY", "admin-secret")
    assert authentication_configured() is True
    assert verify_api_key(key_from_header="api-secret", bearer=None) == "api-secret"
    assert verify_api_key(
        key_from_header=None,
        bearer=SimpleNamespace(credentials="admin-secret"),
    ) == "admin-secret"

    with pytest.raises(HTTPException) as exc:
        verify_admin_key(
            admin_from_header="wrong",
            key_from_header="admin-secret",
            bearer=SimpleNamespace(credentials="admin-secret"),
        )
    assert exc.value.status_code == 403

    assert require_safe_bind(" localhost ", authenticated=False) == "localhost"
    assert require_safe_bind("service.internal", authenticated=True) == "service.internal"
    with pytest.raises(RuntimeError):
        require_safe_bind("service.internal", authenticated=False)

    monkeypatch.setenv("KIRAG_API_HOST", " 0.0.0.0 ")
    assert requested_api_bind_host([]) == "0.0.0.0"
    monkeypatch.delenv("KIRAG_API_HOST")
    monkeypatch.setattr(sys, "argv", ["uvicorn", "api.main:app", "--host", "::1"])
    assert requested_api_bind_host() == "::1"


@pytest.mark.parametrize(
    "part",
    ["", "bad\x00name", "/absolute", "nested/name", r"nested\name", ".", ".."],
)
def test_resolve_under_rejects_every_invalid_component(tmp_path, part):
    with pytest.raises(PathSecurityError):
        resolve_under(tmp_path, part)


@pytest.mark.parametrize(
    "filename",
    ["", "bad\x00.md", "/absolute.md", "nested/file.md", r"nested\file.md", ".", ".."],
)
def test_validate_filename_rejects_every_invalid_shape(filename):
    with pytest.raises(PathSecurityError):
        validate_filename(filename, {"md"})


def test_path_security_helper_branches(tmp_path):
    base = tmp_path / "approved"
    base.mkdir()
    approved = base / "report.MD"
    approved.write_text("safe", encoding="utf-8")

    assert validate_run_name("run_case-123") == "run_case-123"
    with pytest.raises(PathSecurityError):
        validate_run_name("run_\x00case")
    assert validate_filename("report.MD", {"md"}) == "report.MD"
    assert require_approved_file(approved, [base], {".md"}) == approved

    with pytest.raises(PathSecurityError):
        require_approved_file("bad\x00.md", [base], {".md"})
    with pytest.raises(PathSecurityError):
        require_approved_file(base, [base], {".md"})
    with pytest.raises(PathSecurityError):
        require_approved_file(tmp_path / "outside.md", [base], {".md"})

    extension_named_base = tmp_path / "approved.md"
    extension_named_base.mkdir()
    with pytest.raises(PathSecurityError):
        require_approved_file(extension_named_base, [extension_named_base], {".md"})


def test_require_approved_file_rejects_symlink_escape(tmp_path):
    base = tmp_path / "approved"
    base.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = base / "link.md"
    link.symlink_to(outside)

    with pytest.raises(PathSecurityError):
        require_approved_file(link, [base], {".md"})
