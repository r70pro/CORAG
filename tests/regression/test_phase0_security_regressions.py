"""Phase 0 security contracts.

Known defects are strict xfails: they remain visible in the normal suite and
become ordinary failures under ``pytest --runxfail`` for the red baseline.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote, unquote

import pytest
from fastapi import HTTPException

from api.routes import documents
from html_utils import make_case_dashboard_html

pytestmark = pytest.mark.phase0_regression


class _InMemoryUpload:
    def __init__(self, filename="oversized.pdf"):
        self.filename = filename
        self._data = b"%PDF-" + b"A" * 64
        self._offset = 0

    async def read(self, size):
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _make_run(tmp_path):
    workspace = tmp_path / "workspace"
    inputs = workspace / "run_case" / "markdown" / "inputs"
    inputs.mkdir(parents=True)
    (workspace / "run_case" / "inputs").mkdir()
    return workspace


def test_document_info_rejects_absolute_filename(tmp_path, monkeypatch):
    workspace = _make_run(tmp_path)
    secret = tmp_path / "outside-secret.md"
    secret.write_text("PHASE0_ABSOLUTE_PATH_SECRET", encoding="utf-8")
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))

    with pytest.raises(HTTPException) as exc:
        documents.get_run_doc_info("run_case", filename=str(secret))

    assert exc.value.status_code in {400, 403}


def test_document_info_rejects_dot_dot_component(tmp_path, monkeypatch):
    workspace = _make_run(tmp_path)
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))

    with pytest.raises(HTTPException) as exc:
        documents.get_run_doc_info("run_case", filename="../outside-secret.md")

    assert exc.value.status_code == 400


def test_document_info_rejects_url_encoded_traversal(tmp_path, monkeypatch):
    workspace = _make_run(tmp_path)
    secret = tmp_path / "encoded-outside-secret.md"
    secret.write_text("PHASE0_URL_ENCODED_SECRET", encoding="utf-8")
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))

    encoded = quote(str(secret), safe="")
    decoded_by_router = unquote(encoded)

    with pytest.raises(HTTPException) as exc:
        documents.get_run_doc_info("run_case", filename=decoded_by_router)

    assert exc.value.status_code in {400, 403}


def test_markdown_reader_rejects_symlink_escape(tmp_path, monkeypatch):
    workspace = _make_run(tmp_path)
    secret = tmp_path / "symlink-outside-secret.md"
    secret.write_text("PHASE0_SYMLINK_SECRET", encoding="utf-8")
    (workspace / "run_case" / "markdown" / "inputs" / "escape.md").symlink_to(secret)
    monkeypatch.setattr(documents, "WORKSPACE_DIR", str(workspace))

    with pytest.raises(HTTPException) as exc:
        documents.get_markdown("run_case", "escape.md")

    assert exc.value.status_code in {400, 403}


def test_settings_mutation_rejects_unauthorized_request(monkeypatch):
    from api.main import verify_api_key

    monkeypatch.delenv("KIRAG_API_KEY", raising=False)

    with pytest.raises(HTTPException) as exc:
        verify_api_key(key_from_header=None, bearer=None)

    assert exc.value.status_code == 401


def test_pipeline_upload_rejects_oversized_payload(tmp_path, monkeypatch):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files

    monkeypatch.setenv("KIRAG_MAX_UPLOAD_BYTES", "32")
    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_pipeline_files(request=None, files=_InMemoryUpload()))

    assert exc.value.status_code == 413
    assert not (tmp_path / "workspace" / "uploads" / "oversized.pdf").exists()


def test_markdown_upload_rejects_oversized_payload(monkeypatch):
    from api.routes.rag import upload_markdown
    from indexing_service import CorpusIndexingService

    monkeypatch.setenv("KIRAG_MAX_UPLOAD_BYTES", "32")
    monkeypatch.setattr(
        CorpusIndexingService,
        "add_markdown_to_case",
        staticmethod(lambda *_args, **_kwargs: iter(["Done"])),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            upload_markdown(
                files=_InMemoryUpload("oversized.md"),
                case_option="new",
                new_case_name="Phase 0",
            )
        )

    assert exc.value.status_code == 413


def test_case_dashboard_escapes_malicious_html():
    payload = "<img src=x onerror='window.phase0Xss=1'>"
    html = make_case_dashboard_html(
        [
            {
                "run_id": "case-1",
                "run_dir": "/workspace/run_case",
                "total_documents": 1,
                "total_chunks": 1,
                "unique_authors": 1,
            }
        ],
        {"case-1": {"names": [payload], "dob": payload, "injuries": [payload]}},
    )

    assert "<img" not in html.lower()
    assert "&lt;img" in html.lower()
