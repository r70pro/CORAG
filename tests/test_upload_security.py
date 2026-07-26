from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path

import pytest
from fastapi import HTTPException
from pypdf import PdfWriter


class ChunkedUpload:
    def __init__(
        self,
        filename: str,
        data: bytes,
        content_type: str,
        *,
        cancel_after_reads: int | None = None,
    ):
        self.filename = filename
        self.content_type = content_type
        self._data = data
        self._offset = 0
        self._reads = 0
        self.cancel_after_reads = cancel_after_reads
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        self._reads += 1
        if self.cancel_after_reads is not None and self._reads > self.cancel_after_reads:
            raise asyncio.CancelledError
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


def make_pdf() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def test_pdf_upload_uses_bounded_reads_unique_name_and_escaped_metadata(
    tmp_path, monkeypatch
):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files
    from api.upload_security import UPLOAD_CHUNK_BYTES

    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    upload = ChunkedUpload(
        "report<img onerror=alert(1)>.pdf",
        make_pdf(),
        "application/pdf",
    )

    result = asyncio.run(upload_pipeline_files(request=None, files=upload))

    stored_name = result["file_paths"][0]
    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", stored_name)
    assert stored_name != upload.filename
    assert result["files"][0]["original_name"] == (
        "report&lt;img onerror=alert(1)&gt;.pdf"
    )
    assert (tmp_path / "workspace" / "uploads" / stored_name).is_file()
    metadata_path = tmp_path / "workspace" / "uploads" / f"{stored_name}.metadata.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "original_name": "report&lt;img onerror=alert(1)&gt;.pdf"
    }
    assert upload.read_sizes
    assert max(upload.read_sizes) == UPLOAD_CHUNK_BYTES
    assert upload.closed


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        (b"not a pdf", "Upload is not a PDF"),
        (b"%PDF-not-parseable", "Upload is not a parseable PDF"),
    ],
)
def test_pdf_upload_rejects_invalid_content_and_removes_partial_file(
    tmp_path, monkeypatch, payload, expected_detail
):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files

    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    upload = ChunkedUpload("bad.pdf", payload, "application/pdf")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_pipeline_files(request=None, files=upload))

    assert exc.value.status_code == 415
    assert exc.value.detail == expected_detail
    assert not list((tmp_path / "workspace" / "uploads").glob("*.pdf"))


def test_pdf_upload_rolls_back_completed_files_on_aggregate_failure(
    tmp_path, monkeypatch
):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files

    pdf = make_pdf()
    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("KIRAG_MAX_PDF_UPLOAD_BYTES", str(len(pdf) + 10))
    uploads = [
        ChunkedUpload("one.pdf", pdf, "application/pdf"),
        ChunkedUpload("two.pdf", pdf, "application/pdf"),
    ]

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_pipeline_files(request=None, files=uploads))

    assert exc.value.status_code == 413
    assert not list((tmp_path / "workspace" / "uploads").glob("*.pdf"))
    assert all(upload.closed for upload in uploads)


def test_pdf_upload_cancellation_removes_partial_file(tmp_path, monkeypatch):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files

    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    upload = ChunkedUpload(
        "cancelled.pdf",
        make_pdf(),
        "application/pdf",
        cancel_after_reads=1,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(upload_pipeline_files(request=None, files=upload))

    assert not list((tmp_path / "workspace" / "uploads").glob("*.pdf"))
    assert upload.closed


def test_markdown_upload_validates_utf8_and_uses_unique_storage_name(monkeypatch):
    from api.routes.rag import upload_markdown
    from api.upload_security import UPLOAD_CHUNK_BYTES
    from indexing_service import CorpusIndexingService

    captured = []

    def fake_index(files, *_args):
        captured.extend(files)
        return iter(["Done"])

    monkeypatch.setattr(
        CorpusIndexingService,
        "add_markdown_to_case",
        staticmethod(fake_index),
    )
    upload = ChunkedUpload(
        "notes<img onerror=alert(1)>.md",
        "# Valid UTF-8\n\n<table onclick=x>".encode("utf-8"),
        "text/markdown",
    )

    result = asyncio.run(
        upload_markdown(files=upload, case_option="new", new_case_name="Case")
    )

    assert result.success
    assert len(captured) == 1
    assert re.fullmatch(r"[0-9a-f]{32}\.md", Path(captured[0].name).name)
    assert "&lt;img onerror=alert(1)&gt;" in captured[0].original_filename
    assert max(upload.read_sizes) == UPLOAD_CHUNK_BYTES
    assert upload.closed


@pytest.mark.parametrize("payload", [b"\xff\xfe\x00", b"# text\x00binary"])
def test_markdown_upload_rejects_binary_payload(monkeypatch, payload):
    from api.routes.rag import upload_markdown

    upload = ChunkedUpload("binary.md", payload, "text/markdown")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            upload_markdown(files=upload, case_option="new", new_case_name="Case")
        )

    assert exc.value.status_code == 415
    assert upload.closed


def test_upload_file_count_limit_is_enforced_before_reading(tmp_path, monkeypatch):
    import settings_manager
    from api.routes.pipeline import upload_pipeline_files

    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("KIRAG_MAX_PDF_FILES", "1")
    uploads = [
        ChunkedUpload("one.pdf", make_pdf(), "application/pdf"),
        ChunkedUpload("two.pdf", make_pdf(), "application/pdf"),
    ]

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_pipeline_files(request=None, files=uploads))

    assert exc.value.status_code == 413
    assert not any(upload.read_sizes for upload in uploads)
    assert all(upload.closed for upload in uploads)


def test_fastapi_applies_restrictive_security_headers():
    from fastapi.testclient import TestClient

    from api.main import app

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_fastapi_multipart_parser_enforces_file_count(tmp_path, monkeypatch):
    import settings_manager
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("KIRAG_API_KEY", "upload-test-key")
    monkeypatch.setenv("KIRAG_MAX_PDF_FILES", "1")
    monkeypatch.setattr(settings_manager, "WORKSPACE_DIR", str(tmp_path / "workspace"))

    response = TestClient(app).post(
        "/api/pipeline/upload",
        headers={"X-API-Key": "upload-test-key"},
        files=[
            ("files", ("one.pdf", b"%PDF-invalid", "application/pdf")),
            ("files", ("two.pdf", b"%PDF-invalid", "application/pdf")),
        ],
    )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "payload_too_large",
        "message": "Too many uploaded files",
    }
    assert response.headers["x-content-type-options"] == "nosniff"
