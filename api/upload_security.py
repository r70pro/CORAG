"""Shared limits and validation helpers for untrusted uploads."""

from __future__ import annotations

import codecs
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.errors import error_response

UPLOAD_CHUNK_BYTES = 64 * 1024


def _positive_env(name: str, default: int, legacy_name: str | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None and legacy_name:
        raw = os.environ.get(legacy_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class UploadLimits:
    max_files: int
    max_file_bytes: int
    max_total_bytes: int


def pdf_upload_limits() -> UploadLimits:
    return UploadLimits(
        max_files=_positive_env("KIRAG_MAX_PDF_FILES", 20),
        max_file_bytes=_positive_env(
            "KIRAG_MAX_PDF_FILE_BYTES",
            100 * 1024 * 1024,
            legacy_name="KIRAG_MAX_UPLOAD_BYTES",
        ),
        max_total_bytes=_positive_env(
            "KIRAG_MAX_PDF_UPLOAD_BYTES",
            500 * 1024 * 1024,
            legacy_name="KIRAG_MAX_UPLOAD_TOTAL_BYTES",
        ),
    )


def markdown_upload_limits() -> UploadLimits:
    return UploadLimits(
        max_files=_positive_env("KIRAG_MAX_MARKDOWN_FILES", 20),
        max_file_bytes=_positive_env(
            "KIRAG_MAX_MARKDOWN_FILE_BYTES",
            10 * 1024 * 1024,
            legacy_name="KIRAG_MAX_UPLOAD_BYTES",
        ),
        max_total_bytes=_positive_env(
            "KIRAG_MAX_MARKDOWN_UPLOAD_BYTES",
            50 * 1024 * 1024,
            legacy_name="KIRAG_MAX_UPLOAD_TOTAL_BYTES",
        ),
    )


def limits_for_path(path: str) -> UploadLimits | None:
    if path.rstrip("/").endswith("/api/pipeline/upload"):
        return pdf_upload_limits()
    if path.rstrip("/").endswith("/api/rag/upload-markdown"):
        return markdown_upload_limits()
    return None


def max_request_bytes(limits: UploadLimits) -> int:
    multipart_overhead = max(1024 * 1024, limits.max_files * 16 * 1024)
    return limits.max_total_bytes + multipart_overhead


class LimitedUploadRoute(APIRoute):
    """Apply the file-count cap while Starlette is parsing multipart data."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Any]]:
        original_handler = super().get_route_handler()

        async def limited_handler(request: Request):
            limits = limits_for_path(request.url.path)
            content_type = request.headers.get("content-type", "").lower()
            if limits and content_type.startswith("multipart/form-data"):
                try:
                    await request.form(
                        max_files=limits.max_files,
                        max_fields=10,
                        max_part_size=1024 * 1024,
                    )
                except StarletteHTTPException as exc:
                    if exc.status_code == 400 and "too many files" in str(exc.detail).lower():
                        raise HTTPException(
                            status_code=413, detail="Too many uploaded files"
                        ) from exc
                    raise
            return await original_handler(request)

        return limited_handler


class UploadRequestLimitMiddleware:
    """Bound raw upload request bytes before multipart parsing can spool them."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        limits = limits_for_path(str(scope.get("path", "")))
        if limits is None:
            await self.app(scope, receive, send)
            return

        request_limit = max_request_bytes(limits)
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > request_limit:
                    await error_response(
                        413,
                        "payload_too_large",
                        "The upload request exceeds the aggregate size limit",
                    )(scope, receive, send)
                    return
            except ValueError:
                await error_response(
                    400, "invalid_content_length", "Invalid Content-Length header"
                )(scope, receive, send)
                return

        received_bytes = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > request_limit:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except BaseException:
            if not exceeded:
                raise

        if exceeded:
            await error_response(
                413,
                "payload_too_large",
                "The upload request exceeds the aggregate size limit",
            )(scope, receive, send)


def escaped_original_name(upload: UploadFile, extension: str) -> str:
    """Validate an untrusted display name and return inert metadata."""

    raw_name = (upload.filename or "").strip()
    if (
        not raw_name
        or len(raw_name) > 255
        or "\x00" in raw_name
        or "/" in raw_name
        or "\\" in raw_name
        or Path(raw_name).is_absolute()
        or raw_name in {".", ".."}
    ):
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    if Path(raw_name).suffix.lower() != extension:
        raise HTTPException(status_code=415, detail=f"Only {extension} files are supported")
    return escape(raw_name, quote=True)


def unique_upload_name(extension: str) -> str:
    return f"{uuid4().hex}{extension}"


def validate_pdf_file(path: Path) -> None:
    """Require a structurally parseable PDF with at least one page."""

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=True)
        if len(reader.pages) < 1:
            raise ValueError("PDF contains no pages")
    except Exception as exc:
        raise HTTPException(status_code=415, detail="Upload is not a parseable PDF") from exc


def require_content_type(upload: UploadFile, allowed: set[str]) -> None:
    content_type = (getattr(upload, "content_type", None) or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported upload content type")


class MarkdownValidator:
    """Incrementally validate UTF-8 text and reject binary control bytes."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")

    @staticmethod
    def _reject_controls(text: str) -> None:
        if any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
            raise HTTPException(status_code=415, detail="Markdown upload is binary data")

    def feed(self, chunk: bytes) -> None:
        try:
            text = self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415, detail="Markdown upload must be valid UTF-8"
            ) from exc
        self._reject_controls(text)

    def finish(self) -> None:
        try:
            text = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415, detail="Markdown upload must be valid UTF-8"
            ) from exc
        self._reject_controls(text)


async def close_uploads(uploads: list[Any]) -> None:
    for upload in uploads:
        close = getattr(upload, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass
