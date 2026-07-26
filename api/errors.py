"""Typed error helpers shared by API routes and exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from api.models import ErrorDetail, ErrorEnvelope


def error_envelope(
    code: str,
    message: str,
    details: Any | None = None,
) -> ErrorEnvelope:
    return ErrorEnvelope(error=ErrorDetail(code=code, message=message, details=details))


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    envelope = error_envelope(code, message, details)
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def default_error_code(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
        502: "upstream_error",
        503: "service_unavailable",
        504: "upstream_timeout",
    }.get(status_code, "request_failed")
