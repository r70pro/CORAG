"""Fail-closed API and administrative authentication helpers."""

from __future__ import annotations

import hmac
import ipaddress
import os
import sys
from collections.abc import Sequence

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def _matches(provided: str | None, expected: str) -> bool:
    return bool(provided and expected and hmac.compare_digest(provided, expected))


def authentication_configured() -> bool:
    """Return whether at least one API authentication secret is configured."""

    return bool(
        os.environ.get("KIRAG_API_KEY", "").strip()
        or os.environ.get("KIRAG_ADMIN_API_KEY", "").strip()
    )


def verify_api_key(
    key_from_header: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str:
    """Require a configured API/admin key and compare it in constant time."""

    expected_api_key = os.environ.get("KIRAG_API_KEY", "").strip()
    expected_admin_key = os.environ.get("KIRAG_ADMIN_API_KEY", "").strip()
    provided_key = key_from_header or (bearer.credentials if bearer else None)
    api_key_matches = _matches(provided_key, expected_api_key)
    admin_key_matches = _matches(provided_key, expected_admin_key)
    if not (api_key_matches | admin_key_matches):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided_key or ""


def verify_admin_key(
    admin_from_header: str | None = Security(admin_key_header),
    key_from_header: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str:
    """Require the dedicated administrative key for destructive operations."""

    expected_admin_key = os.environ.get("KIRAG_ADMIN_API_KEY", "").strip()
    provided_key = admin_from_header or key_from_header or (bearer.credentials if bearer else None)
    if not expected_admin_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative authorization is not configured",
        )
    if not _matches(provided_key, expected_admin_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative authorization required",
        )
    return provided_key or ""


def has_admin_access(
    admin_from_header: str | None = Security(admin_key_header),
    key_from_header: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> bool:
    """Return admin status without allowing a client-supplied role assertion."""
    expected = os.environ.get("KIRAG_ADMIN_API_KEY", "").strip()
    provided = admin_from_header or key_from_header or (bearer.credentials if bearer else None)
    return _matches(provided, expected)


def require_remote_lifecycle_enabled() -> None:
    """Keep host/container lifecycle control off the normal API surface."""
    if os.environ.get("TESTING") == "true":
        return
    if os.environ.get("KIRAG_ENABLE_REMOTE_LIFECYCLE") != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Remote lifecycle operations are disabled; use systemd on the host",
        )


def require_safe_bind(host: str, *, authenticated: bool | None = None) -> str:
    """Reject non-loopback server binding unless authentication is configured."""

    candidate = host.strip()
    try:
        is_loopback = ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        is_loopback = candidate.lower() == "localhost"
    has_auth = authentication_configured() if authenticated is None else authenticated
    if not is_loopback and not has_auth:
        raise RuntimeError("Non-loopback binding requires authentication")
    return candidate


def requested_api_bind_host(argv: Sequence[str] | None = None) -> str:
    """Resolve the API host from explicit configuration or Uvicorn CLI arguments."""

    configured = os.environ.get("KIRAG_API_HOST", "").strip()
    if configured:
        return configured

    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, argument in enumerate(arguments):
        if argument == "--host" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--host="):
            return argument.split("=", 1)[1]
    return "127.0.0.1"
