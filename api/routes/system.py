"""KIRAG application lifecycle endpoints."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import verify_admin_key
from api.models import AppShutdownRequest, MessageResponse
from audit_log import audit_event

logger = logging.getLogger(__name__)
router = APIRouter()
SHUTDOWN_REQUEST_PATH = Path(
    os.environ.get(
        "KIRAG_SHUTDOWN_REQUEST_PATH",
        f"{os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')}/kirag/app-shutdown-request",
    )
)


def require_app_shutdown_enabled() -> None:
    if os.environ.get("TESTING") == "true":
        return
    if os.environ.get("KIRAG_ENABLE_APP_SHUTDOWN") != "true":
        raise HTTPException(
            status_code=403, detail="App shutdown is disabled by KIRAG_ENABLE_APP_SHUTDOWN"
        )


@router.post(
    "/shutdown",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Gracefully stop KIRAG services and containers",
    dependencies=[Depends(verify_admin_key), Depends(require_app_shutdown_enabled)],
)
def shutdown_app(request: AppShutdownRequest):
    """Ask the user systemd trigger to stop the KIRAG stack."""
    del request
    try:
        descriptor = os.open(
            SHUTDOWN_REQUEST_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as marker:
            marker.write("shutdown\n")
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError:
        return MessageResponse(message="KIRAG shutdown is already in progress.")
    except OSError as exc:
        audit_event("host_shutdown_request", "failure", error=str(exc))
        if exc.errno in {errno.ENOENT, errno.EACCES, errno.EROFS}:
            raise HTTPException(
                status_code=503,
                detail="KIRAG shutdown trigger is unavailable; install the KIRAG desktop services",
            ) from exc
        logger.exception("Unable to create KIRAG shutdown request")
        raise HTTPException(status_code=500, detail="Unable to request KIRAG shutdown") from exc

    audit_event("host_shutdown_request", "success")
    return MessageResponse(
        message=(
            "Shutdown accepted. KIRAG services and containers are stopping and will remain "
            "off across host restarts until the desktop launcher is opened; the DGX will remain powered on."
        )
    )
