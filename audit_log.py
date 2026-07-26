"""Durable, structured audit logging for security-sensitive operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any

_AUDIT_LOGGER_NAME = "kirag.audit"
_handler_lock = Lock()


def _audit_log_path() -> Path:
    configured = os.environ.get("KIRAG_AUDIT_LOG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / "logs" / "audit.jsonl"


def _get_audit_logger() -> logging.Logger:
    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    if getattr(audit_logger, "_kirag_audit_configured", False):
        return audit_logger

    with _handler_lock:
        if getattr(audit_logger, "_kirag_audit_configured", False):
            return audit_logger
        log_path = _audit_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(mode=0o600, exist_ok=True)
        try:
            log_path.chmod(0o600)
        except OSError:
            pass
        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        audit_logger._kirag_audit_configured = True  # type: ignore[attr-defined]
    return audit_logger


def audit_event(action: str, outcome: str, **details: Any) -> None:
    """Write one JSON object without accepting secrets as positional data."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "outcome": outcome,
        **details,
    }
    try:
        _get_audit_logger().info(
            json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        )
    except Exception:
        # Audit logging must not crash cleanup/shutdown paths. Emit a visible
        # application error while preserving the original operation's result.
        logging.getLogger(__name__).exception("Failed to write audit event")
