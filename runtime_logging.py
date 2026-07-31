"""Consistent rotating application logs for unattended deployments."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process": record.process,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def configure_runtime_logging(component: str) -> None:
    """Configure one rotating JSON log without duplicating handlers."""
    if os.environ.get("TESTING") == "true":
        return
    root = logging.getLogger()
    marker = f"_kirag_{component}_logging"
    if getattr(root, marker, False):
        return

    log_dir = Path(os.environ.get("KIRAG_LOG_DIR", Path(__file__).resolve().parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / f"{component}.jsonl",
        maxBytes=int(os.environ.get("KIRAG_LOG_MAX_BYTES", 20 * 1024 * 1024)),
        backupCount=int(os.environ.get("KIRAG_LOG_BACKUP_COUNT", 5)),
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.environ.get("KIRAG_LOG_LEVEL", "INFO").upper())
    setattr(root, marker, True)
