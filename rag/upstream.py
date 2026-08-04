"""Bounded retry and circuit-breaker policy for the local LLM upstream."""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")
_lock = threading.Lock()
_failure_count = 0
_open_until = 0.0


class CircuitOpenError(RuntimeError):
    pass


def _attempts() -> int:
    if os.environ.get("TESTING") == "true":
        return 1
    return max(1, min(int(os.environ.get("KIRAG_LLM_MAX_ATTEMPTS", "3")), 5))


def _before_request() -> None:
    with _lock:
        if time.monotonic() < _open_until:
            raise CircuitOpenError("LLM circuit breaker is temporarily open")


def _record_success() -> None:
    global _failure_count, _open_until
    with _lock:
        _failure_count = 0
        _open_until = 0.0


def _record_failure() -> None:
    global _failure_count, _open_until
    with _lock:
        _failure_count += 1
        threshold = max(1, int(os.environ.get("KIRAG_LLM_CIRCUIT_THRESHOLD", "5")))
        if _failure_count >= threshold:
            _open_until = time.monotonic() + max(
                1.0, float(os.environ.get("KIRAG_LLM_CIRCUIT_RECOVERY_SECONDS", "30"))
            )


def request_with_retry(operation: Callable[[], T], *, retry_read_timeout: bool = True) -> T:
    """Retry connection/timeout and transient HTTP failures before opening the circuit."""
    last_error: Exception | None = None
    for attempt in range(_attempts()):
        _before_request()
        try:
            result = operation()
            status = getattr(result, "status_code", 200)
            if status in {429, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"transient upstream HTTP {status}", request=result.request, response=result
                )
            _record_success()
            return result
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            _record_failure()
            # A read timeout after a long generation may already have consumed
            # minutes of GPU work. Replaying it cannot resume the answer and can
            # multiply latency without improving reliability.
            if isinstance(exc, httpx.ReadTimeout) and not retry_read_timeout:
                raise
            if attempt + 1 >= _attempts():
                raise
            delay = min(0.5 * (2**attempt), 4.0) + random.uniform(0, 0.2)
            time.sleep(delay)
    assert last_error is not None
    raise last_error
