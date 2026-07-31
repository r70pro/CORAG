import httpx
import pytest

from rag import upstream


def setup_function():
    upstream._failure_count = 0
    upstream._open_until = 0.0


def test_request_retries_transient_connection_failure(monkeypatch):
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setenv("KIRAG_LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(upstream.time, "sleep", lambda _delay: None)
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("temporarily unavailable")
        return object()

    assert upstream.request_with_retry(operation) is not None
    assert calls == 3
    assert upstream._failure_count == 0


def test_circuit_opens_after_bounded_failures(monkeypatch):
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setenv("KIRAG_LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("KIRAG_LLM_CIRCUIT_THRESHOLD", "1")

    with pytest.raises(httpx.ConnectError):
        upstream.request_with_retry(lambda: (_ for _ in ()).throw(httpx.ConnectError("down")))
    with pytest.raises(upstream.CircuitOpenError):
        upstream.request_with_retry(lambda: object())
