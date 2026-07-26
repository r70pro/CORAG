"""Reliability regressions for API validation, error envelopes, and RAG limits."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from api.models import (
    DockerCreateRequest,
    DockerModelsResponse,
    EmbeddingConfigRequest,
    RAGQueryRequest,
    SettingsUpdateRequest,
)


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("KIRAG_API_KEY", "test-api-key")
    monkeypatch.setenv("KIRAG_ADMIN_API_KEY", "test-admin-key")
    with TestClient(
        app,
        headers={
            "X-API-Key": "test-api-key",
            "X-Admin-API-Key": "test-admin-key",
        },
        raise_server_exceptions=False,
    ) as client:
        yield client


@pytest.mark.parametrize(
    "model",
    [
        lambda: SettingsUpdateRequest(workers=0),
        lambda: SettingsUpdateRequest(max_concurrent_requests=101),
        lambda: SettingsUpdateRequest(docker_gpu_mem=1.1),
        lambda: SettingsUpdateRequest(docker_max_model_len=1_048_577),
        lambda: SettingsUpdateRequest(retrieval_top_k=0),
        lambda: SettingsUpdateRequest(chunk_size=100, chunk_overlap=100),
        lambda: EmbeddingConfigRequest(chunk_size=100, chunk_overlap=100),
        lambda: EmbeddingConfigRequest(embedding_batch_size=0),
        lambda: DockerCreateRequest(max_model_len=1_048_577),
        lambda: DockerModelsResponse(models=["model"], max_lengths={"model": 0}),
        lambda: RAGQueryRequest(query="x" * 32_769),
        lambda: RAGQueryRequest(query="x", date_from="2026-02-30"),
        lambda: RAGQueryRequest(
            query="x", date_from="2026-02-02", date_to="2026-02-01"
        ),
    ],
)
def test_pydantic_rejects_out_of_bounds_inputs(model):
    with pytest.raises(ValidationError):
        model()


@pytest.mark.parametrize("tail", [0, 10_001])
def test_docker_log_tail_is_bounded(api_client, tail):
    response = api_client.get(f"/api/docker/logs?tail={tail}")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_validation_errors_use_typed_non_200_envelopes(api_client):
    response = api_client.post(
        "/api/rag/query",
        json={
            "query": "test",
            "stream": False,
            "date_from": "2026-07-20",
            "date_to": "2026-07-19",
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request validation failed"
    assert body["error"]["details"]


def test_rag_failure_is_not_an_http_200_error_dictionary(api_client):
    with patch("rag.analyzer.analyze", side_effect=RuntimeError("private failure")):
        response = api_client.post(
            "/api/rag/query",
            json={"query": "test", "stream": False},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "rag_query_failed",
            "message": "RAG query failed",
        }
    }
    assert "private failure" not in response.text


def test_empty_settings_update_is_a_typed_bad_request(api_client):
    response = api_client.put("/api/settings/", json={})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "bad_request",
            "message": "No fields provided to update",
        }
    }


def test_authentication_failure_uses_the_same_error_envelope(monkeypatch):
    monkeypatch.delenv("KIRAG_API_KEY", raising=False)
    monkeypatch.delenv("KIRAG_ADMIN_API_KEY", raising=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/settings/")

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key",
        }
    }
