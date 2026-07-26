"""Regression contract for additive indexing into an existing case."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from indexing_service import CorpusIndexingService

pytestmark = pytest.mark.phase0_regression


class _Cursor:
    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Connection:
    def cursor(self):
        return _Cursor()


@contextmanager
def _connection():
    yield _Connection()


def test_adding_document_preserves_all_existing_case_vectors(tmp_path, monkeypatch):
    import indexing_service
    from rag import cache, chunker, db, embedding, storage

    run_dir = tmp_path / "workspace" / "run_existing"
    (run_dir / "markdown" / "inputs").mkdir(parents=True)
    uploaded = tmp_path / "additional.md"
    uploaded.write_text("Additional clinical evidence.", encoding="utf-8")

    existing_vector_ids = {"existing-vector-a", "existing-vector-b"}
    observed_pre_deletes = []

    def fake_upsert(chunks, batch_size, model_name=None, pre_delete_run_ids=None):
        observed_pre_deletes.append(pre_delete_run_ids)
        if pre_delete_run_ids:
            existing_vector_ids.clear()
        existing_vector_ids.update(chunk["qdrant_point_id"] for chunk in chunks)
        yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}

    new_chunk = {
        "chunk_id": "new-chunk",
        "doc_id": "new-document",
        "run_id": "existing-run",
        "qdrant_point_id": "new-vector",
        "text": "Additional clinical evidence.",
    }

    monkeypatch.setattr(
        db,
        "get_runs_with_stats",
        lambda: [{"run_id": "existing-run", "run_dir": str(run_dir)}],
    )
    monkeypatch.setattr(indexing_service, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(db, "indexing_transaction", lambda *_args, **_kwargs: _connection())
    monkeypatch.setattr(db, "register_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "mark_run_pending", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "get_point_ids_for_documents", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(db, "register_document", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "replace_document_chunks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "get_run_totals", lambda *_args, **_kwargs: (2, 3))
    monkeypatch.setattr(db, "mark_document_indexed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "mark_run_indexed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(storage, "upload_markdown", lambda *_args, **_kwargs: "object-key")
    monkeypatch.setattr(chunker, "chunk_document", lambda **_kwargs: [new_chunk])
    monkeypatch.setattr(
        embedding,
        "prepare_chunk_point_ids",
        lambda chunks, model_name=None: "test-model",
    )
    monkeypatch.setattr(embedding, "init_collection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding, "snapshot_points", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(embedding, "delete_points", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding, "rollback_point_mutations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embedding, "upsert_chunks_generator", fake_upsert)
    monkeypatch.setattr(cache, "invalidate_query_cache", lambda: None)
    monkeypatch.setattr(
        indexing_service,
        "load_settings",
        lambda: {"chunk_size": 800, "chunk_overlap": 100},
    )

    output = list(
        CorpusIndexingService.add_markdown_to_case(
            [SimpleNamespace(name=str(uploaded))],
            case_option="existing-run",
            new_case_name="",
        )
    )

    assert any("Successfully uploaded" in line for line in output)
    assert observed_pre_deletes == [None]
    assert existing_vector_ids == {
        "existing-vector-a",
        "existing-vector-b",
        "new-vector",
    }
