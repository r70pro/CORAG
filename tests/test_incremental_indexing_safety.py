from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from indexing_service import CorpusIndexingService

FAILURE_STAGES = [
    "init_collection",
    "register_run",
    "snapshot",
    "register_document",
    "replace_chunks",
    "embed_before_upsert",
    "embed_after_upsert",
    "delete_stale",
    "file_finalisation",
    "mark_document",
    "get_totals",
    "mark_run",
    "commit",
]


def _install_operation_fakes(monkeypatch, tmp_path, failure_stage=None):
    import indexing_service
    from rag import cache, chunker, db, embedding, storage

    workspace = tmp_path / "workspace"
    run_dir = workspace / "run_existing"
    (run_dir / "markdown" / "inputs").mkdir(parents=True)
    upload = tmp_path / "additional.md"
    upload.write_text("Additional clinical evidence.", encoding="utf-8")

    postgres = {
        "run": {"status": "indexed", "total_documents": 1, "total_chunks": 1},
        "documents": {"old-document": {"status": "indexed"}},
        "chunks": {
            "old-chunk": {
                "doc_id": "old-document",
                "point_id": "old-vector",
            }
        },
    }
    qdrant = {
        "old-vector": {
            "vector": [0.1, 0.2],
            "payload": {"run_id": "existing-run", "chunk_id": "old-chunk"},
        }
    }
    active_transaction = {"state": None}
    observed_upsert_kwargs = []

    def fail(stage):
        if failure_stage == stage:
            raise RuntimeError(f"injected failure: {stage}")

    @contextmanager
    def transaction(_run_id):
        local = copy.deepcopy(postgres)
        active_transaction["state"] = local
        try:
            yield object()
            fail("commit")
            postgres.clear()
            postgres.update(local)
        finally:
            active_transaction["state"] = None

    def state():
        return active_transaction["state"]

    def register_run(*_args, **_kwargs):
        fail("register_run")
        state()["run"]["status"] = "pending"

    def mark_run_pending(*_args, **_kwargs):
        state()["run"]["status"] = "pending"

    def get_point_ids(doc_ids, **_kwargs):
        return {
            chunk["point_id"]
            for chunk in state()["chunks"].values()
            if chunk["doc_id"] in set(doc_ids)
        }

    def register_document(*, doc_id, **_kwargs):
        fail("register_document")
        state()["documents"][doc_id] = {"status": "pending"}

    def replace_chunks(doc_ids, chunks, **_kwargs):
        fail("replace_chunks")
        state()["chunks"] = {
            chunk_id: chunk
            for chunk_id, chunk in state()["chunks"].items()
            if chunk["doc_id"] not in set(doc_ids)
        }
        for chunk in chunks:
            state()["chunks"][chunk["chunk_id"]] = {
                "doc_id": chunk["doc_id"],
                "point_id": chunk["qdrant_point_id"],
            }

    def mark_document(doc_id, **_kwargs):
        fail("mark_document")
        state()["documents"][doc_id]["status"] = "indexed"

    def get_totals(*_args, **_kwargs):
        fail("get_totals")
        return len(state()["documents"]), len(state()["chunks"])

    def mark_run(*_args, total_documents, total_chunks, **_kwargs):
        fail("mark_run")
        state()["run"] = {
            "status": "indexed",
            "total_documents": total_documents,
            "total_chunks": total_chunks,
        }

    def prepare(chunks, model_name=None):
        for chunk in chunks:
            chunk["qdrant_point_id"] = "new-vector"
            chunk["embedding_model"] = "test-model"
        return "test-model"

    def init_collection(*_args, **_kwargs):
        fail("init_collection")

    def snapshot(point_ids, **_kwargs):
        fail("snapshot")
        return {
            point_id: copy.deepcopy(qdrant[point_id])
            for point_id in point_ids
            if point_id in qdrant
        }

    def upsert(chunks, **kwargs):
        observed_upsert_kwargs.append(kwargs)
        fail("embed_before_upsert")
        for chunk in chunks:
            qdrant[chunk["qdrant_point_id"]] = {
                "vector": [0.3, 0.4],
                "payload": {
                    "run_id": chunk["run_id"],
                    "chunk_id": chunk["chunk_id"],
                },
            }
        fail("embed_after_upsert")
        yield {"stage": "indexing", "current": len(chunks), "total": len(chunks)}

    def delete_points(point_ids, **_kwargs):
        fail("delete_stale")
        for point_id in point_ids:
            qdrant.pop(point_id, None)

    def rollback(touched_point_ids, snapshots, **_kwargs):
        for point_id in touched_point_ids:
            if point_id in snapshots:
                qdrant[point_id] = copy.deepcopy(snapshots[point_id])
            else:
                qdrant.pop(point_id, None)

    monkeypatch.setattr(indexing_service, "WORKSPACE_DIR", str(workspace))
    monkeypatch.setattr(
        indexing_service,
        "load_settings",
        lambda: {"chunk_size": 800, "chunk_overlap": 100},
    )
    monkeypatch.setattr(
        db,
        "get_runs_with_stats",
        lambda: [{"run_id": "existing-run", "run_dir": str(run_dir)}],
    )
    monkeypatch.setattr(db, "indexing_transaction", transaction)
    monkeypatch.setattr(db, "register_run", register_run)
    monkeypatch.setattr(db, "mark_run_pending", mark_run_pending)
    monkeypatch.setattr(db, "get_point_ids_for_documents", get_point_ids)
    monkeypatch.setattr(db, "register_document", register_document)
    monkeypatch.setattr(db, "replace_document_chunks", replace_chunks)
    monkeypatch.setattr(db, "mark_document_indexed", mark_document)
    monkeypatch.setattr(db, "get_run_totals", get_totals)
    monkeypatch.setattr(db, "mark_run_indexed", mark_run)
    monkeypatch.setattr(
        chunker,
        "chunk_document",
        lambda **_kwargs: [
            {
                "chunk_id": "new-chunk",
                "doc_id": "new-document",
                "run_id": "existing-run",
                "chunk_index": 0,
                "text": "Additional clinical evidence.",
                "char_start": 0,
                "char_end": 29,
            }
        ],
    )
    monkeypatch.setattr(embedding, "prepare_chunk_point_ids", prepare)
    monkeypatch.setattr(embedding, "init_collection", init_collection)
    monkeypatch.setattr(embedding, "snapshot_points", snapshot)
    monkeypatch.setattr(embedding, "upsert_chunks_generator", upsert)
    monkeypatch.setattr(embedding, "delete_points", delete_points)
    monkeypatch.setattr(embedding, "rollback_point_mutations", rollback)
    monkeypatch.setattr(storage, "upload_markdown", lambda *_args, **_kwargs: "key")
    monkeypatch.setattr(cache, "invalidate_query_cache", lambda: None)

    if failure_stage == "file_finalisation":
        real_replace = indexing_service.os.replace

        def fail_staged_replace(source, destination):
            if ".indexing-" in str(source):
                fail("file_finalisation")
            return real_replace(source, destination)

        monkeypatch.setattr(indexing_service.os, "replace", fail_staged_replace)

    return {
        "run_dir": run_dir,
        "upload": upload,
        "postgres": postgres,
        "qdrant": qdrant,
        "observed_upsert_kwargs": observed_upsert_kwargs,
    }


def test_adding_document_preserves_all_existing_vectors(tmp_path, monkeypatch):
    harness = _install_operation_fakes(monkeypatch, tmp_path)

    output = list(
        CorpusIndexingService.add_markdown_to_case(
            [SimpleNamespace(name=str(harness["upload"]))],
            case_option="existing-run",
            new_case_name="",
        )
    )

    assert any("Successfully uploaded" in line for line in output)
    assert set(harness["qdrant"]) == {"old-vector", "new-vector"}
    assert harness["postgres"]["run"] == {
        "status": "indexed",
        "total_documents": 2,
        "total_chunks": 2,
    }
    assert all(
        kwargs.get("pre_delete_run_ids") in (None, [])
        for kwargs in harness["observed_upsert_kwargs"]
    )


@pytest.mark.parametrize("failure_stage", FAILURE_STAGES)
def test_injected_failure_restores_pre_operation_state(
    tmp_path, monkeypatch, failure_stage
):
    harness = _install_operation_fakes(monkeypatch, tmp_path, failure_stage)
    postgres_before = copy.deepcopy(harness["postgres"])
    qdrant_before = copy.deepcopy(harness["qdrant"])

    output = list(
        CorpusIndexingService.add_markdown_to_case(
            [SimpleNamespace(name=str(harness["upload"]))],
            case_option="existing-run",
            new_case_name="",
        )
    )

    assert any("failed" in line.lower() for line in output)
    assert harness["postgres"] == postgres_before
    assert harness["qdrant"] == qdrant_before
    assert not (harness["run_dir"] / "markdown" / "inputs" / "additional.md").exists()
