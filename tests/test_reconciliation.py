from rag import reconciliation
from rag.embedding import _deterministic_point_id


def test_reconciliation_reports_each_drift_category(monkeypatch):
    point_1 = _deterministic_point_id("chunk-1")
    point_2 = _deterministic_point_id("chunk-2")
    duplicate_point = _deterministic_point_id("duplicate-copy")
    orphan_point = _deterministic_point_id("orphan")

    monkeypatch.setattr(
        reconciliation,
        "_postgres_state",
        lambda: (
            [
                {
                    "run_id": "run-1",
                    "run_dir": "/workspace/run-1",
                    "status": "indexed",
                    "registered_document_count": 2,
                    "registered_chunk_count": 2,
                }
            ],
            [
                {
                    "doc_id": "doc-1",
                    "run_id": "run-1",
                    "status": "indexed",
                    "indexed_at": "now",
                },
                {
                    "doc_id": "doc-2",
                    "run_id": "run-1",
                    "status": "indexed",
                    "indexed_at": "now",
                },
            ],
            [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "run_id": "run-1",
                    "qdrant_point_id": point_1,
                },
                {
                    "chunk_id": "chunk-2",
                    "doc_id": "doc-2",
                    "run_id": "run-1",
                    "qdrant_point_id": point_2,
                },
            ],
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_qdrant_state",
        lambda _collection: [
            {
                "point_id": point_1,
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "run_id": "run-1",
            },
            {
                "point_id": duplicate_point,
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "run_id": "run-1",
            },
            {
                "point_id": orphan_point,
                "chunk_id": "orphan",
                "doc_id": "missing-doc",
                "run_id": "run-1",
            },
        ],
    )

    result = reconciliation.reconcile(collection_name="test-collection")
    report = result["runs"][0]

    assert result["healthy"] is False
    assert [item["chunk_id"] for item in report["postgres_chunks_without_qdrant"]] == [
        "chunk-2"
    ]
    assert {item["point_id"] for item in report["qdrant_points_without_postgres"]} == {
        duplicate_point,
        orphan_point,
    }
    assert report["duplicate_or_stale_point_ids"]
    assert report["indexed_documents_with_incomplete_vectors"] == [
        {
            "doc_id": "doc-2",
            "postgres_chunks": 1,
            "missing_point_ids": [point_2],
        }
    ]


def test_reconciliation_reports_qdrant_only_run(monkeypatch):
    monkeypatch.setattr(reconciliation, "_postgres_state", lambda: ([], [], []))
    monkeypatch.setattr(
        reconciliation,
        "_qdrant_state",
        lambda _collection: [
            {
                "point_id": "orphan-point",
                "chunk_id": "orphan-chunk",
                "doc_id": "orphan-doc",
                "run_id": "orphan-run",
            }
        ],
    )

    result = reconciliation.reconcile(collection_name="test-collection")

    assert result["runs"][0]["run_id"] == "orphan-run"
    assert result["runs"][0]["status"] == "qdrant-only"
    assert result["runs"][0]["qdrant_points_without_postgres"]
