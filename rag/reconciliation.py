"""Read-only PostgreSQL/Qdrant reconciliation for each indexing run."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from rag.db import get_connection
from rag.embedding import (
    _deterministic_point_id,
    get_collection_name,
    get_qdrant_client,
)


def _postgres_state():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, run_dir, status, total_documents, total_chunks
                FROM ocr_runs
                ORDER BY run_id
                """
            )
            runs = [
                {
                    "run_id": str(row[0]),
                    "run_dir": row[1],
                    "status": row[2],
                    "registered_document_count": int(row[3] or 0),
                    "registered_chunk_count": int(row[4] or 0),
                }
                for row in cursor.fetchall()
            ]

            cursor.execute(
                """
                SELECT doc_id, run_id, status, indexed_at
                FROM documents
                ORDER BY run_id, doc_id
                """
            )
            documents = [
                {
                    "doc_id": str(row[0]),
                    "run_id": str(row[1]),
                    "status": row[2],
                    "indexed_at": row[3],
                }
                for row in cursor.fetchall()
            ]

            cursor.execute(
                """
                SELECT chunk_id, doc_id, run_id, qdrant_point_id
                FROM chunks
                ORDER BY run_id, doc_id, chunk_id
                """
            )
            chunks = [
                {
                    "chunk_id": str(row[0]),
                    "doc_id": str(row[1]),
                    "run_id": str(row[2]),
                    "qdrant_point_id": str(row[3]) if row[3] is not None else None,
                }
                for row in cursor.fetchall()
            ]
    return runs, documents, chunks


def _qdrant_state(collection_name):
    client = get_qdrant_client()
    collection_names = {collection.name for collection in client.get_collections().collections}
    if collection_name not in collection_names:
        return []
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=["chunk_id", "doc_id", "run_id"],
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            points.append(
                {
                    "point_id": str(point.id),
                    "chunk_id": (
                        str(payload["chunk_id"]) if payload.get("chunk_id") is not None else None
                    ),
                    "doc_id": (
                        str(payload["doc_id"]) if payload.get("doc_id") is not None else None
                    ),
                    "run_id": (
                        str(payload["run_id"])
                        if payload.get("run_id") is not None
                        else "<missing-run-id>"
                    ),
                }
            )
        if offset is None:
            break
    return points


def reconcile(collection_name=None, run_id=None):
    """Return reconciliation findings, grouped by run.

    The command is deliberately read-only. It compares PostgreSQL against the
    active embedding collection unless an explicit collection is supplied.
    """
    collection_name = collection_name or get_collection_name()
    runs, documents, chunks = _postgres_state()
    qdrant_points = _qdrant_state(collection_name)

    pg_chunks_by_run = defaultdict(list)
    pg_chunks_by_doc = defaultdict(list)
    pg_chunks_by_point = defaultdict(list)
    pg_chunks_by_id = {}
    for chunk in chunks:
        pg_chunks_by_run[chunk["run_id"]].append(chunk)
        pg_chunks_by_doc[chunk["doc_id"]].append(chunk)
        pg_chunks_by_id[chunk["chunk_id"]] = chunk
        if chunk["qdrant_point_id"]:
            pg_chunks_by_point[chunk["qdrant_point_id"]].append(chunk)

    docs_by_run = defaultdict(list)
    for document in documents:
        docs_by_run[document["run_id"]].append(document)

    qdrant_by_run = defaultdict(list)
    qdrant_by_point = {}
    qdrant_by_chunk = defaultdict(list)
    for point in qdrant_points:
        qdrant_by_run[point["run_id"]].append(point)
        qdrant_by_point[point["point_id"]] = point
        if point["chunk_id"]:
            qdrant_by_chunk[point["chunk_id"]].append(point)

    run_metadata = {item["run_id"]: item for item in runs}
    all_run_ids = set(run_metadata) | set(qdrant_by_run)
    if run_id is not None:
        all_run_ids &= {str(run_id)}

    reports = []
    for current_run_id in sorted(all_run_ids):
        run_chunks = pg_chunks_by_run[current_run_id]
        run_points = qdrant_by_run[current_run_id]

        postgres_without_qdrant = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "qdrant_point_id": chunk["qdrant_point_id"],
            }
            for chunk in run_chunks
            if not chunk["qdrant_point_id"] or chunk["qdrant_point_id"] not in qdrant_by_point
        ]

        qdrant_without_postgres = [
            {
                "point_id": point["point_id"],
                "chunk_id": point["chunk_id"],
                "doc_id": point["doc_id"],
            }
            for point in run_points
            if point["point_id"] not in pg_chunks_by_point
        ]

        duplicate_or_stale = []
        for point_id, matching_chunks in pg_chunks_by_point.items():
            if len(matching_chunks) > 1 and any(
                chunk["run_id"] == current_run_id for chunk in matching_chunks
            ):
                duplicate_or_stale.append(
                    {
                        "point_id": point_id,
                        "reason": "duplicate PostgreSQL point reference",
                        "chunk_ids": [chunk["chunk_id"] for chunk in matching_chunks],
                    }
                )

        for chunk in run_chunks:
            expected = _deterministic_point_id(chunk["chunk_id"])
            if chunk["qdrant_point_id"] != expected:
                duplicate_or_stale.append(
                    {
                        "point_id": chunk["qdrant_point_id"],
                        "reason": "PostgreSQL point ID is not deterministic for chunk",
                        "chunk_id": chunk["chunk_id"],
                        "expected_point_id": expected,
                    }
                )

        for chunk_id, matching_points in qdrant_by_chunk.items():
            matching_in_run = [
                point for point in matching_points if point["run_id"] == current_run_id
            ]
            if len(matching_in_run) > 1:
                duplicate_or_stale.append(
                    {
                        "point_ids": [point["point_id"] for point in matching_in_run],
                        "reason": "duplicate Qdrant points for chunk",
                        "chunk_id": chunk_id,
                    }
                )

        for point in run_points:
            if point["chunk_id"]:
                expected = _deterministic_point_id(point["chunk_id"])
                if point["point_id"] != expected:
                    duplicate_or_stale.append(
                        {
                            "point_id": point["point_id"],
                            "reason": "Qdrant point ID is stale or non-deterministic",
                            "chunk_id": point["chunk_id"],
                            "expected_point_id": expected,
                        }
                    )
            pg_chunk = pg_chunks_by_id.get(point["chunk_id"])
            if pg_chunk and (
                pg_chunk["qdrant_point_id"] != point["point_id"]
                or pg_chunk["doc_id"] != point["doc_id"]
                or pg_chunk["run_id"] != point["run_id"]
            ):
                duplicate_or_stale.append(
                    {
                        "point_id": point["point_id"],
                        "reason": "Qdrant payload disagrees with PostgreSQL",
                        "chunk_id": point["chunk_id"],
                    }
                )

        incomplete_documents = []
        for document in docs_by_run[current_run_id]:
            is_indexed = document["status"] == "indexed" or document["indexed_at"] is not None
            if not is_indexed:
                continue
            document_chunks = pg_chunks_by_doc[document["doc_id"]]
            missing = [
                chunk["qdrant_point_id"]
                for chunk in document_chunks
                if not chunk["qdrant_point_id"]
                or chunk["qdrant_point_id"] not in qdrant_by_point
                or qdrant_by_point[chunk["qdrant_point_id"]]["chunk_id"] != chunk["chunk_id"]
                or qdrant_by_point[chunk["qdrant_point_id"]]["doc_id"] != chunk["doc_id"]
                or qdrant_by_point[chunk["qdrant_point_id"]]["run_id"] != chunk["run_id"]
            ]
            if not document_chunks or missing:
                incomplete_documents.append(
                    {
                        "doc_id": document["doc_id"],
                        "postgres_chunks": len(document_chunks),
                        "missing_point_ids": missing,
                    }
                )

        metadata = run_metadata.get(current_run_id, {})
        reports.append(
            {
                "run_id": current_run_id,
                "run_dir": metadata.get("run_dir"),
                "status": metadata.get("status", "qdrant-only"),
                "postgres_chunks": len(run_chunks),
                "qdrant_points": len(run_points),
                "postgres_chunks_without_qdrant": postgres_without_qdrant,
                "qdrant_points_without_postgres": qdrant_without_postgres,
                "duplicate_or_stale_point_ids": duplicate_or_stale,
                "indexed_documents_with_incomplete_vectors": incomplete_documents,
                "healthy": not (
                    postgres_without_qdrant
                    or qdrant_without_postgres
                    or duplicate_or_stale
                    or incomplete_documents
                ),
            }
        )

    issue_counts = Counter(
        {
            "postgres_chunks_without_qdrant": 0,
            "qdrant_points_without_postgres": 0,
            "duplicate_or_stale_point_ids": 0,
            "indexed_documents_with_incomplete_vectors": 0,
        }
    )
    for report in reports:
        issue_counts["postgres_chunks_without_qdrant"] += len(
            report["postgres_chunks_without_qdrant"]
        )
        issue_counts["qdrant_points_without_postgres"] += len(
            report["qdrant_points_without_postgres"]
        )
        issue_counts["duplicate_or_stale_point_ids"] += len(report["duplicate_or_stale_point_ids"])
        issue_counts["indexed_documents_with_incomplete_vectors"] += len(
            report["indexed_documents_with_incomplete_vectors"]
        )

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection_name,
        "healthy": not any(issue_counts.values()),
        "issue_counts": dict(issue_counts),
        "runs": reports,
    }
