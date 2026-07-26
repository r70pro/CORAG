#!/usr/bin/env python3
"""Capture PostgreSQL chunk rows and Qdrant points by run.

This is deliberately read-only. The resulting JSON can be compared before and
after indexing, migration, or repair work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.db import get_connection  # noqa: E402
from rag.embedding import get_qdrant_client  # noqa: E402


def _postgres_counts() -> list[dict]:
    query = """
        SELECT
            r.run_id,
            r.run_dir,
            r.status,
            COUNT(DISTINCT d.doc_id) AS document_count,
            COUNT(DISTINCT c.chunk_id) AS chunk_count,
            COUNT(DISTINCT c.qdrant_point_id)
                FILTER (WHERE c.qdrant_point_id IS NOT NULL) AS vector_reference_count
        FROM ocr_runs AS r
        LEFT JOIN documents AS d ON d.run_id = r.run_id
        LEFT JOIN chunks AS c ON c.run_id = r.run_id
        GROUP BY r.run_id, r.run_dir, r.status
        ORDER BY r.run_id
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    return [
        {
            "run_id": row[0],
            "run_dir": row[1],
            "status": row[2],
            "postgres_documents": row[3],
            "postgres_chunks": row[4],
            "postgres_vector_references": row[5],
        }
        for row in rows
    ]


def _qdrant_counts() -> tuple[dict[str, Counter], dict[str, int]]:
    client = get_qdrant_client()
    counts: dict[str, Counter] = {}
    totals: dict[str, int] = {}
    for collection in client.get_collections().collections:
        collection_name = collection.name
        run_counts: Counter = Counter()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["run_id"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                run_counts[str(payload.get("run_id") or "<missing-run-id>")] += 1
            if offset is None:
                break
        counts[collection_name] = run_counts
        totals[collection_name] = sum(run_counts.values())
    return counts, totals


def capture() -> dict:
    postgres_runs = _postgres_counts()
    qdrant_by_collection, qdrant_totals = _qdrant_counts()
    known_run_ids = {row["run_id"] for row in postgres_runs}

    for row in postgres_runs:
        per_collection = {
            collection: run_counts.get(row["run_id"], 0)
            for collection, run_counts in qdrant_by_collection.items()
        }
        row["qdrant_by_collection"] = per_collection
        row["qdrant_vectors"] = sum(per_collection.values())
        row["delta_qdrant_minus_postgres_chunks"] = row["qdrant_vectors"] - row["postgres_chunks"]

    qdrant_only_runs = {}
    for collection, run_counts in qdrant_by_collection.items():
        for run_id, count in run_counts.items():
            if run_id not in known_run_ids:
                qdrant_only_runs.setdefault(run_id, {})[collection] = count

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "postgres_total_runs": len(postgres_runs),
        "postgres_total_chunks": sum(row["postgres_chunks"] for row in postgres_runs),
        "postgres_total_vector_references": sum(
            row["postgres_vector_references"] for row in postgres_runs
        ),
        "qdrant_collection_totals": qdrant_totals,
        "qdrant_total_vectors": sum(qdrant_totals.values()),
        "runs": postgres_runs,
        "qdrant_only_runs": qdrant_only_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    args = parser.parse_args()
    result = capture()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
