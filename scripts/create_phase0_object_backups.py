#!/usr/bin/env python3
"""Create logical Qdrant and MinIO backups for the Phase 0 baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.embedding import get_qdrant_client, get_qdrant_config  # noqa: E402
from rag.storage import get_client as get_minio_client  # noqa: E402


def _safe_object_path(root: Path, bucket: str, object_name: str) -> Path:
    parts = PurePosixPath(object_name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe MinIO object name: {object_name!r}")
    destination = root / bucket
    for part in parts:
        destination /= part
    return destination


def _backup_qdrant(output: Path) -> dict:
    client = get_qdrant_client()
    snapshot = client.create_full_snapshot(wait=True)
    if snapshot is None:
        raise RuntimeError("Qdrant did not return a full snapshot description")

    source = REPO_ROOT / "workspace" / "qdrant_storage" / "snapshots" / snapshot.name
    destination = output / "qdrant" / snapshot.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, destination)
        source_description = str(source.relative_to(REPO_ROOT))
    else:
        config = get_qdrant_config()
        snapshot_url = f"http://{config['host']}:{config['port']}/snapshots/{snapshot.name}"
        with urlopen(snapshot_url) as response, destination.open("wb") as target:
            shutil.copyfileobj(response, target)
        source_description = snapshot_url
    return {
        "snapshot_name": snapshot.name,
        "source": source_description,
        "backup_path": str(destination),
        "size_bytes": destination.stat().st_size,
    }


def _backup_minio(output: Path) -> dict:
    client = get_minio_client()
    objects = []
    buckets = client.list_buckets()
    for bucket in buckets:
        for item in client.list_objects(bucket.name, recursive=True):
            destination = _safe_object_path(output / "minio", bucket.name, item.object_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(bucket.name, item.object_name, str(destination))
            objects.append(
                {
                    "bucket": bucket.name,
                    "object_name": item.object_name,
                    "etag": item.etag,
                    "size_bytes": item.size,
                    "backup_path": str(destination),
                }
            )
    return {
        "bucket_count": len(buckets),
        "object_count": len(objects),
        "total_bytes": sum(item["size_bytes"] for item in objects),
        "objects": objects,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "qdrant": _backup_qdrant(args.output),
        "minio": _backup_minio(args.output),
    }
    manifest_path = args.output / "object-backup-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
