#!/usr/bin/env python3
"""Fetch and verify a complete model snapshot before production startup."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision", required=True, help="Immutable commit SHA")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--offline-check", action="store_true")
    args = parser.parse_args()

    revision = args.revision.strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        parser.error("--revision must be a 40-character immutable Git commit SHA")

    # ``--cache-dir`` is KIRAG_HF_HOME, matching the directory mounted at
    # /root/.cache/huggingface in vLLM. Hugging Face stores repositories in
    # the ``hub`` child of HF_HOME.
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    hub_cache_dir = cache_dir / "hub"
    hub_cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=revision,
            cache_dir=str(hub_cache_dir),
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=args.offline_check,
            max_workers=4,
        )
    ).resolve()

    # snapshot is <repo-cache>/snapshots/<commit>; unrelated model downloads
    # must not prevent this verified model from starting.
    repo_cache = snapshot.parents[1]
    incomplete = list(repo_cache.rglob("*.incomplete"))
    weight_files = [
        *snapshot.glob("*.safetensors"),
        *snapshot.glob("*.bin"),
        *snapshot.glob("*.gguf"),
    ]
    required_metadata = [snapshot / "config.json"]
    if incomplete or not weight_files or not all(path.is_file() for path in required_metadata):
        raise RuntimeError(
            f"Model verification failed: incomplete={len(incomplete)}, "
            f"weights={len(weight_files)}, config={required_metadata[0].is_file()}"
        )

    manifest_dir = cache_dir / "kirag-manifests"
    manifest_dir.mkdir(exist_ok=True)
    manifest_name = args.model.replace("/", "--") + ".json"
    manifest = {
        "model": args.model,
        "revision": revision,
        "snapshot": str(snapshot),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "weight_files": len(weight_files),
        "offline_verified": bool(args.offline_check),
    }
    (manifest_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
