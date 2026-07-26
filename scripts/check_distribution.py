#!/usr/bin/env python3
"""Validate that KIRAG's wheel and sdist contain the deliberate runtime surface."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

PACKAGES = {
    "api/__init__.py",
    "api/main.py",
    "api/routes/__init__.py",
    "api/routes/rag.py",
    "assets/__init__.py",
    "assets/accessibility.js",
    "assets/theme.css",
    "rag/__init__.py",
    "rag/analyzer.py",
}
TOP_LEVEL_MODULES = {
    "app.py",
    "cli.py",
    "docker_manager.py",
    "indexing_service.py",
    "pipeline_manager.py",
    "rag_infra_manager.py",
    "settings_manager.py",
    "system_diagnostics.py",
}


def _assert_expected(names: set[str], *, archive: Path, sdist: bool) -> None:
    def present(expected: str) -> bool:
        if sdist:
            return any(name.endswith(f"/{expected}") for name in names)
        return expected in names

    missing = sorted(item for item in PACKAGES | TOP_LEVEL_MODULES if not present(item))
    if missing:
        raise SystemExit(f"{archive.name} is missing runtime files: {', '.join(missing)}")

    if sdist:
        compose_present = any(name.endswith("/docker-compose.rag.yml") for name in names)
    else:
        compose_present = any(
            name.endswith(".data/data/share/kirag/docker-compose.rag.yml") for name in names
        )
    if not compose_present:
        raise SystemExit(f"{archive.name} is missing the bundled Compose file")

    forbidden = {"conftest.py", "tests/test_packaging_contract.py"}
    leaked = sorted(item for item in forbidden if present(item))
    if leaked:
        raise SystemExit(f"{archive.name} contains test-only files: {', '.join(leaked)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        _assert_expected(names, archive=args.wheel, sdist=False)

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        for required in ("olmocr", "torch", "psutil", "huggingface-hub"):
            if f"Requires-Dist: {required}" not in metadata:
                raise SystemExit(f"{args.wheel.name} metadata is missing {required}")
        for development_only in ("pytest", "coverage", "ruff"):
            if f"Requires-Dist: {development_only}" in metadata.split("Provides-Extra: dev", 1)[0]:
                raise SystemExit(
                    f"{args.wheel.name} exposes {development_only} as a production dependency"
                )

        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode()
        if "kirag = cli:main" not in entry_points:
            raise SystemExit(f"{args.wheel.name} is missing the kirag console script")
    with tarfile.open(args.sdist, "r:gz") as archive:
        _assert_expected(set(archive.getnames()), archive=args.sdist, sdist=True)

    print(f"distribution contents OK: {args.wheel.name}, {args.sdist.name}")


if __name__ == "__main__":
    main()
