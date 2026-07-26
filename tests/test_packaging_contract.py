from __future__ import annotations

import ast
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_runtime_imports_are_declared_or_standard_library():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        .replace("-", "_")
        for dependency in metadata["project"]["dependencies"]
    }
    local = set(metadata["tool"]["setuptools"]["py-modules"]) | set(
        metadata["tool"]["setuptools"]["packages"]
    )
    import_to_distribution = {
        "docx": "python_docx",
        "huggingface_hub": "huggingface_hub",
        "psycopg2": "psycopg2_binary",
        "qdrant_client": "qdrant_client",
        "sentence_transformers": "sentence_transformers",
    }

    missing: dict[str, set[str]] = {}
    sources = [
        *REPO_ROOT.glob("*.py"),
        *(REPO_ROOT / "api").rglob("*.py"),
        *(REPO_ROOT / "rag").rglob("*.py"),
    ]
    for source in sources:
        if source.name == "conftest.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                normalized = import_to_distribution.get(name, name.lower())
                if (
                    name not in sys.stdlib_module_names
                    and name not in local
                    and normalized not in declared
                ):
                    missing.setdefault(name, set()).add(str(source.relative_to(REPO_ROOT)))

    assert not missing


def test_setuptools_surface_and_assets_are_explicit():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = metadata["tool"]["setuptools"]

    assert {"api", "api.routes", "assets", "rag"} <= set(setuptools["packages"])
    assert "cli" in setuptools["py-modules"]
    assert "conftest" not in setuptools["py-modules"]
    assert metadata["tool"]["setuptools"]["package-data"]["assets"] == ["*.css", "*.js"]
    assert metadata["tool"]["setuptools"]["data-files"]["share/kirag"] == ["docker-compose.rag.yml"]


def test_requirements_txt_matches_project_runtime_metadata():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = {
        dependency.replace(" ", "") for dependency in metadata["project"]["dependencies"]
    }
    requirements_dependencies = {
        line.replace(" ", "")
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirements_dependencies == project_dependencies


def test_lock_files_are_pinned_and_hashed():
    cpu = (REPO_ROOT / "requirements-cpu.lock").read_text(encoding="utf-8")
    cuda = (REPO_ROOT / "requirements-cuda.lock").read_text(encoding="utf-8")

    assert "torch==2.13.0+cpu" in cpu
    assert "download.pytorch.org/whl/cpu" in cpu
    assert "nvidia-cusparselt" not in cpu
    assert "torch==2.13.0" in cuda
    assert "nvidia-cusparselt-cu13==0.8.1" in cuda
    assert "--hash=sha256:" in cpu
    assert "--hash=sha256:" in cuda
    assert ">=" not in "\n".join(
        line for line in cpu.splitlines() if line and not line.startswith(("#", " "))
    )
