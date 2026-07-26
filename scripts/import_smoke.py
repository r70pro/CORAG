#!/usr/bin/env python3
"""Import every installed KIRAG runtime module from outside its checkout."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path

MODULES = [
    "api",
    "api.auth",
    "api.errors",
    "api.main",
    "api.models",
    "api.routes.diagnostics",
    "api.routes.docker",
    "api.routes.documents",
    "api.routes.pipeline",
    "api.routes.rag",
    "api.routes.settings",
    "api.server",
    "api.upload_security",
    "assets",
    "rag",
    "rag.analyzer",
    "rag.cache",
    "rag.chunker",
    "rag.db",
    "rag.embedding",
    "rag.metadata_helper",
    "rag.reconciliation",
    "rag.retriever",
    "rag.storage",
    "app",
    "app_handlers",
    "audit_log",
    "cleanup_manager",
    "cli",
    "docker_manager",
    "download_models",
    "embedding_pipeline_ui",
    "gradio_security",
    "html_utils",
    "indexing_service",
    "path_security",
    "pdf_manager",
    "pipeline_manager",
    "process_state",
    "rag_export",
    "rag_infra_manager",
    "rag_ui",
    "rag_ui_dashboard",
    "rag_ui_handlers",
    "rag_ui_state",
    "secrets_config",
    "settings_manager",
    "system_diagnostics",
    "ui_adapters",
    "ui_theme",
]


def main() -> None:
    checkout = Path(os.environ["KIRAG_CHECKOUT"]).resolve()
    for name in MODULES:
        module = importlib.import_module(name)
        location = getattr(module, "__file__", None)
        if location and Path(location).resolve().is_relative_to(checkout):
            raise RuntimeError(f"{name} leaked from the repository checkout: {location}")

    files = importlib.metadata.files("kirag") or ()
    installed = {item.as_posix() for item in files}
    required_suffixes = {
        "assets/accessibility.js",
        "assets/theme.css",
        "share/kirag/docker-compose.rag.yml",
    }
    for suffix in required_suffixes:
        if not any(item.endswith(suffix) for item in installed):
            raise RuntimeError(f"installed distribution is missing {suffix}")

    print(f"import smoke OK: {len(MODULES)} modules")


if __name__ == "__main__":
    main()
