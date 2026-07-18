"""
Document and run browsing API routes.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from settings_manager import WORKSPACE_DIR

router = APIRouter()


@router.get("/runs", summary="List completed runs")
def list_runs():
    """Return all completed OCR runs with their markdown file listings."""
    from settings_manager import get_available_runs

    runs = get_available_runs()
    result = []
    for display_name, run_dir in runs:
        md_dir = os.path.join(run_dir, "markdown", "inputs")
        files = []
        if os.path.exists(md_dir):
            files = sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))
        result.append(
            {
                "display_name": display_name,
                "run_dir": run_dir,
                "run_name": os.path.basename(run_dir),
                "file_count": len(files),
                "files": files,
            }
        )
    return result


@router.get("/runs/{run_name}/files", summary="List files in a run")
def list_run_files(run_name: str):
    """Return all markdown files in a specific run."""
    run_dir = os.path.join(WORKSPACE_DIR, run_name)
    if not os.path.isdir(run_dir) or ".." in run_name:
        raise HTTPException(status_code=404, detail="Run not found")

    md_dir = os.path.join(run_dir, "markdown", "inputs")
    if not os.path.exists(md_dir):
        return []
    return sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))


@router.get("/runs/{run_name}/markdown/{filename}", summary="Get markdown content")
def get_markdown(run_name: str, filename: str):
    """Return the markdown content of a specific file in a run."""
    if ".." in run_name or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    file_path = os.path.join(WORKSPACE_DIR, run_name, "markdown", "inputs", filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)
