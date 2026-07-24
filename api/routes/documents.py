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


@router.get("/runs/{run_name}/pdf", summary="Get source PDF file")
def get_run_pdf(run_name: str):
    """Return the source PDF file of a specific run if available."""
    from fastapi.responses import FileResponse

    if ".." in run_name:
        raise HTTPException(status_code=400, detail="Invalid run name")

    inputs_dir = os.path.join(WORKSPACE_DIR, run_name, "inputs")
    if not os.path.exists(inputs_dir):
        # Fallback check for Downloads or WORKSPACE_DIR
        fallback = os.path.join("/home/owner/Downloads", "Docling_test_file.pdf")
        if os.path.isfile(fallback):
            return FileResponse(fallback, media_type="application/pdf")
        raise HTTPException(status_code=404, detail="Inputs directory not found")

    pdf_files = [f for f in os.listdir(inputs_dir) if f.endswith(".pdf")]
    if not pdf_files:
        fallback = os.path.join("/home/owner/Downloads", "Docling_test_file.pdf")
        if os.path.isfile(fallback):
            return FileResponse(fallback, media_type="application/pdf")
        raise HTTPException(status_code=404, detail="No PDF found in run inputs")

    pdf_path = os.path.join(inputs_dir, pdf_files[0])
    return FileResponse(pdf_path, media_type="application/pdf")


@router.get("/runs/{run_name}/info", summary="Get detailed page mapping and info for a document")
def get_run_doc_info(run_name: str, filename: str = ""):
    """Return document page count, character page ranges, and per-page markdown content."""
    import json
    import logging

    from pypdf import PdfReader

    logger = logging.getLogger(__name__)

    if ".." in run_name or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    run_dir = os.path.join(WORKSPACE_DIR, run_name)
    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    inputs_dir = os.path.join(run_dir, "inputs")
    pdf_path = None
    if os.path.exists(inputs_dir):
        if filename:
            pdf_name = filename.rsplit(".", 1)[0] + ".pdf"
            candidate = os.path.join(inputs_dir, pdf_name)
            if os.path.isfile(candidate):
                pdf_path = candidate
        if not pdf_path:
            pdf_files = [f for f in os.listdir(inputs_dir) if f.endswith(".pdf")]
            if pdf_files:
                pdf_path = os.path.join(inputs_dir, pdf_files[0])

    if not pdf_path or not os.path.isfile(pdf_path):
        fallback = os.path.join("/home/owner/Downloads", "Docling_test_file.pdf")
        if os.path.isfile(fallback):
            pdf_path = fallback

    total_pages = 1
    if pdf_path and os.path.isfile(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
        except Exception as e:
            logger.error(f"Error reading PDF page count: {e}")

    page_ranges = []
    results_dir = os.path.join(run_dir, "results")
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".jsonl"):
                jsonl_path = os.path.join(results_dir, f)
                try:
                    with open(jsonl_path, encoding="utf-8") as file:
                        for line in file:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            attributes = data.get("attributes", {})
                            pdf_page_numbers = attributes.get("pdf_page_numbers", [])
                            if pdf_page_numbers:
                                page_ranges = pdf_page_numbers
                                break
                except Exception as e:
                    logger.error(f"Error reading jsonl {jsonl_path}: {e}")
                if page_ranges:
                    break

    full_markdown = ""
    if filename:
        md_path = os.path.join(run_dir, "markdown", "inputs", filename)
        if os.path.isfile(md_path):
            try:
                with open(md_path, encoding="utf-8") as f:
                    full_markdown = f.read()
            except Exception:
                pass

    pages_markdown = {}
    if full_markdown:
        if page_ranges:
            for r in page_ranges:
                if len(r) >= 3:
                    s_idx, e_idx, p_num = r[0], r[1], r[2]
                    pages_markdown[str(p_num)] = full_markdown[s_idx:e_idx]
        else:
            chunk_len = max(1, len(full_markdown) // max(1, total_pages))
            for p in range(1, total_pages + 1):
                start = (p - 1) * chunk_len
                end = len(full_markdown) if p == total_pages else p * chunk_len
                pages_markdown[str(p)] = full_markdown[start:end]

    return {
        "run_name": run_name,
        "filename": filename,
        "total_pages": max(1, total_pages),
        "page_ranges": page_ranges,
        "pages_markdown": pages_markdown,
    }

