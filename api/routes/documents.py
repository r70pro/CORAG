"""Document and run browsing API routes."""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pypdf import PdfReader

from path_security import (
    PathSecurityError,
    resolve_file_under,
    resolve_run_under,
    validate_filename,
)
from settings_manager import WORKSPACE_DIR

logger = logging.getLogger(__name__)
router = APIRouter()


def _bad_path() -> HTTPException:
    return HTTPException(status_code=400, detail="Invalid path")


def _run_dir(run_name: str) -> Path:
    try:
        return resolve_run_under(WORKSPACE_DIR, run_name)
    except PathSecurityError as exc:
        raise _bad_path() from exc


def _safe_files(directory: Path, extension: str) -> list[Path]:
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for entry in directory.iterdir():
        try:
            safe_entry = resolve_file_under(directory, entry.name, {extension})
        except PathSecurityError:
            continue
        if safe_entry.is_file():
            result.append(safe_entry)
    return sorted(result, key=lambda path: path.name)


@router.get("/runs", summary="List completed runs")
def list_runs():
    """Return completed OCR runs without disclosing absolute filesystem paths."""
    from settings_manager import get_available_runs

    result = []
    for display_name, candidate_dir in get_available_runs():
        run_name = Path(candidate_dir).name
        try:
            run_dir = resolve_run_under(WORKSPACE_DIR, run_name)
        except PathSecurityError:
            continue
        if Path(candidate_dir).resolve() != run_dir:
            continue
        md_dir = run_dir / "markdown" / "inputs"
        files = [path.name for path in _safe_files(md_dir, ".md")]
        has_pdf = bool(_safe_files(run_dir / "inputs", ".pdf"))
        result.append(
            {
                "display_name": display_name,
                "run_name": run_name,
                "file_count": len(files),
                "files": files,
                "has_pdf": has_pdf,
            }
        )
    return result


@router.get("/runs/{run_name}/files", summary="List files in a run")
def list_run_files(run_name: str):
    """Return all Markdown files in a specific run."""
    run_dir = _run_dir(run_name)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return [path.name for path in _safe_files(run_dir / "markdown" / "inputs", ".md")]


@router.get("/runs/{run_name}/markdown/{filename}", summary="Get markdown content")
def get_markdown(run_name: str, filename: str):
    """Return Markdown content from a file inside the selected run."""
    run_dir = _run_dir(run_name)
    try:
        file_path = resolve_file_under(run_dir / "markdown" / "inputs", filename, {".md"})
    except PathSecurityError as exc:
        raise _bad_path() from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return PlainTextResponse(file_path.read_text(encoding="utf-8"))


@router.get("/runs/{run_name}/markdown.zip", summary="Download all Markdown files")
def download_run_markdown(run_name: str):
    """Return a ZIP containing every safe Markdown output in a selected run."""
    run_dir = _run_dir(run_name)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    markdown_files = _safe_files(run_dir / "markdown" / "inputs", ".md")
    if not markdown_files:
        raise HTTPException(status_code=404, detail="No Markdown files found")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for markdown_file in markdown_files:
            zip_file.write(markdown_file, arcname=markdown_file.name)
    archive.seek(0)
    safe_download_name = re.sub(r"[^A-Za-z0-9_.-]", "_", run_name)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_download_name}_markdown.zip"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/runs/{run_name}/pdf", summary="Get source PDF file")
def get_run_pdf(run_name: str):
    """Return the first source PDF stored inside a selected run."""
    run_dir = _run_dir(run_name)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    pdf_files = _safe_files(run_dir / "inputs", ".pdf")
    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF found")

    headers = {
        "Content-Disposition": "inline; filename=document.pdf",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "unsafe-none",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    return FileResponse(pdf_files[0], media_type="application/pdf", headers=headers)


@router.get("/runs/{run_name}/info", summary="Get detailed page mapping and info for a document")
def get_run_doc_info(run_name: str, filename: str = ""):
    """Return page count, character ranges, and per-page Markdown content."""
    run_dir = _run_dir(run_name)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    if filename:
        try:
            validate_filename(filename, {".md"})
        except PathSecurityError as exc:
            raise _bad_path() from exc

    inputs_dir = run_dir / "inputs"
    pdf_path: Path | None = None
    pdf_filename = ""
    if filename:
        pdf_filename = f"{Path(filename).stem}.pdf"
        try:
            candidate = resolve_file_under(inputs_dir, pdf_filename, {".pdf"})
        except PathSecurityError as exc:
            raise _bad_path() from exc
        if candidate.is_file():
            pdf_path = candidate
    if pdf_path is None:
        pdf_files = _safe_files(inputs_dir, ".pdf")
        if pdf_files:
            pdf_path = pdf_files[0]

    total_pages = 1
    if pdf_path is not None:
        try:
            total_pages = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            logger.warning("Unable to read a run PDF page count")

    page_ranges = []
    for jsonl_path in _safe_files(run_dir / "results", ".jsonl"):
        try:
            with jsonl_path.open(encoding="utf-8") as jsonl_file:
                for line in jsonl_file:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    source_file = data.get("metadata", {}).get("Source-File", "")
                    source_name = Path(str(source_file).replace("\\", "/")).name
                    if not pdf_filename or source_name == pdf_filename:
                        ranges = data.get("attributes", {}).get("pdf_page_numbers", [])
                        if ranges:
                            page_ranges = ranges
                            break
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Unable to read run page mapping metadata")
        if page_ranges:
            break

    full_markdown = ""
    if filename:
        try:
            md_path = resolve_file_under(run_dir / "markdown" / "inputs", filename, {".md"})
        except PathSecurityError as exc:
            raise _bad_path() from exc
        if md_path.is_file():
            try:
                full_markdown = md_path.read_text(encoding="utf-8")
            except OSError:
                logger.warning("Unable to read run Markdown")

    pages_markdown = {}
    if full_markdown:
        valid_ranges = False
        for page_range in page_ranges:
            if len(page_range) >= 3:
                start, end, page_number = page_range[:3]
                if (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(full_markdown)
                ):
                    pages_markdown[str(page_number)] = full_markdown[start:end]
                    valid_ranges = True

        if not valid_ranges:
            splits = re.split(
                r"\n\s*(?:---|<!--\s*page\s*\d+\s*-->)\s*\n",
                full_markdown,
                flags=re.IGNORECASE,
            )
            if len(splits) > 1:
                pages_markdown = {
                    str(index): text.strip() for index, text in enumerate(splits, start=1)
                }
            else:
                pages_markdown = {
                    str(page): full_markdown for page in range(1, max(1, total_pages) + 1)
                }

    return {
        "run_name": run_name,
        "filename": filename,
        "total_pages": max(1, total_pages),
        "page_ranges": page_ranges,
        "pages_markdown": pages_markdown,
    }
