import base64
import json
import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

import process_state
from path_security import (
    PathSecurityError,
    require_approved_file,
    resolve_file_under,
    resolve_run_under,
    resolve_under,
    validate_filename,
)
from settings_manager import WORKSPACE_DIR

logger = logging.getLogger(__name__)


def is_safe_filename(filename):
    try:
        validate_filename(filename, {".md"})
        return True
    except PathSecurityError:
        return False


def make_zip(markdown_dir, zip_path):
    placeholder = require_approved_file(
        Path(markdown_dir) / "_boundary.md", {WORKSPACE_DIR}, {".md"}
    )
    safe_markdown_dir = placeholder.parent
    safe_zip = require_approved_file(zip_path, {WORKSPACE_DIR}, {".zip"})
    with zipfile.ZipFile(safe_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        if not safe_markdown_dir.is_dir():
            return
        for entry in safe_markdown_dir.iterdir():
            try:
                file_path = resolve_file_under(safe_markdown_dir, entry.name, {".md"})
            except PathSecurityError:
                continue
            if file_path.is_file():
                zipf.write(file_path, entry.name)


def _active_run_dir(run_id_state):
    with process_state.active_runs_lock:
        run_info = process_state.active_runs.get(run_id_state)
        if not run_info:
            return None
        candidate = Path(run_info.get("run_dir", ""))
    try:
        run_dir = resolve_run_under(WORKSPACE_DIR, candidate.name)
    except PathSecurityError:
        return None
    return run_dir if candidate.resolve() == run_dir else None


def load_markdown_content(selected_file, run_id_state):
    if not selected_file or not run_id_state:
        return "", "", None

    if not is_safe_filename(selected_file):
        return "Invalid file path.", "Invalid file path.", None

    run_dir = _active_run_dir(run_id_state)
    if run_dir is None:
        return "Run info not found.", "Run info not found.", None
    try:
        file_path = resolve_file_under(
            resolve_under(run_dir, "markdown", "inputs"), selected_file, {".md"}
        )
    except PathSecurityError:
        return "Invalid file path.", "Invalid file path.", None
    if file_path.is_file():
        try:
            content = file_path.read_text(encoding="utf-8")
            return content, content, str(file_path)
        except OSError:
            return "Error reading file.", "Error reading file.", None
    return "File not found.", "File not found.", None


def pil_to_base64(img):
    if img is None:
        return ""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def render_pdf_page(pdf_path, page_num):
    if not pdf_path:
        return None
    try:
        safe_pdf = require_approved_file(pdf_path, {WORKSPACE_DIR}, {".pdf"})
        if not safe_pdf.is_file():
            return None
        doc = pdfium.PdfDocument(str(safe_pdf))
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        bitmap = page.render(scale=2)
        return bitmap.to_pil()
    except Exception as e:
        logger.error(f"Error rendering PDF page: {e}")
        return None


def get_page_mapping_and_pdf_path(selected_file, run_id_state):
    if not selected_file or not run_id_state:
        return None, 0, []

    if not is_safe_filename(selected_file):
        return None, 0, []

    run_dir = _active_run_dir(run_id_state)
    if run_dir is None:
        return None, 0, []

    pdf_filename = selected_file.rsplit(".", 1)[0] + ".pdf"
    try:
        pdf_path = resolve_file_under(resolve_under(run_dir, "inputs"), pdf_filename, {".pdf"})
    except PathSecurityError:
        return None, 0, []
    if not pdf_path.is_file():
        return None, 0, []

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF page count: {e}")
        total_pages = 0

    page_ranges = []
    results_dir = resolve_under(run_dir, "results")
    if results_dir.is_dir():
        for entry in results_dir.iterdir():
            try:
                jsonl_path = resolve_file_under(results_dir, entry.name, {".jsonl"})
            except PathSecurityError:
                continue
            if jsonl_path.is_file():
                try:
                    with jsonl_path.open(encoding="utf-8") as file:
                        for line in file:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            source_file = data.get("metadata", {}).get("Source-File", "")
                            expected_source = f"inputs/{pdf_filename}"
                            if (
                                source_file == expected_source
                                or os.path.basename(source_file) == pdf_filename
                            ):
                                attributes = data.get("attributes", {})
                                pdf_page_numbers = attributes.get("pdf_page_numbers", [])
                                page_ranges = pdf_page_numbers
                                break
                except Exception:
                    logger.error("Error reading run page mapping")
                if page_ranges:
                    break

    return str(pdf_path), total_pages, page_ranges


def get_markdown_for_page(full_markdown, page_ranges, page_num):
    if not full_markdown:
        return ""
    if not page_ranges:
        return full_markdown

    for range_info in page_ranges:
        if len(range_info) >= 3:
            start_idx, end_idx, p_num = range_info[0], range_info[1], range_info[2]
            if p_num == page_num:
                return full_markdown[start_idx:end_idx]

    return ""


def on_file_selected(selected_file, run_id_state):
    if not selected_file or not run_id_state:
        return "", 0, [], "", {"maximum": 2, "value": 1, "interactive": False}, None

    if not is_safe_filename(selected_file):
        return (
            "",
            0,
            [],
            "Invalid file path.",
            {"maximum": 2, "value": 1, "interactive": False},
            None,
        )

    pdf_path, total_pages, page_ranges = get_page_mapping_and_pdf_path(selected_file, run_id_state)

    file_path = None
    full_markdown = ""
    run_dir = _active_run_dir(run_id_state)
    if run_dir is not None:
        try:
            safe_file = resolve_file_under(
                resolve_under(run_dir, "markdown", "inputs"), selected_file, {".md"}
            )
        except PathSecurityError:
            safe_file = None
        if safe_file is not None and safe_file.is_file():
            try:
                full_markdown = safe_file.read_text(encoding="utf-8")
                file_path = str(safe_file)
            except OSError:
                logger.error("Error reading Markdown file")
                full_markdown = "Error reading file."

    return (
        pdf_path or "",
        total_pages or 0,
        page_ranges or [],
        full_markdown,
        {"maximum": max(2, total_pages), "value": 1, "interactive": (total_pages > 1)},
        file_path,
    )


def update_view(
    selected_file, view_mode, page_num, pdf_path, total_pages, page_ranges, full_markdown
):
    if not selected_file:
        return (
            "<div id='pdf-scroll-container' class='sync-scroll-target pdf-viewer-placeholder'>Select a processed document to view.</div>",
            "<div id='raw-scroll-container' class='sync-scroll-target raw-markdown-placeholder'>Select a processed document to view.</div>",
            "Select a processed document to preview.",
        )

    pdf_html = ""
    if view_mode == "Full Document":
        try:
            safe_pdf = (
                require_approved_file(pdf_path, {WORKSPACE_DIR}, {".pdf"}) if pdf_path else None
            )
        except PathSecurityError:
            safe_pdf = None
        if safe_pdf is not None and safe_pdf.is_file():
            pdf_url = f"/gradio_api/file={safe_pdf}"
            pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target pdf-scroll-outer">
                <iframe src="{pdf_url}" class="pdf-iframe"></iframe>
            </div>"""
        else:
            pdf_html = """<div id="pdf-scroll-container" class="sync-scroll-target pdf-viewer-alternative">
                <span>Original PDF file not found.</span>
            </div>"""
    else:
        try:
            safe_pdf = (
                require_approved_file(pdf_path, {WORKSPACE_DIR}, {".pdf"}) if pdf_path else None
            )
        except PathSecurityError:
            safe_pdf = None
        if safe_pdf is not None and safe_pdf.is_file():
            pil_img = render_pdf_page(str(safe_pdf), page_num)
            if pil_img:
                img_b64 = pil_to_base64(pil_img)
                pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target pdf-image-container">
                    <img src="{img_b64}" class="pdf-image-view">
                </div>"""
            else:
                pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target pdf-viewer-alternative">
                    <span>Failed to render page {page_num}</span>
                </div>"""
        else:
            pdf_html = """<div id="pdf-scroll-container" class="sync-scroll-target pdf-viewer-alternative">
                <span>Original PDF file not found.</span>
            </div>"""

    raw_md_text = ""
    if view_mode == "Full Document":
        raw_md_text = full_markdown
    else:
        raw_md_text = get_markdown_for_page(full_markdown, page_ranges, page_num)

    escaped_raw = raw_md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    raw_md_html = f"""<div id="raw-scroll-container" class="sync-scroll-target raw-md-view-container">{escaped_raw}</div>"""

    return pdf_html, raw_md_html, raw_md_text
