import os
import json
import base64
import zipfile
from io import BytesIO
import pypdfium2 as pdfium
from pypdf import PdfReader
import gradio as gr
import state

def make_zip(markdown_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(markdown_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, markdown_dir)
                    zipf.write(file_path, arcname)

def load_markdown_content(selected_file, run_id_state):
    if not selected_file or not run_id_state:
        return "", "", None

    with state.active_runs_lock:
        run_info = state.active_runs.get(run_id_state)
        if not run_info:
            return "Run info not found.", "Run info not found.", None
        run_dir = run_info["run_dir"]

    file_path = os.path.join(run_dir, "markdown", "inputs", selected_file)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, content, file_path
        except Exception as e:
            return f"Error reading file: {e}", f"Error reading file: {e}", None
    return "File not found.", "File not found.", None

def pil_to_base64(img):
    if img is None:
        return ""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def render_pdf_page(pdf_path, page_num):
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        doc = pdfium.PdfDocument(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        bitmap = page.render(scale=2)
        return bitmap.to_pil()
    except Exception as e:
        print(f"Error rendering PDF page: {e}")
        return None

def get_page_mapping_and_pdf_path(selected_file, run_id_state):
    if not selected_file or not run_id_state:
        return None, 0, []

    with state.active_runs_lock:
        run_info = state.active_runs.get(run_id_state)
        if not run_info:
            return None, 0, []
        run_dir = run_info["run_dir"]

    pdf_filename = selected_file.rsplit(".", 1)[0] + ".pdf"
    pdf_path = os.path.join(run_dir, "inputs", pdf_filename)
    if not os.path.exists(pdf_path):
        return None, 0, []

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF page count: {e}")
        total_pages = 0

    page_ranges = []
    results_dir = os.path.join(run_dir, "results")
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".jsonl"):
                jsonl_path = os.path.join(results_dir, f)
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as file:
                        for line in file:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            source_file = data.get("metadata", {}).get("Source-File", "")
                            expected_source = f"inputs/{pdf_filename}"
                            if source_file == expected_source or os.path.basename(source_file) == pdf_filename:
                                attributes = data.get("attributes", {})
                                pdf_page_numbers = attributes.get("pdf_page_numbers", [])
                                page_ranges = pdf_page_numbers
                                break
                except Exception as e:
                    print(f"Error reading jsonl {jsonl_path}: {e}")
                if page_ranges:
                    break

    return pdf_path, total_pages, page_ranges

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
        return "", 0, [], "", gr.update(maximum=2, value=1, interactive=False), None

    pdf_path, total_pages, page_ranges = get_page_mapping_and_pdf_path(selected_file, run_id_state)
    
    full_markdown = ""
    with state.active_runs_lock:
        run_info = state.active_runs.get(run_id_state)
        if run_info:
            run_dir = run_info["run_dir"]
            file_path = os.path.join(run_dir, "markdown", "inputs", selected_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        full_markdown = f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    full_markdown = f"Error reading file: {e}"

    return (
        pdf_path or "",
        total_pages or 0,
        page_ranges or [],
        full_markdown,
        gr.update(maximum=max(2, total_pages), value=1, interactive=(total_pages > 1)),
        pdf_path
    )

def update_view(selected_file, view_mode, page_num, pdf_path, total_pages, page_ranges, full_markdown):
    if not selected_file:
        return (
            "<div id='pdf-scroll-container' class='sync-scroll-target' style='height: 70vh; display: flex; justify-content: center; align-items: center; background: #0f172a; color: #94a3b8; border-radius: 8px;'>Select a processed document to view.</div>",
            "<div id='raw-scroll-container' class='sync-scroll-target' style='height: 70vh; display: flex; justify-content: center; align-items: center; background: #020617; color: #94a3b8; border-radius: 8px;'>Select a processed document to view.</div>",
            "Select a processed document to preview."
        )

    pdf_html = ""
    if view_mode == "Full Document":
        if pdf_path and os.path.exists(pdf_path):
            pdf_url = f"/file={pdf_path}"
            pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target" style="height: 70vh; overflow: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;">
                <iframe src="{pdf_url}" style="width: 100%; height: 100%; border: none;"></iframe>
            </div>"""
        else:
            pdf_html = """<div id="pdf-scroll-container" class="sync-scroll-target" style="height: 70vh; display: flex; justify-content: center; align-items: center; background: #1e293b; color: #94a3b8; border-radius: 8px;">
                <span>Original PDF file not found.</span>
            </div>"""
    else:
        if pdf_path and os.path.exists(pdf_path):
            pil_img = render_pdf_page(pdf_path, page_num)
            if pil_img:
                img_b64 = pil_to_base64(pil_img)
                pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target" style="height: 70vh; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; background: #000; display: flex; justify-content: center; align-items: flex-start;">
                    <img src="{img_b64}" style="width: 100%; height: auto; display: block;">
                </div>"""
            else:
                pdf_html = f"""<div id="pdf-scroll-container" class="sync-scroll-target" style="height: 70vh; display: flex; justify-content: center; align-items: center; background: #1e293b; color: #94a3b8; border-radius: 8px;">
                    <span>Failed to render page {page_num}</span>
                </div>"""
        else:
            pdf_html = """<div id="pdf-scroll-container" class="sync-scroll-target" style="height: 70vh; display: flex; justify-content: center; align-items: center; background: #1e293b; color: #94a3b8; border-radius: 8px;">
                <span>Original PDF file not found.</span>
            </div>"""

    raw_md_text = ""
    if view_mode == "Full Document":
        raw_md_text = full_markdown
    else:
        raw_md_text = get_markdown_for_page(full_markdown, page_ranges, page_num)

    escaped_raw = (
        raw_md_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    raw_md_html = f"""<div id="raw-scroll-container" class="sync-scroll-target" style="height: 70vh; overflow-y: auto; font-family: 'JetBrains Mono', monospace; white-space: pre-wrap; background: #020617; color: #38bdf8; padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; line-height: 1.5;">{escaped_raw}</div>"""

    return pdf_html, raw_md_html, raw_md_text
