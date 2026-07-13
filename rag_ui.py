"""
RAG Analysis UI — Gradio components for the document analysis tab.

Builds the UI layout and event handlers for:
- RAG infrastructure management
- Document indexing (single run + multi-run corpus)
- Chat interface with streaming responses
- Analysis mode selection (Free Q&A, Timeline, Summary, etc.)
- Source citation display
"""

import os
import re
import hashlib
import gradio as gr

from settings_manager import WORKSPACE_DIR, load_settings


# ── Indexing functions ─────────────────────────────────────────

def get_available_runs():
    """Scan workspace for completed OCR runs that have markdown output.

    Returns:
        List of (display_name, run_dir_path) tuples for the dropdown.
    """
    runs = []
    workspace = WORKSPACE_DIR
    if not os.path.exists(workspace):
        return runs

    for name in sorted(os.listdir(workspace), reverse=True):
        run_dir = os.path.join(workspace, name)
        if not os.path.isdir(run_dir) or not name.startswith("run_"):
            continue
        md_dir = os.path.join(run_dir, "markdown", "inputs")
        if os.path.exists(md_dir):
            md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
            if md_files:
                # Format: "run_20260711_092213 (1 file, 9 pages)"
                display = f"{name} ({len(md_files)} file{'s' if len(md_files) != 1 else ''})"
                runs.append((display, run_dir))

    return runs


def index_run(run_dir, progress=None):
    """Index a single OCR run into the RAG system.

    Chunks all markdown files, embeds them, and stores in Qdrant + PostgreSQL.

    Args:
        run_dir: Path to the OCR run directory.
        progress: Optional Gradio progress tracker.

    Yields:
        Status update strings.
    """
    if not run_dir or not os.path.exists(run_dir):
        yield "⚠️ Invalid run directory."
        return

    run_name = os.path.basename(run_dir)

    # Extract a stable run_id from the directory name
    run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]

    yield f"🔄 Starting indexing for **{run_name}**...\n"

    # Check if already indexed
    try:
        from rag.db import is_run_indexed
        if is_run_indexed(run_id):
            yield f"ℹ️ Run **{run_name}** is already indexed. Skipping.\n"
            yield "✅ Done."
            return
    except Exception as e:
        yield f"⚠️ Could not check index status: {e}\n"

    # Step 1: Chunk documents
    yield "📄 Chunking documents...\n"
    try:
        from rag.chunker import chunk_documents_from_run
        settings = load_settings()
        chunk_results = chunk_documents_from_run(
            run_dir=run_dir,
            run_id=run_id,
            max_chunk_size=settings.get("chunk_size", 800),
            chunk_overlap=settings.get("chunk_overlap", 100),
        )
    except Exception as e:
        yield f"❌ Chunking failed: {e}\n"
        return

    if not chunk_results:
        yield "⚠️ No markdown files found in this run.\n"
        return

    total_chunks = sum(len(info["chunks"]) for info in chunk_results.values())
    total_docs = len(chunk_results)
    yield f"  Found **{total_docs}** document(s), **{total_chunks}** chunk(s).\n"

    # Step 2: Register run in PostgreSQL
    yield "💾 Registering run in database...\n"
    try:
        from rag.db import register_run, register_document, insert_chunks, mark_run_indexed, mark_document_indexed
        register_run(run_id, run_dir, total_documents=total_docs)

        for doc_id, info in chunk_results.items():
            md_file = info["md_file"]
            # Extract original filename (strip numeric prefix)
            orig_match = re.match(r"^\d+_(.*)", md_file)
            orig_name = orig_match.group(1) if orig_match else md_file

            pdf_pages = 0
            # Count pages from page ranges if available
            if info.get("page_ranges"):
                pdf_pages = len(info["page_ranges"])

            register_document(
                doc_id=doc_id,
                run_id=run_id,
                original_filename=orig_name,
                pdf_total_pages=pdf_pages,
                markdown_path=info["md_path"],
            )
    except Exception as e:
        yield f"❌ Database registration failed: {e}\n"
        return

    # Step 3: Upload to MinIO
    yield "☁️ Uploading to object storage...\n"
    try:
        from rag.storage import upload_markdown, upload_pdf

        for doc_id, info in chunk_results.items():
            # Upload markdown
            if os.path.exists(info["md_path"]):
                upload_markdown(run_id, doc_id, info["md_path"])

            # Upload corresponding PDF if exists
            pdf_filename = info["md_file"].replace(".md", ".pdf")
            pdf_path = os.path.join(run_dir, "inputs", pdf_filename)
            if os.path.exists(pdf_path):
                upload_pdf(run_id, doc_id, pdf_path)

    except Exception as e:
        yield f"⚠️ Storage upload warning: {e}\n"
        # Non-fatal — indexing can continue without MinIO

    # Step 4: Embed and upsert into Qdrant
    yield "🧠 Embedding chunks and indexing in vector store...\n"
    try:
        from rag.embedding import upsert_chunks

        all_chunks = []
        for doc_id, info in chunk_results.items():
            all_chunks.extend(info["chunks"])

        # Upsert in batches
        updated_chunks = upsert_chunks(all_chunks, batch_size=32)

        # Store chunk metadata in PostgreSQL
        insert_chunks(updated_chunks)

        # Mark documents and run as indexed
        for doc_id in chunk_results:
            mark_document_indexed(doc_id)
        mark_run_indexed(run_id, total_chunks=len(updated_chunks))

    except Exception as e:
        yield f"❌ Embedding/indexing failed: {e}\n"
        return

    # Step 5: Invalidate query cache
    try:
        from rag.cache import invalidate_query_cache
        invalidate_query_cache()
    except Exception:
        pass  # Non-fatal

    yield f"\n✅ Successfully indexed **{run_name}**: {total_docs} document(s), {total_chunks} chunk(s).\n"


def index_all_runs():
    """Index all available OCR runs into the RAG corpus.

    Yields:
        Status update strings.
    """
    runs = get_available_runs()
    if not runs:
        yield "⚠️ No completed OCR runs found in workspace.\n"
        return

    yield f"🔄 Indexing **{len(runs)}** run(s) into the corpus...\n\n"

    for display_name, run_dir in runs:
        yield f"--- Processing: {display_name} ---\n"
        for update in index_run(run_dir):
            yield update
        yield "\n"

    yield "\n✅ All runs processed."


RAG_LOG_BUFFER = []
LAST_CREATED_RUN_ID = None

def log_to_rag(message: str):
    """Log a message to the RAG system log buffer with a timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Clean message of markdown formatting for console
    clean_msg = message.replace("**", "").replace("`", "").strip()
    if clean_msg:
        RAG_LOG_BUFFER.append(f"[{timestamp}] {clean_msg}")
        if len(RAG_LOG_BUFFER) > 500:
            RAG_LOG_BUFFER.pop(0)

def get_rag_logs() -> str:
    """Get all accumulated RAG logs as a single string."""
    return "\n".join(RAG_LOG_BUFFER)


def start_rag_infra_ui_wrapper():
    log_to_rag("Starting RAG infrastructure services...")
    msg, status_html = start_rag_infra_ui()
    log_to_rag(f"Start infrastructure result: {msg}")
    return msg, status_html, get_rag_logs()


def stop_rag_infra_ui_wrapper():
    log_to_rag("Stopping RAG infrastructure services...")
    msg, status_html = stop_rag_infra_ui()
    log_to_rag(f"Stop infrastructure result: {msg}")
    return msg, status_html, get_rag_logs()


def index_run_ui_wrapper(run_dir):
    accumulated_status = ""
    log_to_rag(f"Initiated manual indexing for run directory: {run_dir}")
    for update in index_run(run_dir):
        accumulated_status += update
        log_to_rag(update)
        yield accumulated_status, get_rag_logs()


def index_all_runs_ui_wrapper():
    accumulated_status = ""
    log_to_rag("Initiated bulk indexing for all runs")
    for update in index_all_runs():
        accumulated_status += update
        log_to_rag(update)
        yield accumulated_status, get_rag_logs()


def upload_and_index_markdown(files, case_option, new_case_name):
    if not files:
        yield "⚠️ No files uploaded.\n"
        return

    import os
    import shutil
    import datetime
    import hashlib
    import re
    from settings_manager import WORKSPACE_DIR, load_settings
    from rag.db import register_run, register_document, insert_chunks, mark_run_indexed, mark_document_indexed, get_runs_with_stats, get_connection
    from rag.chunker import chunk_document
    from rag.embedding import upsert_chunks
    from rag.storage import upload_markdown
    from rag.cache import invalidate_query_cache

    global LAST_CREATED_RUN_ID
    if case_option == "new":
        if not new_case_name or not new_case_name.strip():
            yield "❌ Error: New case name is required.\n"
            return
        
        # Create a new run/case directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', new_case_name.strip())
        run_name = f"run_{clean_name}_{timestamp}"
        run_dir = os.path.join(WORKSPACE_DIR, run_name)
        run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]
        LAST_CREATED_RUN_ID = run_id
        
        yield f"📁 Creating new case: **{new_case_name}**...\n"
    else:
        # Find existing run details
        run_id = case_option
        LAST_CREATED_RUN_ID = run_id
        run_dir = None
        try:
            runs = get_runs_with_stats()
            for r in runs:
                if r.get("run_id") == run_id:
                    run_dir = r.get("run_dir")
                    break
        except Exception as e:
            yield f"❌ Failed to fetch case information: {e}\n"
            return
        
        if not run_dir:
            # Fallback if get_runs_with_stats fails or doesn't have it (maybe it's not status='indexed' yet, or exists in ocr_runs)
            # Let's query ocr_runs directly
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT run_dir FROM ocr_runs WHERE run_id = %s", (run_id,))
                        row = cur.fetchone()
                        if row:
                            run_dir = row[0]
            except Exception as e:
                yield f"❌ Failed to retrieve run directory: {e}\n"
                return
        
        if not run_dir:
            yield "❌ Error: Could not locate existing case directory.\n"
            return
        
        run_name = os.path.basename(run_dir)
        yield f"📁 Adding to existing case: **{run_name}**...\n"

    # Set up directory paths
    markdown_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
    try:
        os.makedirs(markdown_inputs_dir, exist_ok=True)
    except Exception as e:
        yield f"❌ Failed to create directories: {e}\n"
        return

    # Step 1: Copy uploaded files to the case directory
    copied_files = []
    for file_info in files:
        # Gradio files can be Tempfile objects or strings
        file_path = file_info.name if hasattr(file_info, "name") else str(file_info)
        if not os.path.exists(file_path):
            continue
        filename = os.path.basename(file_path)
        dest_path = os.path.join(markdown_inputs_dir, filename)
        try:
            shutil.copy(file_path, dest_path)
            copied_files.append((filename, dest_path))
            yield f"📄 Copied **{filename}** to case storage.\n"
        except Exception as e:
            yield f"⚠️ Warning: Could not copy {filename}: {e}\n"

    if not copied_files:
        yield "❌ Error: No files were successfully copied.\n"
        return

    # Step 2: Register/update run in PostgreSQL
    yield "💾 Registering case metadata in database...\n"
    try:
        # Determine total documents currently in DB for this run_id
        current_docs_count = 0
        if case_option != "new":
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents WHERE run_id = %s", (run_id,))
                    current_docs_count = cur.fetchone()[0]
        
        new_total_docs = current_docs_count + len(copied_files)
        register_run(run_id, run_dir, total_documents=new_total_docs)
    except Exception as e:
        yield f"❌ Database run registration failed: {e}\n"
        return

    # Step 3: Process each file (chunk, register document, embed, upsert)
    settings = load_settings()
    max_chunk_size = settings.get("chunk_size", 800)
    chunk_overlap = settings.get("chunk_overlap", 100)
    
    all_new_chunks = []

    for filename, md_path in copied_files:
        yield f"⚙️ Processing **{filename}**...\n"
        
        # Read contents
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
        except Exception as e:
            yield f"⚠️ Error reading {filename}: {e}. Skipping.\n"
            continue

        # Generate deterministic doc_id
        doc_id = hashlib.sha256(f"{run_id}:{filename}".encode()).hexdigest()[:24]

        # Register document
        try:
            register_document(
                doc_id=doc_id,
                run_id=run_id,
                original_filename=filename,
                pdf_total_pages=0,
                markdown_path=md_path,
            )
        except Exception as e:
            yield f"⚠️ Database document registration failed for {filename}: {e}. Skipping.\n"
            continue

        # Upload to MinIO
        try:
            upload_markdown(run_id, doc_id, md_path)
            yield f"☁️ Uploaded **{filename}** to object storage.\n"
        except Exception as e:
            yield f"⚠️ Storage upload warning for {filename}: {e}\n"

        # Chunk document
        try:
            chunks = chunk_document(
                markdown_text=markdown_text,
                doc_id=doc_id,
                run_id=run_id,
                page_ranges=[],
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_new_chunks.extend(chunks)
            yield f"🧩 Created **{len(chunks)}** chunk(s) for {filename}.\n"
        except Exception as e:
            yield f"⚠️ Chunking failed for {filename}: {e}. Skipping.\n"
            continue

    if not all_new_chunks:
        yield "❌ Error: No chunks generated from the uploaded files.\n"
        return

    # Step 4: Embed and upsert into Qdrant & Postgres
    yield f"🧠 Embedding {len(all_new_chunks)} chunks and indexing in vector store...\n"
    try:
        updated_chunks = upsert_chunks(all_new_chunks, batch_size=32)
        insert_chunks(updated_chunks)
        
        # Mark all documents as indexed
        for filename, md_path in copied_files:
            doc_id = hashlib.sha256(f"{run_id}:{filename}".encode()).hexdigest()[:24]
            mark_document_indexed(doc_id)
            
        # Get total chunks for the run to mark run indexed
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE run_id = %s", (run_id,))
                total_chunks_in_run = cur.fetchone()[0]
                
        mark_run_indexed(run_id, total_chunks=total_chunks_in_run)
        
    except Exception as e:
        yield f"❌ Embedding/indexing failed: {e}\n"
        return

    # Step 5: Invalidate query cache
    try:
        invalidate_query_cache()
    except Exception:
        pass

    yield f"\n✅ Successfully uploaded and indexed **{len(copied_files)}** markdown file(s) into case **{run_name}**!\n"


def upload_and_index_markdown_ui_wrapper(files, case_option, new_case_name):
    accumulated_status = ""
    log_to_rag("Initiated external markdown upload and indexing")
    for update in upload_and_index_markdown(files, case_option, new_case_name):
        accumulated_status += update
        log_to_rag(update)
        yield accumulated_status, get_rag_logs()




def extract_text_content(content) -> str:
    """Extract plain text from potential Gradio 6 chatbot content format."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Gradio 6 format: {'text': "...", 'type': 'text'}
                if "text" in item:
                    text_parts.append(item["text"])
        return "".join(text_parts)
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
    return str(content)


# ── Chat functions ─────────────────────────────────────────────

def chat_respond(message, history, analysis_mode, analysis_model_url, analysis_model_name, top_k):
    """Handle a chat message with RAG-augmented response.

    Args:
        message: User's message.
        history: Chat history (list of [user, assistant] pairs).
        analysis_mode: Selected analysis mode key.
        analysis_model_url: vLLM server URL.
        analysis_model_name: Model name.
        top_k: Number of chunks to retrieve.

    Yields:
        Partial response strings for streaming display.
    """
    if not message or not message.strip():
        yield ""
        return

    # Convert Gradio chat history to OpenAI message format
    chat_history = []
    if history:
        for user_msg, assistant_msg in history:
            if user_msg:
                chat_history.append({"role": "user", "content": extract_text_content(user_msg)})
            if assistant_msg:
                chat_history.append({"role": "assistant", "content": extract_text_content(assistant_msg)})

    # Map display mode to internal key
    mode_map = {
        "💬 Free Q&A": "free_qa",
        "📅 Timeline Generator": "timeline",
        "🏥 Injury Summary": "injury_summary",
        "🔍 Inconsistency Finder": "inconsistency_finder",
        "💊 Medication Tracker": "medication_tracker",
    }
    mode_key = mode_map.get(analysis_mode, "free_qa")

    try:
        from rag.analyzer import analyze

        partial_response = ""
        for chunk in analyze(
            query=message,
            mode=mode_key,
            server_url=analysis_model_url,
            model_name=analysis_model_name,
            top_k=int(top_k),
            chat_history=chat_history,
            stream=True,
        ):
            partial_response += chunk
            yield partial_response

    except Exception as e:
        yield f"⚠️ Error: {str(e)}"


def get_corpus_info():
    """Get formatted corpus statistics for display.

    Returns:
        Markdown string with corpus stats.
    """
    try:
        from rag.db import get_corpus_stats
        from rag.embedding import get_collection_info

        db_stats = get_corpus_stats()
        qdrant_info = get_collection_info()

        runs = db_stats.get("indexed_runs", 0)
        docs = db_stats.get("indexed_documents", 0)
        chunks = db_stats.get("total_chunks", 0)
        authors = db_stats.get("unique_authors", 0)
        earliest = db_stats.get("earliest_date", "—")
        latest = db_stats.get("latest_date", "—")
        vectors = qdrant_info.get("points_count", 0)

        return (
            f"**📊 Corpus Statistics**\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Indexed Runs | {runs} |\n"
            f"| Documents | {docs} |\n"
            f"| Chunks | {chunks} |\n"
            f"| Vectors | {vectors} |\n"
            f"| Unique Authors | {authors} |\n"
            f"| Date Range | {earliest} → {latest} |\n"
        )
    except Exception as e:
        return f"⚠️ Could not fetch corpus stats: {e}"


# ── Infrastructure management ─────────────────────────────────

def start_rag_infra_ui():
    """Start RAG infrastructure and return status."""
    try:
        from rag_infra_manager import start_and_init_rag, get_rag_status_html
        success, msg = start_and_init_rag()
        status_html = get_rag_status_html()
        return msg, status_html
    except Exception as e:
        return f"❌ Error: {e}", "<span class='badge-failed'>Error</span>"


def stop_rag_infra_ui():
    """Stop RAG infrastructure and return status."""
    try:
        from rag_infra_manager import stop_rag_infrastructure, get_rag_status_html
        success, msg = stop_rag_infrastructure()
        status_html = get_rag_status_html()
        return msg, status_html
    except Exception as e:
        return f"❌ Error: {e}", "<span class='badge-failed'>Error</span>"


def refresh_rag_status():
    """Refresh RAG infrastructure status badges."""
    try:
        from rag_infra_manager import get_rag_status_html
        return get_rag_status_html()
    except Exception:
        return "<span class='badge-idle'>Unknown</span>"


def refresh_runs_dropdown():
    """Refresh the available runs dropdown."""
    runs = get_available_runs()
    if runs:
        return gr.update(choices=runs, value=runs[0][1])
    return gr.update(choices=[], value=None)


def refresh_corpus_display():
    """Refresh the corpus statistics display."""
    return get_corpus_info()


# ── UI Builder ─────────────────────────────────────────────────

# ── Dashboard rendering helpers ────────────────────────────────

def _build_dashboard_html():
    """Build HTML card grid for the Case Dashboard tab.

    Returns:
        HTML string with styled case cards.
    """
    try:
        from rag.db import get_runs_with_stats
        runs = get_runs_with_stats()
    except Exception as e:
        return f"<div class='dashboard-empty'>⚠️ Cannot load cases: {e}</div>"

    if not runs:
        return (
            "<div class='dashboard-empty'>"
            "<div style='font-size:2.5rem; margin-bottom:12px;'>📂</div>"
            "<div>No indexed cases yet.</div>"
            "<div style='font-size:0.9rem; margin-top:8px; color:#4b5563;'>"
            "Upload and index documents using the Analysis tab to see them here.</div>"
            "</div>"
        )

    cards = []
    for run in runs:
        run_dir = run.get("run_dir", "")
        run_name = os.path.basename(run_dir) if run_dir else run.get("run_id", "unknown")
        docs = run.get("total_documents", 0)
        chunks = run.get("total_chunks", 0)
        authors = run.get("unique_authors", 0)
        earliest = run.get("earliest_date", None)
        latest = run.get("latest_date", None)
        indexed_at = run.get("indexed_at", None)

        date_range = "—"
        if earliest and latest:
            date_range = f"{earliest} → {latest}"
        elif earliest:
            date_range = f"{earliest} → ..."
        elif latest:
            date_range = f"... → {latest}"

        indexed_str = ""
        if indexed_at:
            try:
                indexed_str = indexed_at.strftime("%Y-%m-%d %H:%M")
            except Exception:
                indexed_str = str(indexed_at)[:16]

        card = f"""
        <div class="case-card">
            <div class="case-card-title">📁 {run_name}</div>
            <div class="case-card-stats">
                <span>Documents: <span class="stat-val">{docs}</span></span>
                <span>Chunks: <span class="stat-val">{chunks}</span></span>
                <span>Authors: <span class="stat-val">{authors}</span></span>
                <span>Date Range: <span class="stat-val">{date_range}</span></span>
            </div>
            <div style="font-size:0.78rem; color:#6b7280; margin-bottom:8px;">
                <span class="badge-success" style="font-size:0.75rem;">✓ Indexed</span>
                {f'&nbsp; {indexed_str}' if indexed_str else ''}
            </div>
        </div>
        """
        cards.append(card)

    return f"<div class='case-dashboard-grid'>{''.join(cards)}</div>"


def _get_indexed_run_choices():
    """Get indexed runs as dropdown choices for the Active Case Selector.

    Returns:
        List of (display_label, run_id) tuples, plus an 'All Cases' option.
    """
    choices = [("🌐 All Cases (no filter)", "")]
    try:
        from rag.db import get_runs_with_stats
        runs = get_runs_with_stats()
        for run in runs:
            run_dir = run.get("run_dir", "")
            run_name = os.path.basename(run_dir) if run_dir else "unknown"
            run_id = run.get("run_id", "")
            docs = run.get("total_documents", 0)
            chunks = run.get("total_chunks", 0)
            label = f"📁 {run_name} ({docs} docs, {chunks} chunks)"
            choices.append((label, run_id))
    except Exception:
        pass
    return choices


def _refresh_active_case_after_upload():
    global LAST_CREATED_RUN_ID
    choices = _get_indexed_run_choices()
    val = LAST_CREATED_RUN_ID if LAST_CREATED_RUN_ID else ""
    return gr.update(choices=choices, value=val)


def _get_case_banner_html(active_case_label):
    """Generate the active case indicator banner HTML."""
    if not active_case_label or "All Cases" in str(active_case_label):
        return (
            "<div class='active-case-banner'>"
            "<span class='banner-icon'>🌐</span>"
            "<span><span class='banner-label'>Active Case:</span> "
            "<span class='banner-value'>All Cases — querying entire corpus</span></span>"
            "</div>"
        )
    # Extract case name from the label
    name = str(active_case_label)
    return (
        "<div class='active-case-banner'>"
        "<span class='banner-icon'>📂</span>"
        "<span><span class='banner-label'>Active Case:</span> "
        f"<span class='banner-value'>{name}</span></span>"
        "</div>"
    )


def build_case_dashboard_ui():
    """Build the Case Dashboard UI components.
    
    Returns:
        Dict of component references.
    """
    with gr.Row():
        gr.HTML(
            "<div class='dashboard-header'>"
            "<h2 style='margin:0; color:#c7d2fe; font-size:1.2rem;'>📊 Indexed Cases</h2>"
            "</div>"
        )
    with gr.Row():
        dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary", size="sm")
        dashboard_delete_selector = gr.Dropdown(
            label="Select case to delete",
            choices=[],
            interactive=True,
            scale=3,
        )
        dashboard_delete_btn = gr.Button("🗑️ Delete Case", variant="stop", size="sm")
    dashboard_html = gr.HTML(value=_build_dashboard_html())
    dashboard_status = gr.Markdown("")

    def _refresh_dashboard():
        html = _build_dashboard_html()
        choices = _get_indexed_run_choices()
        # Build delete selector choices (skip "All Cases")
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), ""

    dashboard_refresh_btn.click(
        _refresh_dashboard,
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )

    def _delete_case(run_id):
        if not run_id:
            return _build_dashboard_html(), gr.update(), "⚠️ No case selected."
        try:
            from rag.db import delete_run_data
            delete_run_data(run_id)
            log_to_rag(f"Deleted case data from PostgreSQL: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"DB delete warning: {e}")

        try:
            from rag.embedding import delete_run_vectors
            delete_run_vectors(run_id)
            log_to_rag(f"Deleted vectors from Qdrant: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"Vector delete warning: {e}")

        try:
            from rag.storage import delete_run_objects
            delete_run_objects(run_id)
            log_to_rag(f"Deleted blobs from MinIO: {run_id[:12]}...")
        except Exception as e:
            log_to_rag(f"Storage delete warning: {e}")

        try:
            from rag.cache import invalidate_query_cache
            invalidate_query_cache()
        except Exception:
            pass

        html = _build_dashboard_html()
        choices = _get_indexed_run_choices()
        del_choices = [(lbl, rid) for lbl, rid in choices if rid]
        return html, gr.update(choices=del_choices, value=None), "✅ Case deleted successfully."

    dashboard_delete_btn.click(
        _delete_case,
        inputs=[dashboard_delete_selector],
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status],
    )

    return {
        "dashboard_html": dashboard_html,
        "dashboard_delete_selector": dashboard_delete_selector,
        "dashboard_status": dashboard_status,
        "refresh_btn": dashboard_refresh_btn,
        "refresh_fn": _refresh_dashboard,
    }


def build_rag_chat_ui():
    """Build the Gradio UI components for RAG Analysis Chat.

    Returns:
        Dict of component references.
    """
    settings = load_settings()

    with gr.Row():
        # ── Left sidebar: Controls ──
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            # Infrastructure
            with gr.Accordion("🔧 RAG Infrastructure", open=False):
                rag_infra_status = gr.HTML(
                    value="<span class='badge-idle'>Not checked</span>",
                    label="Service Status"
                )
                with gr.Row():
                    rag_start_btn = gr.Button("▶️ Start", variant="primary", size="sm")
                    rag_stop_btn = gr.Button("⏹️ Stop", variant="stop", size="sm")
                rag_infra_msg = gr.Markdown("")

            # Indexing
            with gr.Accordion("📦 Document Indexing", open=True):
                corpus_stats = gr.Markdown(value="*Click refresh to load stats*")
                refresh_corpus_btn = gr.Button("🔄 Refresh Stats", variant="secondary", size="sm")

                gr.Markdown("---")
                gr.Markdown("**Index a specific run:**")
                run_selector = gr.Dropdown(
                    label="Select OCR Run",
                    choices=get_available_runs(),
                    interactive=True,
                )
                index_run_btn = gr.Button("📥 Index Selected Run", variant="primary", size="sm")

                gr.Markdown("---")
                index_all_btn = gr.Button("📥 Index All Runs", variant="secondary", size="sm")
                index_status = gr.Markdown("")

            # Upload External Markdown
            with gr.Accordion("📥 Upload External Markdown", open=False):
                gr.Markdown("Upload markdown files directly into a new or existing case, bypassing the ingestion pipeline.")
                external_md_uploader = gr.File(
                    label="Select Markdown Files (.md)",
                    file_count="multiple",
                    file_types=[".md"],
                )
                target_case_dropdown = gr.Dropdown(
                    label="Target Case",
                    choices=[("🆕 Create New Case", "new")] + [choice for choice in _get_indexed_run_choices() if choice[1] != ""],
                    value="new",
                    interactive=True,
                )
                new_case_name = gr.Textbox(
                    label="New Case Name",
                    placeholder="e.g. My Custom Case",
                    visible=True,
                )
                upload_md_btn = gr.Button("📥 Upload & Index", variant="primary", size="sm")
                upload_status = gr.Markdown("")

            # Analysis settings
            with gr.Accordion("⚙️ Analysis Settings", open=False):
                analysis_model_url = gr.Textbox(
                    label="Analysis LLM Server URL",
                    value=settings.get("analysis_server_url", "http://localhost:8000/v1"),
                    placeholder="http://localhost:8000/v1",
                )
                current_analysis_model = settings.get("analysis_model_name", "nvidia/Phi-4-reasoning-plus-NVFP4")
                analysis_choices = [
                    "allenai/olmOCR-2-7B-1025-FP8",
                    "nvidia/Qwen3.6-35B-A3B-NVFP4",
                    "nvidia/Phi-4-reasoning-plus-NVFP4",
                    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                    "nvidia/Llama-3.3-70B-Instruct-NVFP4"
                ]
                if current_analysis_model not in analysis_choices:
                    analysis_choices.append(current_analysis_model)

                analysis_model_name = gr.Dropdown(
                    label="Analysis Model Name",
                    choices=analysis_choices,
                    value=current_analysis_model,
                    interactive=True,
                )

                retrieval_top_k = gr.Slider(
                    label="Retrieval Top-K",
                    minimum=3,
                    maximum=20,
                    step=1,
                    value=settings.get("retrieval_top_k", 8),
                )
                current_embedding_model = settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
                embedding_choices = [
                    "BAAI/bge-large-en-v1.5",
                ]
                if current_embedding_model not in embedding_choices:
                    embedding_choices.append(current_embedding_model)

                embedding_model = gr.Dropdown(
                    label="Embedding Model Name",
                    choices=embedding_choices,
                    value=current_embedding_model,
                    interactive=True,
                    allow_custom_value=False,
                )
                save_analysis_btn = gr.Button("💾 Save Analysis Configuration", variant="secondary")
                analysis_config_status = gr.Markdown()

            # ── NEW: Search Filters ──
            with gr.Accordion("🔍 Search Filters", open=True):
                gr.Markdown("**🎯 Active Case** *(isolates queries to a single case)*")
                active_case_selector = gr.Dropdown(
                    label="Active Case",
                    choices=_get_indexed_run_choices(),
                    value="",
                    interactive=True,
                )

                gr.HTML("<hr style='border-color: rgba(255,255,255,0.06); margin: 8px 0;'>")
                gr.Markdown("**📋 Metadata Filters** *(narrow search scope)*")

                filter_doc_type = gr.Dropdown(
                    label="Document Type",
                    choices=[
                        ("All Types", ""),
                        ("Specialist Letter", "specialist_letter"),
                        ("Clinical Notes", "clinical_notes"),
                        ("Radiology Report", "radiology_report"),
                        ("Physiotherapy Report", "physiotherapy_report"),
                        ("Medicolegal Report", "medicolegal_report"),
                        ("Referral Letter", "referral_letter"),
                    ],
                    value="",
                    interactive=True,
                )
                filter_author = gr.Dropdown(
                    label="Author",
                    choices=[("All Authors", "")],
                    value="",
                    interactive=True,
                )
                with gr.Row():
                    filter_date_from = gr.Textbox(
                        label="Date From",
                        placeholder="YYYY-MM-DD",
                        scale=1,
                    )
                    filter_date_to = gr.Textbox(
                        label="Date To",
                        placeholder="YYYY-MM-DD",
                        scale=1,
                    )
                gr.HTML(
                    "<div class='filter-hint'>Leave blank to search all dates. "
                    "Filters apply to the next query.</div>"
                )

        # ── Right: Chat interface ──
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            # Active case banner
            active_case_banner = gr.HTML(value=_get_case_banner_html(None))

            # Analysis mode (moved here for prominence)
            with gr.Row():
                analysis_mode = gr.Dropdown(
                    label="Analysis Mode",
                    choices=[
                        "💬 Free Q&A",
                        "📅 Timeline Generator",
                        "🏥 Injury Summary",
                        "🔍 Inconsistency Finder",
                        "💊 Medication Tracker",
                    ],
                    value="💬 Free Q&A",
                    interactive=True,
                    scale=3,
                )

            chatbot = gr.Chatbot(
                label="Document Analysis Chat",
                height=1000,
                buttons=["copy"],
                avatar_images=(None, None),
                elem_classes=["analysis-chatbot"],
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Ask a question about your documents",
                    placeholder="e.g., What injuries did the patient sustain and when?",
                    scale=4,
                    lines=2,
                )
                chat_submit_btn = gr.Button("🚀 Ask", variant="primary", scale=1)

            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                export_md_btn = gr.Button("📝 Export .md", variant="secondary", size="sm")
                export_txt_btn = gr.Button("📄 Export .txt", variant="secondary", size="sm")
                export_csv_btn = gr.Button("📊 Export .csv", variant="secondary", size="sm")
                gr.HTML(
                    "<div class='shortcut-hint'>"
                    "<kbd>Ctrl</kbd>+<kbd>Enter</kbd> Send &nbsp; "
                    "<kbd>Ctrl</kbd>+<kbd>⇧</kbd>+<kbd>N</kbd> Clear &nbsp; "
                    "<kbd>Ctrl</kbd>+<kbd>⇧</kbd>+<kbd>C</kbd> Copy"
                    "</div>"
                )

            # Export file download (hidden until triggered)
            export_file_output = gr.File(label="📥 Download Export", visible=False)

            gr.Markdown(
                value="*💡 Tip: Switch analysis mode for specialised outputs "
                      "(Timeline, Summary, Inconsistencies, Medications)*",
                elem_classes=["mode-hint"],
            )

            # ── RAG System Log viewer — directly under Chat question field ──
            gr.Markdown("## 📜 RAG System Log")
            rag_log_viewer = gr.Code(
                label="System Logs",
                language="shell",
                value="",
                interactive=False,
                lines=30,
                elem_classes=["log-console"]
            )

    # ── Event wiring ──────────────────────────────────────────

    # Infrastructure controls
    rag_start_btn.click(
        start_rag_infra_ui_wrapper,
        outputs=[rag_infra_msg, rag_infra_status, rag_log_viewer],
    )
    rag_stop_btn.click(
        stop_rag_infra_ui_wrapper,
        outputs=[rag_infra_msg, rag_infra_status, rag_log_viewer],
    )

    # Corpus stats refresh
    refresh_corpus_btn.click(
        refresh_corpus_display,
        outputs=[corpus_stats],
    )

    # Run dropdown refresh (also triggered on accordion open)
    refresh_corpus_btn.click(
        refresh_runs_dropdown,
        outputs=[run_selector],
    )

    # Also refresh active case choices on corpus refresh
    def _refresh_case_selector():
        choices = _get_indexed_run_choices()
        return gr.update(choices=choices)

    def _refresh_target_case_choices():
        choices = [("🆕 Create New Case", "new")] + [choice for choice in _get_indexed_run_choices() if choice[1] != ""]
        return gr.update(choices=choices, value="new")

    def toggle_new_case_textbox(choice):
        return gr.update(visible=(choice == "new"))

    target_case_dropdown.change(
        toggle_new_case_textbox,
        inputs=[target_case_dropdown],
        outputs=[new_case_name]
    )

    refresh_corpus_btn.click(
        _refresh_case_selector,
        outputs=[active_case_selector],
    )
    refresh_corpus_btn.click(
        _refresh_target_case_choices,
        outputs=[target_case_dropdown],
    )

    # Index single run
    index_run_btn.click(
        index_run_ui_wrapper,
        inputs=[run_selector],
        outputs=[index_status, rag_log_viewer],
    ).then(
        refresh_corpus_display,
        outputs=[corpus_stats],
    ).then(
        _refresh_case_selector,
        outputs=[active_case_selector],
    ).then(
        _refresh_target_case_choices,
        outputs=[target_case_dropdown],
    )

    # Index all runs
    index_all_btn.click(
        index_all_runs_ui_wrapper,
        outputs=[index_status, rag_log_viewer],
    ).then(
        refresh_corpus_display,
        outputs=[corpus_stats],
    ).then(
        _refresh_case_selector,
        outputs=[active_case_selector],
    ).then(
        _refresh_target_case_choices,
        outputs=[target_case_dropdown],
    )

    # Upload external markdown
    upload_md_btn.click(
        upload_and_index_markdown_ui_wrapper,
        inputs=[external_md_uploader, target_case_dropdown, new_case_name],
        outputs=[upload_status, rag_log_viewer],
    ).then(
        refresh_corpus_display,
        outputs=[corpus_stats],
    ).then(
        _refresh_active_case_after_upload,
        outputs=[active_case_selector],
    ).then(
        _refresh_target_case_choices,
        outputs=[target_case_dropdown],
    )

    # ── Active Case Selector → update banner + populate filters ──

    def on_case_selected(run_id):
        """When a case is selected, update the banner and populate filter dropdowns."""
        # Find the label for this run_id
        choices = _get_indexed_run_choices()
        label = None
        for lbl, rid in choices:
            if rid == run_id:
                label = lbl
                break

        banner_html = _get_case_banner_html(label)

        # Populate authors
        author_choices = [("All Authors", "")]
        if run_id:
            try:
                from rag.db import get_authors_for_run
                authors = get_authors_for_run(run_id)
                for a in authors:
                    author_choices.append((a, a))
            except Exception:
                pass

        # Populate date range hints
        date_from_val = ""
        date_to_val = ""
        if run_id:
            try:
                from rag.db import get_date_range_for_run
                dr = get_date_range_for_run(run_id)
                if dr.get("earliest"):
                    date_from_val = dr["earliest"]
                if dr.get("latest"):
                    date_to_val = dr["latest"]
            except Exception:
                pass

        return (
            banner_html,
            gr.update(choices=author_choices, value=""),
            gr.update(value=date_from_val, placeholder=f"From: {date_from_val}" if date_from_val else "YYYY-MM-DD"),
            gr.update(value=date_to_val, placeholder=f"To: {date_to_val}" if date_to_val else "YYYY-MM-DD"),
        )

    active_case_selector.change(
        on_case_selected,
        inputs=[active_case_selector],
        outputs=[active_case_banner, filter_author, filter_date_from, filter_date_to],
    )

    # ── Chat submission with filters ──

    def user_message_submit(message, history):
        """Append user message to chat history and clear input."""
        if not message or not message.strip():
            return "", history
        history = history or []
        history.append({"role": "user", "content": message})
        return "", history

    def bot_respond(history, mode, model_url, model_name, top_k,
                    active_case, doc_type, author, date_from, date_to):
        """Generate bot response with streaming, applying all active filters."""
        if not history:
            yield history, get_rag_logs()
            return

        last_user_msg = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = extract_text_content(msg.get("content"))
                break

        if not last_user_msg:
            yield history, get_rag_logs()
            return

        # Convert history to pairs for the analyze function
        chat_pairs = []
        for msg in history[:-1]:  # Exclude the last user message
            role = msg.get("role")
            content = extract_text_content(msg.get("content", ""))
            chat_pairs.append({"role": role, "content": content})

        # Resolve filter values (empty string → None)
        run_id_f = active_case if active_case else None
        doc_type_f = doc_type if doc_type else None
        author_f = author if author else None
        date_from_f = date_from.strip() if date_from and date_from.strip() else None
        date_to_f = date_to.strip() if date_to and date_to.strip() else None

        # Stream the response
        history.append({"role": "assistant", "content": ""})

        log_to_rag(f"RAG query received: '{last_user_msg}'")
        filter_desc = []
        if run_id_f:
            filter_desc.append(f"case={run_id_f[:8]}...")
        if doc_type_f:
            filter_desc.append(f"type={doc_type_f}")
        if author_f:
            filter_desc.append(f"author={author_f}")
        if date_from_f:
            filter_desc.append(f"from={date_from_f}")
        if date_to_f:
            filter_desc.append(f"to={date_to_f}")
        if filter_desc:
            log_to_rag(f"Active filters: {', '.join(filter_desc)}")
        log_to_rag(f"Retrieving top {top_k} matching chunks from vector database...")

        try:
            mode_map = {
                "💬 Free Q&A": "free_qa",
                "📅 Timeline Generator": "timeline",
                "🏥 Injury Summary": "injury_summary",
                "🔍 Inconsistency Finder": "inconsistency_finder",
                "💊 Medication Tracker": "medication_tracker",
            }
            mode_key = mode_map.get(mode, "free_qa")

            from rag.analyzer import analyze

            partial = ""
            for chunk in analyze(
                query=last_user_msg,
                mode=mode_key,
                server_url=model_url,
                model_name=model_name,
                top_k=int(top_k),
                chat_history=chat_pairs,
                run_id_filter=run_id_f,
                doc_type_filter=doc_type_f,
                author_filter=author_f,
                date_from=date_from_f,
                date_to=date_to_f,
                stream=True,
            ):
                partial += chunk
                history[-1]["content"] = partial
                yield history, get_rag_logs()

            log_to_rag("LLM response generation finished successfully.")

        except Exception as e:
            history[-1]["content"] = f"⚠️ Error: {str(e)}"
            log_to_rag(f"RAG query error: {str(e)}")
            yield history, get_rag_logs()

    _bot_inputs = [
        chatbot, analysis_mode, analysis_model_url, analysis_model_name,
        retrieval_top_k, active_case_selector, filter_doc_type,
        filter_author, filter_date_from, filter_date_to,
    ]

    chat_input.submit(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=_bot_inputs,
        outputs=[chatbot, rag_log_viewer],
    )

    chat_submit_btn.click(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=_bot_inputs,
        outputs=[chatbot, rag_log_viewer],
    )

    # ── Analysis settings save ──

    def save_analysis_settings(url, name, top_k, emb_model):
        try:
            from settings_manager import save_settings
            settings = load_settings()
            settings.update({
                "analysis_server_url": url,
                "analysis_model_name": name,
                "retrieval_top_k": int(top_k),
                "embedding_model": emb_model,
            })
            save_settings(settings)
            return "✅ Analysis configuration saved successfully."
        except Exception as e:
            return f"❌ Error: {e}"

    save_analysis_btn.click(
        save_analysis_settings,
        inputs=[analysis_model_url, analysis_model_name, retrieval_top_k, embedding_model],
        outputs=[analysis_config_status]
    )

    clear_chat_btn.click(
        lambda: [],
        outputs=[chatbot],
    )

    # ── Export handlers ──

    def _do_export_md(history, mode, active_case):
        try:
            from rag_export import export_chat_markdown
            choices = _get_indexed_run_choices()
            case_label = "All Cases"
            for lbl, rid in choices:
                if rid == active_case:
                    case_label = lbl
                    break
            path = export_chat_markdown(history, mode, case_label)
            if path:
                log_to_rag(f"Exported chat as Markdown: {os.path.basename(path)}")
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)
        except Exception as e:
            log_to_rag(f"Export error: {e}")
            return gr.update(visible=False)

    def _do_export_txt(history, mode, active_case):
        try:
            from rag_export import export_chat_text
            choices = _get_indexed_run_choices()
            case_label = "All Cases"
            for lbl, rid in choices:
                if rid == active_case:
                    case_label = lbl
                    break
            path = export_chat_text(history, mode, case_label)
            if path:
                log_to_rag(f"Exported chat as Text: {os.path.basename(path)}")
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)
        except Exception as e:
            log_to_rag(f"Export error: {e}")
            return gr.update(visible=False)

    def _do_export_csv(history, active_case):
        try:
            from rag_export import export_timeline_csv
            choices = _get_indexed_run_choices()
            case_label = "All Cases"
            for lbl, rid in choices:
                if rid == active_case:
                    case_label = lbl
                    break
            path = export_timeline_csv(history, case_label)
            if path:
                log_to_rag(f"Exported timeline as CSV: {os.path.basename(path)}")
                return gr.update(value=path, visible=True)
            log_to_rag("CSV export: no table data found in chat history.")
            return gr.update(visible=False)
        except Exception as e:
            log_to_rag(f"Export error: {e}")
            return gr.update(visible=False)

    export_md_btn.click(
        _do_export_md,
        inputs=[chatbot, analysis_mode, active_case_selector],
        outputs=[export_file_output],
    )
    export_txt_btn.click(
        _do_export_txt,
        inputs=[chatbot, analysis_mode, active_case_selector],
        outputs=[export_file_output],
    )
    export_csv_btn.click(
        _do_export_csv,
        inputs=[chatbot, active_case_selector],
        outputs=[export_file_output],
    )

    return {
        "rag_infra_status": rag_infra_status,
        "chatbot": chatbot,
        "analysis_mode": analysis_mode,
        "corpus_stats": corpus_stats,
        "rag_log_viewer": rag_log_viewer,
        "active_case_selector": active_case_selector,
        "target_case_dropdown": target_case_dropdown,
        "refresh_corpus_btn": refresh_corpus_btn,
        "refresh_fn": _refresh_case_selector,
        "save_analysis_btn": save_analysis_btn,
        "analysis_model_url": analysis_model_url,
        "analysis_model_name": analysis_model_name,
        "retrieval_top_k": retrieval_top_k,
        "embedding_model": embedding_model,
        "analysis_config_status": analysis_config_status,
    }


def build_analysis_ui():
    """Build the Gradio UI components for the RAG analysis section (for backwards compatibility/testing).

    Returns:
        Dict of component references.
    """
    gr.HTML("<hr class='section-divider'>")
    gr.HTML(
        "<h1 class='gradient-title' style='margin:0; font-size:1.8rem;'>🧠 Document Analysis (RAG)</h1>"
        "<p style='color:#9ca3af; margin:4px 0 12px 0; font-size:0.95rem;'>"
        "Query, summarise, and cross-reference indexed medicolegal documents using local LLMs</p>"
    )

    with gr.Tabs() as rag_tabs:
        with gr.Tab("📊 Case Dashboard", id="tab-dashboard") as tab_dashboard:
            dash = build_case_dashboard_ui()
            dashboard_html = dash["dashboard_html"]
            dashboard_delete_selector = dash["dashboard_delete_selector"]
            dashboard_status = dash["dashboard_status"]
            _refresh_dashboard = dash["refresh_fn"]

        with gr.Tab("💬 Analysis", id="tab-analysis") as tab_analysis:
            chat = build_rag_chat_ui()
            rag_infra_status = chat["rag_infra_status"]
            chatbot = chat["chatbot"]
            analysis_mode = chat["analysis_mode"]
            corpus_stats = chat["corpus_stats"]
            rag_log_viewer = chat["rag_log_viewer"]
            active_case_selector = chat["active_case_selector"]
            target_case_dropdown = chat.get("target_case_dropdown")
            _refresh_case_selector = chat["refresh_fn"]

    # Automatically refresh dashboard on tab select
    tab_dashboard.select(
        _refresh_dashboard,
        outputs=[dashboard_html, dashboard_delete_selector, dashboard_status]
    )

    def _refresh_analysis_tab_selectors():
        choices = _get_indexed_run_choices()
        target_choices = [("🆕 Create New Case", "new")] + [choice for choice in choices if choice[1] != ""]
        return gr.update(choices=choices), gr.update(choices=target_choices, value="new")

    # Automatically refresh case choices on tab select
    tab_analysis.select(
        _refresh_analysis_tab_selectors,
        outputs=[active_case_selector, target_case_dropdown]
    )

    return {
        "rag_infra_status": rag_infra_status,
        "chatbot": chatbot,
        "analysis_mode": analysis_mode,
        "corpus_stats": corpus_stats,
        "rag_log_viewer": rag_log_viewer,
        "active_case_selector": active_case_selector,
        "rag_tabs": rag_tabs,
    }


