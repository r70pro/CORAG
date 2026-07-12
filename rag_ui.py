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

def build_analysis_ui():
    """Build the Gradio UI components for the RAG analysis section.

    Returns:
        Dict of component references for event wiring in app.py.
    """
    settings = load_settings()

    gr.HTML("<hr class='section-divider'>")
    gr.HTML(
        "<h1 class='gradient-title' style='margin:0; font-size:1.8rem;'>🧠 Document Analysis (RAG)</h1>"
        "<p style='color:#9ca3af; margin:4px 0 12px 0; font-size:0.95rem;'>"
        "Query, summarise, and cross-reference indexed medicolegal documents using local LLMs</p>"
    )

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

            # Analysis settings
            with gr.Accordion("⚙️ Analysis Settings", open=False):
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
                )
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
                current_embedding_model = settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                embedding_choices = [
                    "sentence-transformers/all-MiniLM-L6-v2",
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

        # ── Right: Chat interface ──
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
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
                lines=10,
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

    # Index single run
    index_run_btn.click(
        index_run_ui_wrapper,
        inputs=[run_selector],
        outputs=[index_status, rag_log_viewer],
    ).then(
        refresh_corpus_display,
        outputs=[corpus_stats],
    )

    # Index all runs
    index_all_btn.click(
        index_all_runs_ui_wrapper,
        outputs=[index_status, rag_log_viewer],
    ).then(
        refresh_corpus_display,
        outputs=[corpus_stats],
    )

    # Chat submission
    def user_message_submit(message, history):
        """Append user message to chat history and clear input."""
        if not message or not message.strip():
            return "", history
        history = history or []
        history.append({"role": "user", "content": message})
        return "", history

    def bot_respond(history, mode, model_url, model_name, top_k):
        """Generate bot response with streaming."""
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

        # Stream the response
        history.append({"role": "assistant", "content": ""})

        log_to_rag(f"RAG query received: '{last_user_msg}'")
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

    chat_input.submit(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=[chatbot, analysis_mode, analysis_model_url, analysis_model_name, retrieval_top_k],
        outputs=[chatbot, rag_log_viewer],
    )

    chat_submit_btn.click(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=[chatbot, analysis_mode, analysis_model_url, analysis_model_name, retrieval_top_k],
        outputs=[chatbot, rag_log_viewer],
    )

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

    # Return component references for external use
    return {
        "rag_infra_status": rag_infra_status,
        "chatbot": chatbot,
        "analysis_mode": analysis_mode,
        "corpus_stats": corpus_stats,
        "rag_log_viewer": rag_log_viewer,
    }
