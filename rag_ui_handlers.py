import os

import gradio as gr

import rag_ui_state
from rag_ui_state import extract_text_content, get_available_runs, get_rag_logs, log_to_rag
from settings_manager import load_settings


# Forward declarations / imports to avoid circular reference issues
def _get_indexed_run_choices():
    from rag_ui_dashboard import _get_indexed_run_choices as get_choices

    return get_choices()


def _build_dashboard_html():
    from rag_ui_dashboard import _build_dashboard_html as build_html

    return build_html()


def index_run(run_dir, progress=None, force=False):
    """Index a single OCR run into the RAG system."""
    from indexing_service import CorpusIndexingService

    yield from CorpusIndexingService.index_run(run_dir, force=force)


def index_all_runs(get_available_runs_fn=None, force=False):
    """Index all available OCR runs into the RAG corpus."""
    from indexing_service import CorpusIndexingService

    yield from CorpusIndexingService.index_all_runs(get_available_runs_fn, force=force)


def start_rag_infra_ui():
    """Start RAG infrastructure and return status."""
    try:
        from rag_infra_manager import get_rag_status_html, start_and_init_rag

        success, msg = start_and_init_rag()
        status_html = get_rag_status_html()
        return msg, status_html
    except Exception as e:
        return f"❌ Error: {e}", "<span class='badge-failed'>Error</span>"


def stop_rag_infra_ui():
    """Stop RAG infrastructure and return status."""
    try:
        from rag_infra_manager import get_rag_status_html, stop_rag_infrastructure

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


def get_corpus_info():
    """Get formatted corpus statistics for display."""
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


def refresh_corpus_display():
    """Refresh the corpus statistics display."""
    return get_corpus_info()


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
    if not run_dir:
        runs = get_available_runs()
        if runs:
            run_dir = runs[0][1]
        else:
            yield "⚠️ No available OCR run directory selected.", get_rag_logs()
            return
    log_to_rag(f"Initiated manual indexing for run directory: {run_dir}")
    for update in index_run(run_dir, force=True):
        if not update.startswith("[PROGRESS:"):
            log_to_rag(update)
        accumulated_status += update
        yield accumulated_status, get_rag_logs()


def index_all_runs_ui_wrapper():
    accumulated_status = ""
    log_to_rag("Initiated bulk indexing for all runs")
    for update in index_all_runs(force=True):
        if not update.startswith("[PROGRESS:"):
            log_to_rag(update)
        accumulated_status += update
        yield accumulated_status, get_rag_logs()


def upload_and_index_markdown(files, case_option, new_case_name):
    from indexing_service import CorpusIndexingService

    yield from CorpusIndexingService.add_markdown_to_case(files, case_option, new_case_name)
    rag_ui_state.LAST_CREATED_RUN_ID = CorpusIndexingService.last_created_run_id


def upload_and_index_markdown_ui_wrapper(files, case_option, new_case_name):
    accumulated_status = ""
    log_to_rag("Initiated external markdown upload and indexing")
    for update in upload_and_index_markdown(files, case_option, new_case_name):
        accumulated_status += update
        log_to_rag(update)
        yield accumulated_status, get_rag_logs()


def user_message_submit(message, history):
    """Append user message to chat history and clear input."""
    if not message or not message.strip():
        return "", history
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history


def bot_respond(
    history,
    mode,
    model_url,
    model_name,
    top_k,
    active_case,
    doc_type,
    author,
    date_from,
    date_to,
    use_reranker_val=None,
    reranker_model_val=None,
    reranker_device_val=None,
    progress=None,
):
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
    history.append(
        {"role": "assistant", "content": "🔍 Retrieving and reranking matching chunks..."}
    )
    yield history, get_rag_logs()

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
        from rag.analyzer import ANALYSIS_MODE_MAP

        mode_key = ANALYSIS_MODE_MAP.get(mode, "free_qa")

        from rag.analyzer import analyze

        def retrieve_progress(pct, desc):
            if progress is not None:
                progress(pct * 0.7, desc=desc)

        if progress is not None:
            progress(0.0, desc="Initiating RAG pipeline...")

        partial = ""
        first_chunk = True
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
            use_reranker=use_reranker_val,
            reranker_model=reranker_model_val,
            reranker_device=reranker_device_val,
            progress_callback=retrieve_progress,
        ):
            if first_chunk:
                first_chunk = False
                partial = ""
                if progress is not None:
                    progress(0.8, desc="Synthesizing response...")
            partial += chunk
            history[-1]["content"] = partial
            yield history, get_rag_logs()

        if progress is not None:
            progress(1.0, desc="Finished.")

        log_to_rag("LLM response generation finished successfully.")

    except Exception as e:
        history[-1]["content"] = f"⚠️ Error: {str(e)}"
        log_to_rag(f"RAG query error: {str(e)}")
        yield history, get_rag_logs()


def save_analysis_settings(
    url,
    name,
    top_k,
    emb_model,
    use_reranker_val=None,
    reranker_model_val=None,
    reranker_device_val=None,
):
    try:
        from settings_manager import save_settings

        settings = load_settings()
        if use_reranker_val is None:
            use_reranker_val = settings.get("use_reranker", True)
        if reranker_model_val is None:
            reranker_model_val = settings.get("reranker_model", "BAAI/bge-reranker-large")
        if reranker_device_val is None:
            reranker_device_val = settings.get("reranker_device", "cuda")

        settings.update(
            {
                "analysis_server_url": url,
                "analysis_model_name": name,
                "retrieval_top_k": int(top_k),
                "embedding_model": emb_model,
                "use_reranker": bool(use_reranker_val),
                "reranker_model": reranker_model_val,
                "reranker_device": reranker_device_val,
            }
        )
        save_settings(settings)
        # Switching the embedding model changes vector dimensionality, which
        # invalidates every cached embedding and any query result built on the
        # previous collection. Drop them so the next query re-embeds correctly.
        try:
            from rag.cache import invalidate_all_caches

            invalidate_all_caches()
        except Exception:
            pass
        return "✅ Analysis configuration saved successfully."
    except Exception as e:
        return f"❌ Error: {e}"


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


def _do_export_docx(history, mode, active_case):
    try:
        from rag_export import export_chat_docx

        choices = _get_indexed_run_choices()
        case_label = "All Cases"
        for lbl, rid in choices:
            if rid == active_case:
                case_label = lbl
                break
        path = export_chat_docx(history, mode, case_label)
        if path:
            log_to_rag(f"Exported chat as DOCX: {os.path.basename(path)}")
            return gr.update(value=path, visible=True)
        return gr.update(visible=False)
    except Exception as e:
        log_to_rag(f"Export error: {e}")
        return gr.update(visible=False)


def _do_export_timeline_docx(history, active_case):
    try:
        from rag_export import export_timeline_docx

        choices = _get_indexed_run_choices()
        case_label = "All Cases"
        for lbl, rid in choices:
            if rid == active_case:
                case_label = lbl
                break
        path = export_timeline_docx(history, case_label)
        if path:
            log_to_rag(f"Exported timeline as DOCX: {os.path.basename(path)}")
            return gr.update(value=path, visible=True)
        log_to_rag("DOCX timeline export: no table data found in chat history.")
        return gr.update(visible=False)
    except Exception as e:
        log_to_rag(f"Export error: {e}")
        return gr.update(visible=False)


def save_embedding_settings(model_name, device, chunk_size, chunk_overlap, batch_size):
    try:
        from settings_manager import save_settings

        new_settings = {
            "embedding_model": model_name,
            "embedding_device": device,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "embedding_batch_size": int(batch_size),
        }
        save_settings(new_settings)
        log_to_rag(
            f"Updated embedding pipeline settings: model={model_name}, device={device}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, batch_size={batch_size}"
        )
        return "✅ Embedding pipeline configuration saved successfully!"
    except Exception as e:
        log_to_rag(f"Failed to save embedding configuration: {e}")
        return f"❌ Save error: {e}"


def get_embedding_pipeline_info_html():
    try:
        from rag.embedding import get_collection_info, get_collection_name
        from settings_manager import load_settings

        settings = load_settings()
        model_name = settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
        device = settings.get("embedding_device", "auto")
        col_name = get_collection_name(model_name)
        info = get_collection_info(model_name)

        active_device = device
        if device == "auto" or not device:
            try:
                import torch

                active_device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
            except Exception:
                active_device = "CPU"

        html = f"""
        <div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 0.82rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                <span style='color: #94a3b8;'>Active Compute Engine:</span>
                <span style='font-weight: 600; color: #34d399;'>⚡ {str(active_device).upper()}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                <span style='color: #94a3b8;'>Qdrant Collection:</span>
                <span style='font-family: monospace; color: #cbd5e1;'>{col_name}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                <span style='color: #94a3b8;'>Indexed Vectors:</span>
                <span style='font-weight: 600; color: #60a5fa;'>{info.get("points_count", 0)} points</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: #94a3b8;'>Collection Status:</span>
                <span style='color: #a7f3d0;'>{str(info.get("status", "unknown")).upper()}</span>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return (
            f"<div style='font-size:0.8rem; color:#ef4444;'>Error loading vector metrics: {e}</div>"
        )
