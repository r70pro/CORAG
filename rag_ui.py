"""
RAG Analysis UI — Gradio components for the document analysis tab.

Slimmed layout builder module. All business logic and handlers have been
extracted to sub-modules. Exposes all functions and globals via a dynamic
module class to guarantee full compatibility with the existing test suites.
"""

import sys
import types

import gradio as gr

import rag_ui_dashboard  # noqa: F401
import rag_ui_handlers  # noqa: F401

# Import state, handlers, and dashboard components to delegate calls
import rag_ui_state  # noqa: F401

# Re-expose RAG logs and shared state dynamically via RagUiModule below
from rag_ui_state import extract_text_content, get_rag_logs, log_to_rag  # noqa: F401
from settings_manager import WORKSPACE_DIR, load_settings  # noqa: F401
from settings_manager import get_available_runs as _sm_get_available_runs


def get_available_runs(workspace_dir: str | None = None):
    """Canonical forwarding wrapper for rag_ui namespace."""
    import sys

    rag_ui_mod = sys.modules.get("rag_ui")
    ws_override = workspace_dir
    if ws_override is None and rag_ui_mod is not None:
        sm_mod = sys.modules.get("settings_manager")
        sm_ws = getattr(sm_mod, "WORKSPACE_DIR", None)
        rag_ws = getattr(rag_ui_mod, "WORKSPACE_DIR", None)
        if rag_ws is not None and rag_ws != sm_ws:
            ws_override = rag_ws
    return _sm_get_available_runs(workspace_dir=ws_override)


# --- Forwarding wrappers to keep functions in rag_ui namespace for patching ---


def index_run(run_dir, progress=None, force=False):
    yield from rag_ui_handlers.index_run(run_dir, progress, force=force)


def index_all_runs(get_available_runs_fn=None, force=False):
    import sys

    rag_ui = sys.modules[__name__]
    if get_available_runs_fn is None:
        get_available_runs_fn = rag_ui.get_available_runs
    yield from rag_ui_handlers.index_all_runs(get_available_runs_fn, force=force)


def start_rag_infra_ui():
    return rag_ui_handlers.start_rag_infra_ui()


def stop_rag_infra_ui():
    return rag_ui_handlers.stop_rag_infra_ui()


def refresh_rag_status():
    return rag_ui_handlers.refresh_rag_status()


def refresh_runs_dropdown():
    import sys

    rag_ui = sys.modules[__name__]
    runs = rag_ui.get_available_runs()
    if runs:
        return gr.update(choices=runs, value=runs[0][1])
    return gr.update(choices=[], value=None)


def get_corpus_info():
    return rag_ui_handlers.get_corpus_info()


def refresh_corpus_display():
    import sys

    rag_ui = sys.modules[__name__]
    return rag_ui.get_corpus_info()


def start_rag_infra_ui_wrapper():
    import sys

    rag_ui = sys.modules[__name__]
    rag_ui.log_to_rag("Starting RAG infrastructure services...")
    msg, status_html = rag_ui.start_rag_infra_ui()
    rag_ui.log_to_rag(f"Start infrastructure result: {msg}")
    return msg, status_html, rag_ui.get_rag_logs()


def stop_rag_infra_ui_wrapper():
    import sys

    rag_ui = sys.modules[__name__]
    rag_ui.log_to_rag("Stopping RAG infrastructure services...")
    msg, status_html = rag_ui.stop_rag_infra_ui()
    rag_ui.log_to_rag(f"Stop infrastructure result: {msg}")
    return msg, status_html, rag_ui.get_rag_logs()


def parse_progress_state(accumulated_status: str):
    import re

    stages = [
        {
            "id": "prepare",
            "label": "📁 Creating case & preparing storage",
            "status": "pending",
            "details": "",
        },
        {
            "id": "upload",
            "label": "☁️ Uploading files to object store",
            "status": "pending",
            "details": "",
        },
        {"id": "chunk", "label": "🧩 Chunking documents", "status": "pending", "details": ""},
        {
            "id": "embed",
            "label": "🧠 Embedding chunks (Dense vector)",
            "status": "pending",
            "details": "",
        },
        {
            "id": "index",
            "label": "⚡ Indexing into Qdrant & Database",
            "status": "pending",
            "details": "",
        },
    ]

    # Analyze accumulated_status line by line
    lines = accumulated_status.split("\n")

    total_chunks = 0
    embed_current = 0
    embed_total = 0
    index_current = 0
    index_total = 0

    active_stage = "prepare"

    for line in lines:
        if "Initiated" in line or "Creating new case" in line or "Adding to existing case" in line:
            active_stage = "prepare"
        if "Copied" in line or "Registering case metadata" in line:
            active_stage = "prepare"
        if "Uploaded" in line or "Uploading to object storage" in line:
            active_stage = "upload"
        if "Created" in line or "Chunking documents" in line:
            active_stage = "chunk"
            if "Created" in line:
                match = re.search(r"Created \*\*?(\d+)\*\*? chunk", line)
                if match:
                    total_chunks += int(match.group(1))
        if "Embedding and indexing" in line or "Embedding chunks" in line:
            active_stage = "embed"
            match = re.search(r"indexing \*\*?(\d+)\*\*? chunks", line)
            if match:
                embed_total = int(match.group(1))
                index_total = embed_total
        if "[PROGRESS:embedding:" in line:
            active_stage = "embed"
            match = re.search(r"\[PROGRESS:embedding:(\d+)/(\d+)\]", line)
            if match:
                embed_current = int(match.group(1))
                embed_total = int(match.group(2))
        if "[PROGRESS:indexing:" in line:
            active_stage = "index"
            match = re.search(r"\[PROGRESS:indexing:(\d+)/(\d+)\]", line)
            if match:
                index_current = int(match.group(1))
                index_total = int(match.group(2))
        if "Successfully indexed" in line or "Successfully uploaded" in line or "✅ Done." in line:
            active_stage = "done"

    # Set statuses based on active_stage
    stage_sequence = ["prepare", "upload", "chunk", "embed", "index"]
    try:
        active_idx = stage_sequence.index(active_stage)
    except ValueError:
        active_idx = len(stage_sequence)

    for idx, stage in enumerate(stages):
        if idx < active_idx:
            stage["status"] = "success"
        elif idx == active_idx:
            stage["status"] = "running"
        else:
            stage["status"] = "pending"

    if active_stage == "done":
        for s in stages:
            s["status"] = "success"

    # Add details
    if embed_total > 0:
        stages[3]["details"] = f"{embed_current}/{embed_total} chunks"
        if active_stage == "embed":
            stages[3]["progress"] = int((embed_current / embed_total) * 100)
    if index_total > 0:
        stages[4]["details"] = f"{index_current}/{index_total} chunks"
        if active_stage == "index":
            stages[4]["progress"] = int((index_current / index_total) * 100)
            stages[3]["status"] = "success"  # ensure previous is marked success

    return stages, active_stage


def make_indexing_progress_card(stages, active_stage):
    html = """
    <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; font-family: "Outfit", sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.3);'>
        <h3 style='margin-top:0; margin-bottom: 15px; color: #f3f4f6; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;'>
            <span style='animation: pulse 2s infinite;'>⏳</span> RAG Indexing Progress
        </h3>
        <div style='display: flex; flex-direction: column; gap: 12px;'>
    """

    for s in stages:
        badge_style = ""
        icon = ""
        desc_color = "#9ca3af"

        if s["status"] == "success":
            badge_style = "background-color: #064e3b; color: #34d399;"
            icon = "✅"
            desc_color = "#34d399"
        elif s["status"] == "running":
            badge_style = "background-color: #1e3a8a; color: #60a5fa; animation: pulse 2s infinite;"
            icon = "🔄"
            desc_color = "#e2e8f0"
        else:  # pending
            badge_style = "background-color: #1e293b; color: #94a3b8;"
            icon = "💤"
            desc_color = "#4b5563"

        details_str = (
            f" <span style='font-size:0.8rem; font-weight:normal; color:#94a3b8;'>({s['details']})</span>"
            if s["details"]
            else ""
        )

        progress_bar_html = ""
        if s["status"] == "running" and "progress" in s:
            pct = s["progress"]
            progress_bar_html = f"""
            <div style='width: 100%; background: #1e293b; border-radius: 4px; height: 6px; margin-top: 6px; overflow: hidden;'>
                <div style='width: {pct}%; height: 100%; background: linear-gradient(90deg, #60a5fa, #3b82f6); transition: width 0.3s ease;'></div>
            </div>
            """

        html += f"""
        <div style='display: flex; flex-direction: column;'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <span style='font-weight: 500; color: {desc_color}; display: flex; align-items: center; gap: 8px;'>
                    <span style='font-size: 1.1rem;'>{icon}</span> {s["label"]} {details_str}
                </span>
                <span style='font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px; {badge_style}'>
                    {s["status"].upper()}
                </span>
            </div>
            {progress_bar_html}
        </div>
        """

    html += """
        </div>
    </div>
    """
    return html


def index_run_ui_wrapper(run_dir, progress=gr.Progress()):
    import sys

    rag_ui = sys.modules[__name__]
    accumulated_status = ""
    rag_ui.log_to_rag(f"Initiated manual indexing for run directory: {run_dir}")

    if progress is not None:
        progress(0.0, desc="Starting manual indexing...")

    for update in rag_ui.index_run(run_dir, force=True):
        if not update.startswith("[PROGRESS:"):
            rag_ui.log_to_rag(update)
        accumulated_status += update

        stages, active = parse_progress_state(accumulated_status)
        progress_html = make_indexing_progress_card(stages, active)

        if progress is not None:
            if active == "prepare":
                progress(0.1, desc="Preparing indexing...")
            elif active == "upload":
                progress(0.2, desc="Uploading to storage...")
            elif active == "chunk":
                progress(0.4, desc="Chunking documents...")
            elif active == "embed":
                embed_pct = stages[3].get("progress", 0) / 100.0
                progress(
                    0.4 + 0.4 * embed_pct, desc=f"Embedding chunks ({stages[3]['details']})..."
                )
            elif active == "index":
                index_pct = stages[4].get("progress", 0) / 100.0
                progress(0.8 + 0.2 * index_pct, desc=f"Indexing chunks ({stages[4]['details']})...")
            elif active == "done":
                progress(1.0, desc="Indexing completed successfully!")

        yield progress_html, rag_ui.get_rag_logs()


def index_all_runs_ui_wrapper(progress=gr.Progress()):
    import sys

    rag_ui = sys.modules[__name__]
    accumulated_status = ""
    rag_ui.log_to_rag("Initiated bulk indexing for all runs")

    if progress is not None:
        progress(0.0, desc="Starting bulk indexing...")

    for update in rag_ui.index_all_runs(
        get_available_runs_fn=rag_ui.get_available_runs, force=True
    ):
        if not update.startswith("[PROGRESS:"):
            rag_ui.log_to_rag(update)
        accumulated_status += update

        stages, active = parse_progress_state(accumulated_status)
        progress_html = make_indexing_progress_card(stages, active)

        if progress is not None:
            if active == "prepare":
                progress(0.1, desc="Preparing bulk indexing...")
            elif active == "upload":
                progress(0.2, desc="Uploading to storage...")
            elif active == "chunk":
                progress(0.4, desc="Chunking documents...")
            elif active == "embed":
                embed_pct = stages[3].get("progress", 0) / 100.0
                progress(
                    0.4 + 0.4 * embed_pct, desc=f"Embedding chunks ({stages[3]['details']})..."
                )
            elif active == "index":
                index_pct = stages[4].get("progress", 0) / 100.0
                progress(0.8 + 0.2 * index_pct, desc=f"Indexing chunks ({stages[4]['details']})...")
            elif active == "done":
                progress(1.0, desc="Bulk indexing completed successfully!")

        yield progress_html, rag_ui.get_rag_logs()


def upload_and_index_markdown(files, case_option, new_case_name):
    yield from rag_ui_handlers.upload_and_index_markdown(files, case_option, new_case_name)


def upload_and_index_markdown_ui_wrapper(files, case_option, new_case_name, progress=gr.Progress()):
    import sys

    rag_ui = sys.modules[__name__]
    accumulated_status = ""
    rag_ui.log_to_rag("Initiated external markdown upload and indexing")

    if progress is not None:
        progress(0.0, desc="Preparing upload and case directories...")

    for update in rag_ui.upload_and_index_markdown(files, case_option, new_case_name):
        if not update.startswith("[PROGRESS:"):
            rag_ui.log_to_rag(update)
        accumulated_status += update

        stages, active = parse_progress_state(accumulated_status)
        progress_html = make_indexing_progress_card(stages, active)

        if progress is not None:
            if active == "prepare":
                progress(0.1, desc="Creating case & prepping storage...")
            elif active == "upload":
                progress(0.2, desc="Uploading markdown files to object storage...")
            elif active == "chunk":
                progress(0.4, desc="Chunking markdown files...")
            elif active == "embed":
                embed_pct = stages[3].get("progress", 0) / 100.0
                progress(
                    0.4 + 0.4 * embed_pct, desc=f"Embedding chunks ({stages[3]['details']})..."
                )
            elif active == "index":
                index_pct = stages[4].get("progress", 0) / 100.0
                progress(0.8 + 0.2 * index_pct, desc=f"Indexing chunks ({stages[4]['details']})...")
            elif active == "done":
                progress(1.0, desc="Upload and indexing completed successfully!")

        yield progress_html, rag_ui.get_rag_logs()


def user_message_submit(message, history):
    return rag_ui_handlers.user_message_submit(message, history)


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
    progress=gr.Progress(),
):
    yield from rag_ui_handlers.bot_respond(
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
        use_reranker_val,
        reranker_model_val,
        reranker_device_val,
        progress=progress,
    )


def save_analysis_settings(
    url,
    name,
    top_k,
    emb_model,
    use_reranker_val=None,
    reranker_model_val=None,
    reranker_device_val=None,
):
    return rag_ui_handlers.save_analysis_settings(
        url, name, top_k, emb_model, use_reranker_val, reranker_model_val, reranker_device_val
    )


def _do_export_md(history, mode, active_case):
    return rag_ui_handlers._do_export_md(history, mode, active_case)


def _do_export_txt(history, mode, active_case):
    return rag_ui_handlers._do_export_txt(history, mode, active_case)


def _do_export_csv(history, active_case):
    return rag_ui_handlers._do_export_csv(history, active_case)


def _do_export_docx(history, mode, active_case):
    return rag_ui_handlers._do_export_docx(history, mode, active_case)


def _do_export_timeline_docx(history, active_case):
    return rag_ui_handlers._do_export_timeline_docx(history, active_case)


def _build_dashboard_html():
    return rag_ui_dashboard._build_dashboard_html()


def _get_indexed_run_choices():
    return rag_ui_dashboard._get_indexed_run_choices()


def _refresh_active_case_after_upload():
    import sys

    import gradio as gr

    rag_ui = sys.modules[__name__]
    choices = rag_ui._get_indexed_run_choices()
    val = rag_ui.LAST_CREATED_RUN_ID if rag_ui.LAST_CREATED_RUN_ID else ""
    return gr.update(choices=choices, value=val)


def _get_case_banner_html(active_case_label):
    return rag_ui_dashboard._get_case_banner_html(active_case_label)


def build_case_dashboard_ui():
    return rag_ui_dashboard.build_case_dashboard_ui()


# --- Deprecated/Compatibility wrappers for test cases ---


def chat_respond(message, history, analysis_mode, analysis_model_url, analysis_model_name, top_k):
    """Legacy chat_respond wrapper for backwards compatibility with tests."""
    if not message or not message.strip():
        yield ""
        return

    chat_history = []
    if history:
        for user_msg, assistant_msg in history:
            if user_msg:
                chat_history.append({"role": "user", "content": extract_text_content(user_msg)})
            if assistant_msg:
                chat_history.append(
                    {"role": "assistant", "content": extract_text_content(assistant_msg)}
                )

    from rag.analyzer import ANALYSIS_MODE_MAP

    mode_key = ANALYSIS_MODE_MAP.get(analysis_mode, "free_qa")

    from rag.analyzer import analyze

    try:
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


def build_analysis_ui():
    """Build the Gradio UI components for the RAG analysis section (for backwards compatibility/testing)."""
    gr.HTML("<hr class='section-divider'>")
    gr.HTML(
        "<h1 class='gradient-title inline-header-title'>🧠 Document Analysis (RAG)</h1>"
        "<p class='inline-rag-subtitle'>"
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

    tab_dashboard.select(
        _refresh_dashboard, outputs=[dashboard_html, dashboard_delete_selector, dashboard_status]
    )
    tab_dashboard.select(
        None,
        js="""() => {
            const checkboxes = document.querySelectorAll('.case-select-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = false;
                const card = cb.closest('.case-card');
                if (card) card.classList.remove('selected');
            });
            const txtEl = document.querySelector('#selected-cases-input textarea, #selected-cases-input input');
            if (txtEl) {
                txtEl.value = '';
                txtEl.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }""",
    )

    def _refresh_analysis_tab_selectors():
        choices = _get_indexed_run_choices()
        target_choices = [("🆕 Create New Case", "new")] + [
            choice for choice in choices if choice[1] != ""
        ]
        return gr.update(choices=choices), gr.update(choices=target_choices, value="new")

    tab_analysis.select(
        _refresh_analysis_tab_selectors, outputs=[active_case_selector, target_case_dropdown]
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


# --- UI Builder ---


def build_rag_chat_ui():
    """Build the Gradio UI components for RAG Analysis Chat.

    Returns:
        Dict of component references.
    """
    settings = load_settings()
    sidebar_visible = gr.State(True)

    with gr.Row():
        # ── Left sidebar: Controls ──
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]) as controls_sidebar:
            # Infrastructure
            with gr.Accordion("🔧 RAG Infrastructure", open=False):
                rag_infra_status = gr.HTML(
                    value="<span class='badge-idle'>Not checked</span>", label="Service Status"
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
                index_status = gr.HTML("")

            target_case_dropdown = gr.State("new")
            embedding_model_input = gr.State(
                settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
            )

            # Analysis settings
            with gr.Accordion("⚙️ Analysis Settings", open=False):
                analysis_model_url = gr.Textbox(
                    label="Analysis LLM Server URL",
                    value=settings.get("analysis_server_url", "http://localhost:8000/v1"),
                    placeholder="http://localhost:8000/v1",
                )
                from settings_manager import SUPPORTED_MODELS

                current_analysis_model = settings.get(
                    "analysis_model_name", "nvidia/Phi-4-reasoning-plus-NVFP4"
                )
                analysis_choices = list(SUPPORTED_MODELS)
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
                    maximum=500,
                    step=1,
                    value=settings.get("retrieval_top_k", 8),
                )

                gr.HTML("<hr class='inline-section-divider'>")
                gr.Markdown("**🔄 Cross-Encoder Reranker Settings**")

                use_reranker = gr.Checkbox(
                    label="Enable Cross-Encoder Reranking",
                    value=settings.get("use_reranker", True),
                    interactive=True,
                )

                reranker_choices = [
                    "BAAI/bge-reranker-large",
                    "BAAI/bge-reranker-base",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ]
                current_reranker_model = settings.get("reranker_model", "BAAI/bge-reranker-large")
                if current_reranker_model not in reranker_choices:
                    reranker_choices.append(current_reranker_model)

                reranker_model = gr.Dropdown(
                    label="Reranker Model Name",
                    choices=reranker_choices,
                    value=current_reranker_model,
                    interactive=True,
                    allow_custom_value=True,
                )

                reranker_device = gr.Dropdown(
                    label="Reranker Device",
                    choices=["cuda", "cpu"],
                    value=settings.get("reranker_device", "cuda"),
                    interactive=True,
                )

                save_analysis_btn = gr.Button("💾 Save Analysis Configuration", variant="secondary")
                analysis_config_status = gr.Markdown()

            # ── Search Filters ──
            with gr.Accordion("🔍 Search Filters", open=True):
                gr.Markdown("**🎯 Active Case** *(isolates queries to a single case)*")
                active_case_selector = gr.Dropdown(
                    label="Active Case",
                    choices=_get_indexed_run_choices(),
                    value="",
                    interactive=True,
                )
                gr.HTML("<hr class='inline-section-divider'>")
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
        with gr.Column(scale=5, elem_classes=["glass-panel"]):
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
                        "🧬 Causation Analysis",
                        "📈 Prognosis Analysis",
                        "🧑‍💼 Work Capacity",
                        "🩺 Treatment Planning",
                    ],
                    value="💬 Free Q&A",
                    interactive=True,
                    scale=3,
                )
                toggle_sidebar_btn = gr.Button(
                    value="⬅️ Hide Controls",
                    variant="secondary",
                    scale=1,
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
                chat_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                export_md_btn = gr.Button("📝 Export .md", variant="secondary", size="sm")
                export_txt_btn = gr.Button("📄 Export .txt", variant="secondary", size="sm")
                export_csv_btn = gr.Button("📊 Export .csv", variant="secondary", size="sm")
                export_docx_btn = gr.Button("📄 Export .docx", variant="secondary", size="sm")
                export_timeline_docx_btn = gr.Button(
                    "📊 Timeline .docx", variant="secondary", size="sm"
                )
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
                elem_classes=["log-console"],
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
        choices = [("🆕 Create New Case", "new")] + [
            choice for choice in _get_indexed_run_choices() if choice[1] != ""
        ]
        return gr.update(choices=choices, value="new")

    refresh_corpus_btn.click(
        _refresh_case_selector,
        outputs=[active_case_selector],
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
    )

    # ── Active Case Selector → update banner + populate filters ──

    def on_case_selected(run_id):
        """When a case is selected, update the banner and populate filter dropdowns."""
        choices = _get_indexed_run_choices()
        label = None
        for lbl, rid in choices:
            if rid == run_id:
                label = lbl
                break

        banner_html = _get_case_banner_html(label)

        author_choices = [("All Authors", "")]
        if run_id:
            try:
                from rag.db import get_authors_for_run

                authors = get_authors_for_run(run_id)
                for a in authors:
                    author_choices.append((a, a))
            except Exception:
                pass

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
            gr.update(
                value=date_from_val,
                placeholder=f"From: {date_from_val}" if date_from_val else "YYYY-MM-DD",
            ),
            gr.update(
                value=date_to_val, placeholder=f"To: {date_to_val}" if date_to_val else "YYYY-MM-DD"
            ),
        )

    active_case_selector.change(
        on_case_selected,
        inputs=[active_case_selector],
        outputs=[active_case_banner, filter_author, filter_date_from, filter_date_to],
    )

    # ── Chat submission with filters ──

    _bot_inputs = [
        chatbot,
        analysis_mode,
        analysis_model_url,
        analysis_model_name,
        retrieval_top_k,
        active_case_selector,
        filter_doc_type,
        filter_author,
        filter_date_from,
        filter_date_to,
        use_reranker,
        reranker_model,
        reranker_device,
    ]

    submit_event1 = chat_input.submit(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=_bot_inputs,
        outputs=[chatbot, rag_log_viewer],
    )

    submit_event2 = chat_submit_btn.click(
        user_message_submit,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot],
    ).then(
        bot_respond,
        inputs=_bot_inputs,
        outputs=[chatbot, rag_log_viewer],
    )

    def handle_chat_stop():
        log_to_rag("RAG chat and model inference stopped by user.")
        return get_rag_logs()

    chat_stop_btn.click(
        fn=handle_chat_stop,
        inputs=None,
        outputs=[rag_log_viewer],
        cancels=[submit_event1, submit_event2],
    )

    # ── Analysis settings save ──

    save_analysis_btn.click(
        save_analysis_settings,
        inputs=[
            analysis_model_url,
            analysis_model_name,
            retrieval_top_k,
            embedding_model_input,
            use_reranker,
            reranker_model,
            reranker_device,
        ],
        outputs=[analysis_config_status],
    )

    clear_chat_btn.click(
        lambda: [],
        outputs=[chatbot],
    )

    def toggle_sidebar(visible):
        new_val = not visible
        new_text = "➡️ Show Controls" if not new_val else "⬅️ Hide Controls"
        return gr.update(visible=new_val), new_text, new_val

    toggle_sidebar_btn.click(
        toggle_sidebar,
        inputs=[sidebar_visible],
        outputs=[controls_sidebar, toggle_sidebar_btn, sidebar_visible],
    )

    # ── Export handlers ──

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
    export_docx_btn.click(
        _do_export_docx,
        inputs=[chatbot, analysis_mode, active_case_selector],
        outputs=[export_file_output],
    )
    export_timeline_docx_btn.click(
        _do_export_timeline_docx,
        inputs=[chatbot, active_case_selector],
        outputs=[export_file_output],
    )

    return {
        "rag_infra_status": rag_infra_status,
        "chatbot": chatbot,
        "analysis_mode": analysis_mode,
        "corpus_stats": corpus_stats,
        "rag_log_viewer": rag_log_viewer,
        "chat_stop_btn": chat_stop_btn,
        "active_case_selector": active_case_selector,
        "target_case_dropdown": target_case_dropdown,
        "refresh_corpus_btn": refresh_corpus_btn,
        "refresh_fn": _refresh_case_selector,
        "save_analysis_btn": save_analysis_btn,
        "analysis_model_url": analysis_model_url,
        "analysis_model_name": analysis_model_name,
        "retrieval_top_k": retrieval_top_k,
        "embedding_model": embedding_model_input,
        "analysis_config_status": analysis_config_status,
    }


# --- Dynamic properties module setup ---


class RagUiModule(types.ModuleType):
    @property
    def LAST_CREATED_RUN_ID(self):
        return rag_ui_state.LAST_CREATED_RUN_ID

    @LAST_CREATED_RUN_ID.setter
    def LAST_CREATED_RUN_ID(self, value):
        rag_ui_state.LAST_CREATED_RUN_ID = value

    @property
    def RAG_LOG_BUFFER(self):
        return rag_ui_state.RAG_LOG_BUFFER

    @RAG_LOG_BUFFER.setter
    def RAG_LOG_BUFFER(self, value):
        rag_ui_state.RAG_LOG_BUFFER = value


# Substitute current entry in sys.modules to trigger descriptor lookups on references
sys.modules[__name__].__class__ = RagUiModule
