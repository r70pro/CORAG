import atexit
import logging
from collections.abc import Generator
from typing import Any

import gradio as gr

from app_handlers import (
    check_backing_services,  # noqa: F401
    get_gpu_metrics,  # noqa: F401
    go_next_page,
    go_prev_page,
    handle_delete_installed_model_ui,
    handle_get_installed_models_ui,
    periodic_diagnostics_check,
    periodic_status_check,
    select_view,
    trigger_save_settings,
    ui_header_start,
    ui_header_stop,
    ui_recreate_container,
    ui_shutdown_all_containers,
    ui_start_container,
    ui_stop_container,
)

# Resets & Space metrics
from cleanup_manager import format_size, get_dir_size, perform_reset_cleanup  # noqa: F401

# Docker Container operations
from docker_manager import (  # noqa: F401
    check_server_ready,
    cleanup_docker,
    create_docker_container,
    get_docker_status,
    get_docker_status_str,
    shutdown_docker_container,
    start_docker_container,
    stop_docker_container,
)
from embedding_pipeline_ui import build_embedding_pipeline_ui

# HTML templates
from html_utils import (  # noqa: F401
    get_simulated_sparkline,
    make_backing_services_html,
    make_file_status_html,
    make_gpu_metrics_html,
    make_progress_bar_html,
    make_system_health_badge_html,
    make_upload_manifest_html,
)

# PDF rendering and file conversions
from pdf_manager import (  # noqa: F401
    load_markdown_content,
    make_zip,
    on_file_selected,
    update_view,
)
from pipeline_manager import cleanup_active_runs, process_pdfs, stop_processing

# Expose additional state and utility functions for tests
from process_state import active_runs, active_runs_lock  # noqa: F401

# RAG Document Analysis UI
from rag_ui import build_case_dashboard_ui, build_rag_chat_ui  # noqa: F401
from secrets_config import credentials_are_default

# Settings
from settings_manager import (  # noqa: F401
    MODEL_MAX_CONTENT_LENGTHS,
    SUPPORTED_MODELS,
    VERSION,
    WORKSPACE_DIR,
    load_settings,
    save_settings,
)
from system_diagnostics import (  # noqa: F401
    check_backing_services_data,
    get_gpu_metrics_data,
    get_service_latency,
    get_vllm_loading_progress,
)
from ui_adapters import file_selection_to_gradio, pipeline_result_to_gradio

# Styling and theme properties
from ui_theme import custom_css, dark_theme  # noqa: F401

logger = logging.getLogger(__name__)

# Register exit hooks
atexit.register(cleanup_docker)
atexit.register(cleanup_active_runs)


def process_pdfs_ui_wrapper(*args: Any, **kwargs: Any) -> Generator[tuple[Any, ...], None, None]:
    for result in process_pdfs(*args, **kwargs):
        yield pipeline_result_to_gradio(result)


# GUI layout construction

settings = load_settings()

# Warn loudly (but do not hard-block) when backing-service credentials are still
# using the documented unsafe defaults. This guards against accidentally
# running the RAG stack with publicly-known passwords.
if credentials_are_default():
    logger.warning(
        "⚠️  SECURITY WARNING: PostgreSQL, MinIO, or API credentials are using default "
        "placeholder values from .env. Set strong, unique values before exposing this "
        "workstation on any network."
    )

current_model = settings.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
model_choices = list(SUPPORTED_MODELS)
if current_model not in model_choices:
    model_choices.append(current_model)

with gr.Blocks(title="OLMOCR PDF Suite") as demo:
    # State tracking
    active_run_id = gr.State("")
    current_pdf_path = gr.State("")
    current_total_pages = gr.State(0)
    current_page_ranges = gr.State([])
    current_full_markdown = gr.State("")

    # ── Main Content with Sidebar ───────────────────────────────────
    with gr.Row():
        # Left sidebar
        with gr.Column(scale=1, elem_classes=["sidebar-panel", "main-sidebar"]):
            gr.HTML(
                "<div class='sidebar-logo-container'>"
                "<div class='sidebar-logo-title'>IQ-RAG Client</div>"
                "<div class='sidebar-logo-sub'>Mission Control</div>"
                "</div>"
            )

            # Nav buttons
            ingestion_btn = gr.Button(
                "📥 Ingestion Pipeline",
                variant="secondary",
                elem_classes=["nav-btn", "active-nav-btn"],
            )
            inspector_btn = gr.Button(
                "🔍 Layout Inspector", variant="secondary", elem_classes=["nav-btn"]
            )
            embedding_pipeline_btn = gr.Button(
                "🧠 Embedding Pipeline", variant="secondary", elem_classes=["nav-btn"]
            )
            rag_dashboard_btn = gr.Button(
                "📊 Case Dashboard", variant="secondary", elem_classes=["nav-btn"]
            )
            rag_chat_btn = gr.Button(
                "💬 RAG Processing", variant="secondary", elem_classes=["nav-btn"]
            )
            diagnostics_btn = gr.Button(
                "🖥️ System Diagnostics", variant="secondary", elem_classes=["nav-btn"]
            )

            with gr.Accordion(
                "🐳 Inference Server (Docker)", open=True, elem_classes=["glass-panel"]
            ):
                docker_status_info = gr.Markdown("Manage the local GPU inference container.")

                hf_token_input = gr.Textbox(
                    label="Hugging Face Token", value=settings["hf_token"], type="password"
                )
                docker_model_name_input = gr.Dropdown(
                    label="Model Name", choices=model_choices, value=current_model, interactive=True
                )
                docker_port_input = gr.Number(
                    label="Docker Host Port", value=settings["docker_port"], precision=0
                )
                docker_gpu_mem_input = gr.Slider(
                    label="GPU Memory Utilization",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=settings["docker_gpu_mem"],
                )
                initial_max_len = MODEL_MAX_CONTENT_LENGTHS.get(current_model, 131072)
                docker_max_model_len_input = gr.Slider(
                    label="Max Content Length",
                    minimum=2048,
                    maximum=initial_max_len,
                    step=1024,
                    value=min(settings.get("docker_max_model_len", 131072), initial_max_len),
                )
                docker_tensor_parallel_input = gr.Slider(
                    label="Tensor Parallel Size (GPUs)",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=settings.get("docker_tensor_parallel", 1),
                )

                with gr.Row():
                    docker_start_btn = gr.Button("▶️ Start", variant="secondary")
                    docker_stop_btn = gr.Button("⏹️ Stop", variant="secondary")

                with gr.Row():
                    docker_recreate_btn = gr.Button("🔄 Recreate & Run", variant="primary")
                    docker_shutdown_btn = gr.Button("🛑 Shut Down", variant="stop")
                docker_action_status = gr.Markdown()

            # Sidebar Footer
            with gr.Column(elem_classes=["sidebar-footer-container"]):
                gr.Markdown("Active Role:")
                active_role = gr.Dropdown(
                    choices=["Admin", "Clinical Reviewer", "Legal Specialist"],
                    value="Admin",
                    show_label=False,
                    container=False,
                )

                with gr.Row():
                    btn_comfortable = gr.Button("Comfortable", size="sm", variant="secondary")
                    btn_compact = gr.Button("Compact", size="sm", variant="secondary")

                gr.HTML(f"<span class='sidebar-version'>IQ-RAG Workstation v{VERSION}</span>")

        # Right main contents
        with gr.Column(scale=4):
            # Header
            with gr.Row():
                with gr.Column(scale=3):
                    page_title = gr.HTML(
                        "<h1 class='inline-header-title'>Ingestion Pipeline</h1>"
                        "<p class='inline-header-subtitle'>Upload and process documents through the OCR pipeline</p>"
                    )
                with gr.Column(scale=1, elem_classes=["status-container"]):
                    system_health_badge = gr.HTML(
                        "<span class='badge-success'>✓ System Healthy</span>"
                    )
                    backend_status_badge = gr.HTML(
                        "<span class='badge-idle inline-hide-badge'>Checking Backend...</span>",
                        visible=False,
                    )

            # Define the 5 Panels

            # ────────────────────────────────────────────────────────
            # PANEL 1: Ingestion Pipeline
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=True) as ingestion_panel:
                with gr.Row():
                    # Left column — Settings (collapsible)
                    with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
                        with gr.Accordion(
                            "⚙️ Pipeline Settings", open=False, elem_classes=["glass-panel"]
                        ):
                            server_url_input = gr.Textbox(
                                label="vLLM OpenAI Server URL",
                                value=settings["server_url"],
                                placeholder="http://localhost:8000/v1",
                            )
                            model_name_input = gr.Dropdown(
                                label="Model Name",
                                choices=model_choices,
                                value=current_model,
                                interactive=True,
                            )

                            with gr.Accordion("Advanced Parameters", open=False):
                                workers_input = gr.Slider(
                                    label="Workers",
                                    minimum=1,
                                    maximum=64,
                                    step=1,
                                    value=settings["workers"],
                                )
                                max_concurrent_input = gr.Slider(
                                    label="Max Concurrent Requests",
                                    minimum=1,
                                    maximum=2000,
                                    step=10,
                                    value=settings["max_concurrent_requests"],
                                )
                                target_dim_input = gr.Slider(
                                    label="Target Longest Image Dimension",
                                    minimum=512,
                                    maximum=2048,
                                    step=64,
                                    value=settings["target_longest_image_dim"],
                                )
                                max_retries_input = gr.Slider(
                                    label="Max Page Retries",
                                    minimum=1,
                                    maximum=20,
                                    step=1,
                                    value=settings["max_page_retries"],
                                )
                                guided_decoding_input = gr.Checkbox(
                                    label="Enable Guided Decoding (YAML structure)",
                                    value=settings["guided_decoding"],
                                )

                            save_config_btn = gr.Button(
                                "💾 Save Configuration", variant="secondary"
                            )
                            config_status = gr.Markdown()

                    # Center — Upload, Processing Controls, Monitoring, Log
                    with gr.Column(scale=3):
                        with gr.Row():
                            # Upload area
                            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                                gr.Markdown("## 📥 Source Documents")
                                pdf_uploader = gr.File(
                                    label="Upload / Drag-and-drop PDFs",
                                    file_count="multiple",
                                    file_types=[".pdf"],
                                )
                                with gr.Row():
                                    start_btn = gr.Button(
                                        "🚀 Start Batch Processing", variant="primary"
                                    )
                                    stop_btn = gr.Button(
                                        "🛑 Stop Process", variant="stop", interactive=False
                                    )

                            # Monitoring cards
                            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                                gr.Markdown("## 📊 Monitoring")
                                status_badge = gr.HTML(
                                    "<span class='badge-idle'>Idle</span>", label="Status"
                                )
                                progress_bar = gr.HTML(
                                    make_progress_bar_html(0, 0), label="Batch Progress"
                                )
                                with gr.Row():
                                    completed_pages_card = gr.HTML(
                                        "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Completed Pages</div></div>",
                                        visible=True,
                                    )
                                    failed_pages_card = gr.HTML(
                                        "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Failed Pages</div></div>",
                                        visible=True,
                                    )

                        # Upload manifest + File status (side by side)
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                                gr.Markdown("## 📋 Upload Manifest")
                                upload_manifest_display = gr.HTML(
                                    "", elem_classes=["file-status-wrap"]
                                )
                            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                                gr.Markdown("## 📁 Per-File Status")
                                file_status_table = gr.HTML("", elem_classes=["file-status-wrap"])

                        # Log viewer — full width under the upload + monitoring row
                        with gr.Row(elem_classes=["glass-panel"]):
                            with gr.Column():
                                gr.Markdown("## 📜 System Output Log")
                                log_viewer = gr.Code(
                                    label="Logs",
                                    language="shell",
                                    value="",
                                    interactive=False,
                                    lines=30,
                                    elem_classes=["log-console"],
                                )

            # ────────────────────────────────────────────────────────
            # PANEL 2: Layout Inspector
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=False) as inspector_panel:
                with gr.Row(elem_classes=["glass-panel"]):
                    with gr.Column(scale=2):
                        file_selector = gr.Dropdown(
                            label="📄 Select Processed Document", choices=[], interactive=True
                        )
                    with gr.Column(scale=1):
                        view_mode = gr.Radio(
                            choices=["Page-by-Page", "Full Document"],
                            value="Page-by-Page",
                            label="👁️ View Mode",
                            interactive=True,
                        )
                    with gr.Column(scale=2):
                        with gr.Row():
                            prev_page_btn = gr.Button("⬅️ Prev Page", variant="secondary", size="sm")
                            page_selector = gr.Slider(
                                label="Page",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=1,
                                interactive=True,
                            )
                            next_page_btn = gr.Button("Next Page ➡️", variant="secondary", size="sm")
                    with gr.Column(scale=1):
                        sync_scroll_check = gr.Checkbox(
                            label="Sync Scroll",
                            value=True,
                            elem_id="sync-scroll-checkbox",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        download_individual_btn = gr.File(
                            label="Download Markdown",
                            interactive=False,
                            elem_classes=["compact-download"],
                        )
                    with gr.Column(scale=1):
                        download_zip_btn = gr.File(
                            label="Download All (ZIP)",
                            interactive=False,
                            elem_classes=["compact-download"],
                        )

                with gr.Row():
                    # Column 1: PDF Viewer
                    with gr.Column(scale=1, elem_classes=["glass-panel"]):
                        gr.Markdown("## 📄 Original PDF")
                        pdf_viewer_panel = gr.HTML(
                            value="<div id='pdf-scroll-container' class='sync-scroll-target pdf-viewer-placeholder'>Select a processed document to view.</div>"
                        )

                    # Column 2: Raw Markdown
                    with gr.Column(scale=1, elem_classes=["glass-panel"]):
                        with gr.Row():
                            gr.Markdown("## ✍️ Raw Markdown Output")
                            copy_btn = gr.Button("📋 Copy", variant="secondary", size="sm")
                        raw_markdown_panel = gr.HTML(
                            value="<div id='raw-scroll-container' class='sync-scroll-target raw-markdown-placeholder'>Select a processed document to view.</div>"
                        )

                    # Column 3: Rendered Preview
                    with gr.Column(scale=1, elem_classes=["glass-panel"]):
                        gr.Markdown("## 👁️ Rendered Preview")
                        rendered_markdown = gr.Markdown(
                            value="Select a processed document to preview.",
                            elem_id="preview-scroll-container",
                        )

            # ────────────────────────────────────────────────────────
            # PANEL 3: Embedding Pipeline
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=False) as embedding_pipeline_panel:
                embed_components = build_embedding_pipeline_ui()

            # ────────────────────────────────────────────────────────
            # PANEL 4: Case Dashboard
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=False) as rag_dashboard_panel:
                dash_components = build_case_dashboard_ui()

            # ────────────────────────────────────────────────────────
            # PANEL 5: RAG Processing
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=False) as rag_chat_panel:
                chat_components = build_rag_chat_ui()

            # ────────────────────────────────────────────────────────
            # PANEL 6: System Diagnostics
            # ────────────────────────────────────────────────────────
            with gr.Column(visible=False) as diagnostics_panel:
                # Top header bar: status info and quick buttons
                with gr.Row(elem_classes=["diagnostics-header-row"]):
                    with gr.Column(scale=3):
                        gr.HTML(
                            "<div style='display: flex; align-items: center; gap: 12px; margin-top: 10px;'>"
                            "<span class='badge-running' style='padding: 6px 12px; font-weight: 700; font-size: 0.85rem;'>⚡ SERVICE HEALTH & DIAGNOSTICS</span>"
                            "<span style='color: #fbbf24; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;'>OPTIMIZED PERFORMANCE</span>"
                            "</div>"
                        )
                    with gr.Column(scale=2, elem_classes=["diagnostics-actions-col"]):
                        with gr.Row():
                            refresh_diag_btn = gr.Button(
                                "🔄 Refresh Status", variant="secondary", size="sm"
                            )
                            download_report_btn = gr.Button(
                                "📥 Download Report", variant="primary", size="sm"
                            )

                        # Hidden file download component
                        download_file_comp = gr.File(
                            label="Diagnostic Report File", visible=False, interactive=False
                        )

                        with gr.Row(elem_classes=["header-docker-widget"]):
                            gr.HTML(
                                "<span style='font-size: 0.75rem; color: #94a3b8; font-weight: 700; display: flex; align-items: center; margin-right: 8px;'>🐳 DOCKER SERVER:</span>"
                            )
                            docker_widget_stop_btn = gr.Button(
                                "Stop", size="sm", variant="secondary"
                            )
                            docker_widget_recreate_btn = gr.Button(
                                "Recreate", size="sm", variant="secondary"
                            )

                with gr.Row():
                    # Left Column: Services & Cleanup
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["glass-panel"]):
                            gr.HTML(
                                "<h3 class='inline-section-header'>⚡ Backing Services Health</h3>"
                            )
                            backing_services_html = gr.HTML(
                                value="Loading backing services status..."
                            )

                        with gr.Column(elem_classes=["glass-panel"]):
                            gr.HTML(
                                "<h3 class='inline-section-header'>🧹 Reset & Cleanup Manager</h3>"
                            )
                            gr.Markdown("Select components to clean up and reclaim disk space:")
                            with gr.Row():
                                clean_runs_chk = gr.Checkbox(
                                    label="Obsolete run directories (workspace/run_*)", value=True
                                )
                                clean_gradio_chk = gr.Checkbox(
                                    label="Gradio upload temp files (/tmp/gradio)", value=True
                                )
                            with gr.Row():
                                clean_pycache_chk = gr.Checkbox(
                                    label="Python bytecode cache (__pycache__)", value=True
                                )
                                clean_hf_chk = gr.Checkbox(
                                    label="Hugging Face model cache (~/.cache/huggingface)",
                                    value=False,
                                )

                            gr.HTML(
                                "<span class='inline-warning-text' style='margin-bottom: 12px; margin-top: 4px; display: block;'>⚠️ WARNING: Hugging Face deletion will require re-downloading model weights (approx 10-30GB).</span>"
                            )

                            with gr.Row():
                                reset_cleanup_btn = gr.Button(
                                    "🛑 Clear & Reset Selected", variant="stop"
                                )
                            reset_cleanup_status = gr.Markdown()

                    # Right Column: GPU Spec & VRAM + Processes
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["glass-panel"]):
                            gr.HTML(
                                "<h3 class='inline-section-header'>🖥️ Hardware & GPU Resource Utilization</h3>"
                            )
                            hardware_utilization_html = gr.HTML(
                                value="Loading hardware utilization..."
                            )

                with gr.Row():
                    with gr.Column(elem_classes=["glass-panel"]):
                        gr.HTML(
                            "<h3 class='inline-section-header'>📦 Installed Models & Local Disk Storage</h3>"
                        )
                        installed_models_df = gr.Dataframe(
                            headers=["Model ID", "Category", "Context Window", "Disk Size", "Status", "Last Modified"],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            col_count=(6, "fixed"),
                            interactive=False,
                        )
                        with gr.Row():
                            delete_model_dropdown = gr.Dropdown(
                                label="Select Installed Model to Delete",
                                choices=[],
                                interactive=True,
                            )
                            delete_model_btn = gr.Button("🗑️ Delete Selected Model Cache", variant="stop")
                        delete_model_status = gr.Markdown()

    # Extra hidden buttons for backwards compatibility / test suite (since tests patch header buttons)
    with gr.Row(visible=False):
        header_docker_start_btn = gr.Button("Start", visible=False)
        header_docker_stop_btn = gr.Button("Stop", visible=False)

    # ── Event handlers ─────────────────────────────────────────────

    # Sidebar Navigation View Toggling

    nav_outputs = [
        page_title,
        ingestion_btn,
        inspector_btn,
        embedding_pipeline_btn,
        rag_dashboard_btn,
        rag_chat_btn,
        diagnostics_btn,
        ingestion_panel,
        inspector_panel,
        embedding_pipeline_panel,
        rag_dashboard_panel,
        rag_chat_panel,
        diagnostics_panel,
    ]

    ingestion_btn.click(lambda: select_view(0), outputs=nav_outputs)
    inspector_btn.click(lambda: select_view(1), outputs=nav_outputs)
    embedding_pipeline_btn.click(lambda: select_view(2), outputs=nav_outputs).then(
        embed_components["refresh_fn"], outputs=[embed_components["telemetry_comp"]]
    )
    rag_dashboard_btn.click(lambda: select_view(3), outputs=nav_outputs).then(
        dash_components["refresh_fn"],
        outputs=[
            dash_components["dashboard_html"],
            dash_components["dashboard_delete_selector"],
            dash_components["dashboard_status"],
        ],
    )
    rag_chat_btn.click(lambda: select_view(4), outputs=nav_outputs).then(
        chat_components["refresh_fn"], outputs=[chat_components["active_case_selector"]]
    )
    diagnostics_btn.click(lambda: select_view(5), outputs=nav_outputs).then(
        handle_get_installed_models_ui,
        outputs=[installed_models_df, delete_model_dropdown],
    )

    # Comfortable vs Compact Layout spacing handlers
    btn_comfortable.click(
        None,
        js="() => { document.querySelector('.gradio-container').classList.remove('layout-compact'); }",
    )
    btn_compact.click(
        None,
        js="() => { document.querySelector('.gradio-container').classList.add('layout-compact'); }",
    )

    # Copy to clipboard via JS
    copy_btn.click(
        None,
        js="() => { const rawContainer = document.getElementById('raw-scroll-container'); const text = rawContainer ? rawContainer.innerText : ''; navigator.clipboard.writeText(text); }",
    )

    save_config_btn.click(
        trigger_save_settings,
        inputs=[
            server_url_input,
            model_name_input,
            workers_input,
            max_concurrent_input,
            target_dim_input,
            max_retries_input,
            guided_decoding_input,
            docker_port_input,
            docker_gpu_mem_input,
            docker_max_model_len_input,
            hf_token_input,
        ],
        outputs=[config_status],
    )

    start_btn.click(
        process_pdfs_ui_wrapper,
        inputs=[
            pdf_uploader,
            server_url_input,
            model_name_input,
            workers_input,
            max_concurrent_input,
            max_retries_input,
            target_dim_input,
            guided_decoding_input,
        ],
        outputs=[
            log_viewer,
            status_badge,
            progress_bar,
            completed_pages_card,
            failed_pages_card,
            file_selector,
            download_zip_btn,
            download_individual_btn,
            start_btn,
            active_run_id,
            file_status_table,
            upload_manifest_display,
            stop_btn,
        ],
    )

    stop_btn.click(stop_processing, inputs=[active_run_id], outputs=[status_badge])

    stop_btn.click(lambda: gr.update(interactive=False), outputs=[stop_btn])

    # Selection and update events
    file_selector.change(
        lambda sel, rid: file_selection_to_gradio(on_file_selected(sel, rid)),
        inputs=[file_selector, active_run_id],
        outputs=[
            current_pdf_path,
            current_total_pages,
            current_page_ranges,
            current_full_markdown,
            page_selector,
            download_individual_btn,
        ],
    ).then(
        update_view,
        inputs=[
            file_selector,
            view_mode,
            page_selector,
            current_pdf_path,
            current_total_pages,
            current_page_ranges,
            current_full_markdown,
        ],
        outputs=[pdf_viewer_panel, raw_markdown_panel, rendered_markdown],
    )

    view_mode.change(
        update_view,
        inputs=[
            file_selector,
            view_mode,
            page_selector,
            current_pdf_path,
            current_total_pages,
            current_page_ranges,
            current_full_markdown,
        ],
        outputs=[pdf_viewer_panel, raw_markdown_panel, rendered_markdown],
    )

    page_selector.change(
        update_view,
        inputs=[
            file_selector,
            view_mode,
            page_selector,
            current_pdf_path,
            current_total_pages,
            current_page_ranges,
            current_full_markdown,
        ],
        outputs=[pdf_viewer_panel, raw_markdown_panel, rendered_markdown],
    )

    prev_page_btn.click(go_prev_page, inputs=[page_selector], outputs=[page_selector])

    next_page_btn.click(
        go_next_page, inputs=[page_selector, current_total_pages], outputs=[page_selector]
    )

    # Sidebar Docker buttons
    docker_start_btn.click(
        ui_start_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge],
    )

    docker_stop_btn.click(
        ui_stop_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge],
    )

    docker_recreate_btn.click(
        ui_recreate_container,
        inputs=[
            hf_token_input,
            docker_port_input,
            docker_model_name_input,
            docker_gpu_mem_input,
            docker_max_model_len_input,
            docker_tensor_parallel_input,
        ],
        outputs=[docker_action_status, backend_status_badge, server_url_input],
    )

    docker_shutdown_btn.click(
        ui_shutdown_all_containers,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge],
    )

    model_name_input.change(
        lambda x: x, inputs=[model_name_input], outputs=[docker_model_name_input]
    )

    def update_max_content_length(model_name, current_val):
        max_len = MODEL_MAX_CONTENT_LENGTHS.get(model_name, 131072)
        new_val = min(current_val, max_len)
        return gr.update(maximum=max_len, value=new_val)

    docker_model_name_input.change(
        update_max_content_length,
        inputs=[docker_model_name_input, docker_max_model_len_input],
        outputs=[docker_max_model_len_input],
    )
    docker_model_name_input.change(
        lambda x: x, inputs=[docker_model_name_input], outputs=[model_name_input]
    )

    reset_cleanup_btn.click(
        perform_reset_cleanup,
        inputs=[clean_runs_chk, clean_gradio_chk, clean_pycache_chk, clean_hf_chk],
        outputs=[reset_cleanup_status],
    )

    # Wire up the new header actions & widget controls
    refresh_diag_btn.click(
        periodic_diagnostics_check,
        inputs=[docker_port_input],
        outputs=[backing_services_html, hardware_utilization_html, system_health_badge],
    )

    from app_handlers import trigger_download_report

    download_report_btn.click(
        trigger_download_report, inputs=[docker_port_input], outputs=[download_file_comp]
    )

    docker_widget_stop_btn.click(
        ui_stop_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge],
    )

    docker_widget_recreate_btn.click(
        ui_recreate_container,
        inputs=[
            hf_token_input,
            docker_port_input,
            docker_model_name_input,
            docker_gpu_mem_input,
            docker_max_model_len_input,
            docker_tensor_parallel_input,
        ],
        outputs=[docker_action_status, backend_status_badge, server_url_input],
    )

    header_docker_start_btn.click(
        ui_header_start, inputs=[docker_port_input], outputs=[backend_status_badge]
    )

    header_docker_stop_btn.click(
        ui_header_stop, inputs=[docker_port_input], outputs=[backend_status_badge]
    )

    # Periodic Backend Status Check
    status_timer = gr.Timer(value=5)

    status_timer.tick(
        periodic_status_check,
        inputs=[docker_port_input],
        outputs=[backend_status_badge],
        api_name="periodic_status_check",
    )

    status_timer.tick(
        periodic_diagnostics_check,
        inputs=[docker_port_input],
        outputs=[backing_services_html, hardware_utilization_html, system_health_badge],
    )

    demo.load(periodic_status_check, inputs=[docker_port_input], outputs=[backend_status_badge])

    refresh_diag_btn.click(
        handle_get_installed_models_ui,
        outputs=[installed_models_df, delete_model_dropdown],
    )

    delete_model_btn.click(
        handle_delete_installed_model_ui,
        inputs=[delete_model_dropdown],
        outputs=[delete_model_status, installed_models_df, delete_model_dropdown],
    )

    demo.load(
        periodic_diagnostics_check,
        inputs=[docker_port_input],
        outputs=[backing_services_html, hardware_utilization_html, system_health_badge],
    )

    demo.load(
        handle_get_installed_models_ui,
        outputs=[installed_models_df, delete_model_dropdown],
    )

    import os

    js_path = os.path.join(os.path.dirname(__file__), "assets", "accessibility.js")
    with open(js_path, encoding="utf-8") as f:
        accessibility_js = f.read()

    demo.load(None, js=accessibility_js)

if __name__ == "__main__":  # pragma: no cover
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css,
        theme=dark_theme,
        allowed_paths=["/home/owner"],
    )
