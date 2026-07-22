"""
Embedding Pipeline UI Module for KIRAG.

Provides a dedicated workspace for Stage 2 Dense Vector Embedding:
- Device Acceleration (Auto CUDA GPU, CUDA, CPU)
- Model Selection & Batching
- Chunking Hyperparameters (Chunk Size & Overlap)
- Live Qdrant & Redis Telemetry
- Batch Indexing Actions & Live Log Output
"""

import gradio as gr
from rag.embedding import get_collection_info, get_collection_name
from rag_ui_handlers import (
    get_available_runs,
    get_rag_logs,
    index_all_runs_ui_wrapper,
    index_run_ui_wrapper,
    log_to_rag,
)
from settings_manager import load_settings, save_settings


def get_embedding_telemetry_html():
    """Build dynamic telemetry HTML for Qdrant vector store & Redis cache."""
    try:
        settings = load_settings()
        model_name = settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
        device = settings.get("embedding_device", "auto")
        col_name = get_collection_name(model_name)
        info = get_collection_info(model_name)

        active_device = device
        if device == "auto" or not device:
            try:
                import torch

                active_device = "CUDA GPU (NVIDIA GB10)" if torch.cuda.is_available() else "CPU Mode"
            except Exception:
                active_device = "CPU Mode"

        # Redis cache info
        cached_count = "N/A"
        try:
            import rag.cache as cache

            if cache.is_healthy():
                info_cache = cache.get_cache_info()
                cached_count = f"{info_cache.get('cached_embeddings', 0)} vectors"
        except Exception:
            pass

        return f"""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 15px;'>
            <div style='background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 15px;'>
                <div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;'>Compute Engine</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #34d399;'>⚡ {str(active_device).upper()}</div>
                <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>Device Target: {device}</div>
            </div>
            <div style='background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 15px;'>
                <div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;'>Qdrant Points Count</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #60a5fa;'>{info.get("points_count", 0)} Points</div>
                <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>Collection: {col_name[:24]}...</div>
            </div>
            <div style='background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 15px;'>
                <div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;'>Vector Dimension</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #a7f3d0;'>1024-dim</div>
                <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>Metric: Cosine Similarity</div>
            </div>
            <div style='background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 15px;'>
                <div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;'>Redis Vector Cache</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #f472b6;'>{cached_count}</div>
                <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>Bulk pipeline cached</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color: #ef4444;'>Error loading telemetry: {e}</div>"


def save_embedding_pipeline_settings(model_name, device, chunk_size, chunk_overlap, batch_size):
    """Save embedding settings and refresh model."""
    try:
        new_settings = {
            "embedding_model": model_name,
            "embedding_device": device,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "embedding_batch_size": int(batch_size),
        }
        save_settings(new_settings)
        log_to_rag(
            f"Saved embedding configuration: model={model_name}, device={device}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, batch_size={batch_size}"
        )
        return "✅ Embedding configuration saved successfully!"
    except Exception as e:
        log_to_rag(f"Failed to save embedding configuration: {e}")
        return f"❌ Save error: {e}"


def purge_embedding_cache():
    """Purge Redis embedding cache."""
    try:
        import rag.cache as cache

        cache.invalidate_embedding_cache()
        log_to_rag("Purged Redis embedding cache.")
        return "✅ Redis embedding cache cleared!"
    except Exception as e:
        return f"❌ Cache purge error: {e}"


def build_embedding_pipeline_ui():
    """Build the standalone Embedding & Vector Indexing Pipeline page layout."""
    settings = load_settings()

    with gr.Column():
        # Top Header Bar
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(
                    "<div style='display: flex; align-items: center; gap: 12px; margin-top: 5px;'>"
                    "<span class='badge-running' style='padding: 6px 12px; font-weight: 700; font-size: 0.85rem;'>🧠 STAGE 2: DENSE VECTOR EMBEDDING & INDEXING</span>"
                    "<span style='color: #34d399; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.2); padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;'>50x GPU ACCELERATED</span>"
                    "</div>"
                )
            with gr.Column(scale=1):
                refresh_telemetry_btn = gr.Button("🔄 Refresh Telemetry", variant="secondary", size="sm")

        # Telemetry Row
        telemetry_html_comp = gr.HTML(value=get_embedding_telemetry_html)

        # Configuration Row (2 Columns)
        with gr.Row():
            # Left Box: Engine & Hardware
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("### ⚡ Compute Engine & Hardware Acceleration")

                device_dropdown = gr.Dropdown(
                    label="Compute Engine Device",
                    choices=[
                        ("⚡ Auto (CUDA GPU when available)", "auto"),
                        ("🚀 CUDA GPU Dedicated", "cuda"),
                        ("💻 CPU Mode", "cpu"),
                    ],
                    value=settings.get("embedding_device", "auto"),
                    interactive=True,
                )

                current_model = settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
                model_choices = [
                    "BAAI/bge-large-en-v1.5",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                    "nomic-ai/nomic-embed-text-v1.5",
                ]
                if current_model not in model_choices:
                    model_choices.append(current_model)

                model_dropdown = gr.Dropdown(
                    label="Embedding Model Name",
                    choices=model_choices,
                    value=current_model,
                    interactive=True,
                    allow_custom_value=True,
                )

                batch_size_slider = gr.Slider(
                    label="Embedding Batch Size",
                    minimum=16,
                    maximum=512,
                    step=16,
                    value=settings.get("embedding_batch_size", 64),
                    interactive=True,
                )

            # Right Box: Chunking & Indexing Hyperparameters
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("### 🧩 Chunking & Indexing Hyperparameters")

                chunk_size_slider = gr.Slider(
                    label="Max Chunk Size (Characters)",
                    minimum=200,
                    maximum=2000,
                    step=50,
                    value=settings.get("chunk_size", 800),
                    interactive=True,
                )

                chunk_overlap_slider = gr.Slider(
                    label="Chunk Overlap (Characters)",
                    minimum=0,
                    maximum=500,
                    step=10,
                    value=settings.get("chunk_overlap", 100),
                    interactive=True,
                )

                with gr.Row():
                    save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
                    purge_cache_btn = gr.Button("🧹 Clear Vector Cache", variant="secondary")

                save_status_msg = gr.Markdown("")

        # Section 3: Direct External Markdown Upload & Indexing
        gr.HTML("<hr class='inline-section-divider'>")
        gr.Markdown("### 📥 Direct External Markdown Upload & Indexing")
        gr.Markdown("Upload markdown files directly into a new or existing case, bypassing OCR ingestion to chunk and generate dense vectors.")

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                external_md_uploader = gr.File(
                    label="Select Markdown Files (.md)",
                    file_count="multiple",
                    file_types=[".md"],
                )
                target_case_dropdown = gr.Dropdown(
                    label="Target Case",
                    choices=[("🆕 Create New Case", "new")]
                    + [choice for choice in get_available_runs() if choice[1] != ""],
                    value="new",
                    interactive=True,
                )
                new_case_name = gr.Textbox(
                    label="New Case Name",
                    placeholder="e.g. My Custom Case",
                    visible=True,
                )
                upload_md_btn = gr.Button("📥 Upload & Index Markdown", variant="primary")

            with gr.Column(scale=3):
                upload_status_card = gr.HTML("")

        # Indexing Execution & Live Logging Section
        gr.HTML("<hr class='inline-section-divider'>")
        gr.Markdown("### ⚙️ Batch Indexing Operations & Real-Time Console")

        with gr.Row():
            with gr.Column(scale=2):
                run_selector = gr.Dropdown(
                    label="Select OCR Run to Index",
                    choices=get_available_runs(),
                    interactive=True,
                )
                with gr.Row():
                    index_selected_btn = gr.Button("📥 Index Selected Run", variant="primary")
                    index_all_btn = gr.Button("📥 Index All Runs", variant="secondary")

            with gr.Column(scale=3):
                indexing_status_card = gr.HTML("")

        log_console = gr.Textbox(
            label="Real-Time Indexing Execution Log",
            value=get_rag_logs,
            lines=8,
            max_lines=12,
            interactive=False,
        )

        # Event Handlers
        from rag_ui_handlers import upload_and_index_markdown_ui_wrapper

        def _get_updated_case_choices():
            return gr.update(
                choices=[("🆕 Create New Case", "new")]
                + [choice for choice in get_available_runs() if choice[1] != ""]
            )

        def _get_updated_run_selector_choices():
            return gr.update(choices=get_available_runs())

        upload_md_btn.click(
            upload_and_index_markdown_ui_wrapper,
            inputs=[external_md_uploader, target_case_dropdown, new_case_name],
            outputs=[upload_status_card, log_console],
        ).then(get_embedding_telemetry_html, outputs=[telemetry_html_comp]).then(
            _get_updated_case_choices, outputs=[target_case_dropdown]
        ).then(
            _get_updated_run_selector_choices, outputs=[run_selector]
        )

        # Event Handlers
        save_config_btn.click(
            save_embedding_pipeline_settings,
            inputs=[
                model_dropdown,
                device_dropdown,
                chunk_size_slider,
                chunk_overlap_slider,
                batch_size_slider,
            ],
            outputs=[save_status_msg],
        ).then(get_embedding_telemetry_html, outputs=[telemetry_html_comp])

        purge_cache_btn.click(purge_embedding_cache, outputs=[save_status_msg]).then(
            get_embedding_telemetry_html, outputs=[telemetry_html_comp]
        )

        refresh_telemetry_btn.click(get_embedding_telemetry_html, outputs=[telemetry_html_comp])

        index_selected_btn.click(
            index_run_ui_wrapper,
            inputs=[run_selector],
            outputs=[indexing_status_card, log_console],
        ).then(get_embedding_telemetry_html, outputs=[telemetry_html_comp])

        index_all_btn.click(
            index_all_runs_ui_wrapper,
            outputs=[indexing_status_card, log_console],
        ).then(get_embedding_telemetry_html, outputs=[telemetry_html_comp])

    return {
        "refresh_fn": get_embedding_telemetry_html,
        "telemetry_comp": telemetry_html_comp,
    }
