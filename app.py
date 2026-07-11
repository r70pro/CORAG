import atexit
import gradio as gr

# Shared State

# Settings
from settings_manager import load_settings, save_settings, WORKSPACE_DIR  # noqa: F401

# Docker Container operations
from docker_manager import (
    get_docker_status_str,
    start_docker_container,
    stop_docker_container,
    create_docker_container,
    cleanup_docker
)

# Resets & Space metrics
from cleanup_manager import perform_reset_cleanup

# HTML templates
from html_utils import make_progress_bar_html

# PDF rendering and file conversions
from pdf_manager import (
    on_file_selected,
    update_view
)

# Pipeline runners
from pipeline_manager import process_pdfs, stop_processing, cleanup_active_runs

# Styling and theme properties
from ui_theme import custom_css, dark_theme

# RAG Document Analysis UI
from rag_ui import build_analysis_ui

# Expose additional state and utility functions for tests
from state import active_runs, active_runs_lock  # noqa: F401
from docker_manager import check_server_ready, get_docker_status  # noqa: F401
from html_utils import make_upload_manifest_html, make_file_status_html  # noqa: F401
from pdf_manager import make_zip, load_markdown_content  # noqa: F401
from cleanup_manager import get_dir_size, format_size  # noqa: F401

# Register exit hooks
atexit.register(cleanup_docker)
atexit.register(cleanup_active_runs)



# GUI layout construction

settings = load_settings()


with gr.Blocks(title="OLMOCR PDF Suite") as demo:
    # State tracking
    active_run_id = gr.State("")
    current_pdf_path = gr.State("")
    current_total_pages = gr.State(0)
    current_page_ranges = gr.State([])
    current_full_markdown = gr.State("")

    # ── Header ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(
                "<h1 class='gradient-title' style='margin:0;'>OLMOCR PDF-to-Markdown Suite</h1>"
                "<p style='color:#9ca3af; margin:4px 0 0 0; font-size:1rem;'>High-performance layout-aware PDF OCR pipeline using vision-language models</p>"
            )
        with gr.Column(scale=1, elem_classes=["glass-panel", "status-container"]):
            gr.HTML("<h2 style='margin:0 0 8px 0; color:#e2e8f0; font-size:1rem; font-weight:600; text-align:center;'>🐳 Inference Status</h2>")
            backend_status_badge = gr.HTML("<span class='badge-idle'>Checking Backend...</span>")
            with gr.Row():
                header_docker_start_btn = gr.Button("▶️ Start", variant="secondary", size="sm")
                header_docker_stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="sm")

    # ── Main Content ────────────────────────────────────────────────
    with gr.Row():
        # Left sidebar — Settings (collapsible)
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            with gr.Accordion("⚙️ Pipeline Settings", open=False, elem_classes=["glass-panel"]):
                server_url_input = gr.Textbox(
                    label="vLLM OpenAI Server URL", 
                    value=settings["server_url"], 
                    placeholder="http://localhost:8000/v1"
                )
                current_model = settings.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
                model_choices = [
                    "allenai/olmOCR-2-7B-1025-FP8",
                    "nvidia/Qwen3.6-35B-A3B-NVFP4",
                    "nvidia/Phi-4-reasoning-plus-NVFP4",
                    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                    "nvidia/Llama-3.3-70B-Instruct-NVFP4"
                ]
                if current_model not in model_choices:
                    model_choices.append(current_model)

                model_name_input = gr.Dropdown(
                    label="Model Name",
                    choices=model_choices,
                    value=current_model,
                    interactive=True
                )

                
                with gr.Accordion("Advanced Parameters", open=False):
                    workers_input = gr.Slider(
                        label="Workers", 
                        minimum=1, maximum=64, step=1, 
                        value=settings["workers"]
                    )
                    max_concurrent_input = gr.Slider(
                        label="Max Concurrent Requests", 
                        minimum=1, maximum=2000, step=10, 
                        value=settings["max_concurrent_requests"]
                    )
                    target_dim_input = gr.Slider(
                        label="Target Longest Image Dimension", 
                        minimum=512, maximum=2048, step=64, 
                        value=settings["target_longest_image_dim"]
                    )
                    max_retries_input = gr.Slider(
                        label="Max Page Retries", 
                        minimum=1, maximum=20, step=1, 
                        value=settings["max_page_retries"]
                    )
                    guided_decoding_input = gr.Checkbox(
                        label="Enable Guided Decoding (YAML structure)", 
                        value=settings["guided_decoding"]
                    )
                
                save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
                config_status = gr.Markdown()
                
            with gr.Accordion("🐳 Local Inference Server (Docker)", open=False, elem_classes=["glass-panel"]):
                docker_status_info = gr.Markdown("Manage the local GPU inference container.")
                
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token", 
                    value=settings["hf_token"], 
                    type="password"
                )
                docker_port_input = gr.Number(
                    label="Docker Host Port", 
                    value=settings["docker_port"], 
                    precision=0
                )
                docker_gpu_mem_input = gr.Slider(
                    label="GPU Memory Utilization", 
                    minimum=0.1, maximum=1.0, step=0.05, 
                    value=settings["docker_gpu_mem"]
                )
                docker_max_model_len_input = gr.Slider(
                    label="Max Model Length", 
                    minimum=2048, maximum=32768, step=1024, 
                    value=settings["docker_max_model_len"]
                )
                
                with gr.Row():
                    docker_start_btn = gr.Button("▶️ Start", variant="secondary")
                    docker_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                
                docker_recreate_btn = gr.Button("🔄 Recreate & Run", variant="primary")
                docker_action_status = gr.Markdown()

            with gr.Accordion("🧹 Reset & Cleanup", open=False, elem_classes=["glass-panel"]):
                gr.Markdown("Select components to clean up and reclaim disk space:")
                clean_runs_chk = gr.Checkbox(label="Obsolete run directories (workspace/run_*)", value=True)
                clean_gradio_chk = gr.Checkbox(label="Gradio upload temp files (/tmp/gradio)", value=True)
                clean_pycache_chk = gr.Checkbox(label="Python bytecode cache (__pycache__)", value=True)
                clean_hf_chk = gr.Checkbox(label="Hugging Face model cache (~/.cache/huggingface)", value=False)
                gr.HTML("<span style='color: #f87171; font-size: 0.85rem; display: block; margin-top: -8px; margin-bottom: 8px;'>⚠️ WARNING: Hugging Face deletion will require re-downloading model weights (approx 10-30GB).</span>")
                
                reset_cleanup_btn = gr.Button("🧹 Clean & Reset", variant="stop")
                reset_cleanup_status = gr.Markdown()

        # Center — Upload, Processing Controls, Monitoring, Log
        with gr.Column(scale=4):
            with gr.Row():
                # Upload area
                with gr.Column(scale=2, elem_classes=["glass-panel"]):
                    gr.Markdown("## 📥 Source Documents")
                    pdf_uploader = gr.File(
                        label="Upload / Drag-and-drop PDFs", 
                        file_count="multiple", 
                        file_types=[".pdf"]
                    )
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Batch Processing", variant="primary")
                        stop_btn = gr.Button("🛑 Stop Process", variant="stop", interactive=False)

                # Monitoring cards
                with gr.Column(scale=1, elem_classes=["glass-panel"]):
                    gr.Markdown("## 📊 Monitoring")
                    status_badge = gr.HTML("<span class='badge-idle'>Idle</span>", label="Status")
                    progress_bar = gr.HTML(
                        make_progress_bar_html(0, 0),
                        label="Batch Progress"
                    )
                    with gr.Row():
                        completed_pages_card = gr.HTML(
                            "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Completed Pages</div></div>",
                            visible=True
                        )
                        failed_pages_card = gr.HTML(
                            "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Failed Pages</div></div>",
                            visible=True
                        )

            # Upload manifest + File status (side by side)
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["glass-panel"]):
                    gr.Markdown("## 📋 Upload Manifest")
                    upload_manifest_display = gr.HTML("", elem_classes=["file-status-wrap"])
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
                        lines=10,
                        elem_classes=["log-console"]
                    )

    # ── Results Section (full-width, below main) ───────────────────
    gr.HTML("<hr class='section-divider'>")
    with gr.Row(elem_classes=["glass-panel"]):
        with gr.Column(scale=2):
            file_selector = gr.Dropdown(
                label="📄 Select Processed Document", 
                choices=[], 
                interactive=True
            )
        with gr.Column(scale=1):
            view_mode = gr.Radio(
                choices=["Page-by-Page", "Full Document"],
                value="Page-by-Page",
                label="👁️ View Mode",
                interactive=True
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
                    interactive=True
                )
                next_page_btn = gr.Button("Next Page ➡️", variant="secondary", size="sm")
        with gr.Column(scale=1):
            sync_scroll_check = gr.Checkbox(
                label="Sync Scroll",
                value=True,
                elem_id="sync-scroll-checkbox",
                interactive=True
            )
        with gr.Column(scale=1):
            download_individual_btn = gr.File(
                label="Download Markdown", 
                interactive=False,
                elem_classes=["compact-download"]
            )
        with gr.Column(scale=1):
            download_zip_btn = gr.File(
                label="Download All (ZIP)", 
                interactive=False,
                elem_classes=["compact-download"]
            )

    with gr.Row():
        # Column 1: PDF Viewer
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("## 📄 Original PDF")
            pdf_viewer_panel = gr.HTML(
                value="<div id='pdf-scroll-container' class='sync-scroll-target' style='height: 70vh; display: flex; justify-content: center; align-items: center; background: #0f172a; color: #94a3b8; border-radius: 8px;'>Select a processed document to view.</div>"
            )
            
        # Column 2: Raw Markdown
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            with gr.Row():
                gr.Markdown("## ✍️ Raw Markdown Output")
                copy_btn = gr.Button("📋 Copy", variant="secondary", size="sm")
            raw_markdown_panel = gr.HTML(
                value="<div id='raw-scroll-container' class='sync-scroll-target' style='height: 70vh; display: flex; justify-content: center; align-items: center; background: #020617; color: #94a3b8; border-radius: 8px;'>Select a processed document to view.</div>"
            )
            
        # Column 3: Rendered Preview
        with gr.Column(scale=1, elem_id="preview-scroll-container", elem_classes=["glass-panel"]):
            gr.Markdown("## 👁️ Rendered Preview")
            rendered_markdown = gr.Markdown(value="Select a processed document to preview.")

    # ── RAG Document Analysis Section ──────────────────────────────
    rag_components = build_analysis_ui()

    # Script block moved to demo.load to avoid innerHTML execution restrictions

    # ── Event handlers ─────────────────────────────────────────────

    # Copy to clipboard via JS
    copy_btn.click(
        None,
        js="() => { const rawContainer = document.getElementById('raw-scroll-container'); const text = rawContainer ? rawContainer.innerText : ''; navigator.clipboard.writeText(text); }"
    )

    def trigger_save_settings(url, model, wrk, concat, dim, retries, guided, d_port, d_gpu, d_maxlen, d_token):
        user_config = {
            "server_url": url,
            "model_name": model,
            "workers": int(wrk),
            "max_concurrent_requests": int(concat),
            "target_longest_image_dim": int(dim),
            "max_page_retries": int(retries),
            "guided_decoding": guided,
            "docker_port": int(d_port),
            "docker_gpu_mem": float(d_gpu),
            "docker_max_model_len": int(d_maxlen),
            "hf_token": d_token
        }
        return save_settings(user_config)

    save_config_btn.click(
        trigger_save_settings,
        inputs=[
            server_url_input, model_name_input, 
            workers_input, max_concurrent_input, 
            target_dim_input, max_retries_input, 
            guided_decoding_input,
            docker_port_input, docker_gpu_mem_input,
            docker_max_model_len_input, hf_token_input
        ],
        outputs=[config_status]
    )

    start_btn.click(
        process_pdfs,
        inputs=[
            pdf_uploader, server_url_input, model_name_input,
            workers_input, max_concurrent_input, max_retries_input,
            target_dim_input, guided_decoding_input
        ],
        outputs=[
            log_viewer, status_badge, progress_bar, 
            completed_pages_card, failed_pages_card,
            file_selector, download_zip_btn, download_individual_btn,
            start_btn, active_run_id,
            file_status_table, upload_manifest_display,
        ]
    )
    
    # Update stop button interactivity when run starts
    start_btn.click(
        lambda: gr.update(interactive=True),
        outputs=[stop_btn]
    )

    # Stop button: route feedback to status_badge (not config_status)
    stop_btn.click(
        stop_processing,
        inputs=[active_run_id],
        outputs=[status_badge]
    )
    
    # Disable stop button after stopped/completed
    stop_btn.click(
        lambda: gr.update(interactive=False),
        outputs=[stop_btn]
    )

    # Selection and update events
    file_selector.change(
        on_file_selected,
        inputs=[file_selector, active_run_id],
        outputs=[
            current_pdf_path,
            current_total_pages,
            current_page_ranges,
            current_full_markdown,
            page_selector,
            download_individual_btn
        ]
    ).then(
        update_view,
        inputs=[
            file_selector, view_mode, page_selector,
            current_pdf_path, current_total_pages, current_page_ranges, current_full_markdown
        ],
        outputs=[
            pdf_viewer_panel, raw_markdown_panel, rendered_markdown
        ]
    )

    view_mode.change(
        update_view,
        inputs=[
            file_selector, view_mode, page_selector,
            current_pdf_path, current_total_pages, current_page_ranges, current_full_markdown
        ],
        outputs=[
            pdf_viewer_panel, raw_markdown_panel, rendered_markdown
        ]
    )

    page_selector.change(
        update_view,
        inputs=[
            file_selector, view_mode, page_selector,
            current_pdf_path, current_total_pages, current_page_ranges, current_full_markdown
        ],
        outputs=[
            pdf_viewer_panel, raw_markdown_panel, rendered_markdown
        ]
    )

    def go_prev_page(current_page):
        return max(1, current_page - 1)

    def go_next_page(current_page, total_pages):
        return min(total_pages, current_page + 1)

    prev_page_btn.click(
        go_prev_page,
        inputs=[page_selector],
        outputs=[page_selector]
    )

    next_page_btn.click(
        go_next_page,
        inputs=[page_selector, current_total_pages],
        outputs=[page_selector]
    )

    # Docker event handlers
    def ui_start_container(port):
        success, msg = start_docker_container()
        _, badge = get_docker_status_str(port)
        return msg, badge

    def ui_stop_container(port):
        success, msg = stop_docker_container()
        _, badge = get_docker_status_str(port)
        return msg, badge

    def ui_recreate_container(hf_token, port, model, gpu_mem, max_model_len):
        success, msg = create_docker_container(hf_token, port, model, gpu_mem, max_model_len)
        _, badge = get_docker_status_str(port)
        
        settings = load_settings()
        settings.update({
            "hf_token": hf_token,
            "docker_port": int(port),
            "model_name": model,
            "docker_gpu_mem": float(gpu_mem),
            "docker_max_model_len": int(max_model_len),
            "server_url": f"http://localhost:{int(port)}/v1"
        })
        save_settings(settings)
        new_url = f"http://localhost:{int(port)}/v1"
        return msg, badge, new_url

    # Sidebar Docker buttons
    docker_start_btn.click(
        ui_start_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge]
    )

    docker_stop_btn.click(
        ui_stop_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge]
    )

    docker_recreate_btn.click(
        ui_recreate_container,
        inputs=[
            hf_token_input, docker_port_input, model_name_input,
            docker_gpu_mem_input, docker_max_model_len_input
        ],
        outputs=[docker_action_status, backend_status_badge, server_url_input]
    )

    reset_cleanup_btn.click(
        perform_reset_cleanup,
        inputs=[clean_runs_chk, clean_gradio_chk, clean_pycache_chk, clean_hf_chk],
        outputs=[reset_cleanup_status]
    )

    # Header Docker buttons (same handlers, no status text output)
    def ui_header_start(port):
        start_docker_container()
        _, badge = get_docker_status_str(port)
        return badge

    def ui_header_stop(port):
        stop_docker_container()
        _, badge = get_docker_status_str(port)
        return badge

    header_docker_start_btn.click(
        ui_header_start,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    header_docker_stop_btn.click(
        ui_header_stop,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    # Periodic Backend Status Check
    status_timer = gr.Timer(value=5)
    
    def periodic_status_check(port_val):
        if port_val is None:
            port_val = 8000
        _, badge_html = get_docker_status_str(int(port_val))
        return badge_html

    status_timer.tick(
        periodic_status_check,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    demo.load(
        periodic_status_check,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    demo.load(
        None,
        js="""
        () => {
            // 1. Declare page language and force dark theme (WCAG 2.2 compliance)
            document.documentElement.setAttribute('lang', 'en');

            const forceDarkMode = () => {
                if (!document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.add('dark');
                    document.documentElement.style.colorScheme = 'dark';
                }
            };
            forceDarkMode();

            const darkThemeObserver = new MutationObserver(forceDarkMode);
            darkThemeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

            // 2. Accessibility enhancer for interactive and non-text elements
            function addAriaLabels() {
                // Emojis and action button labels
                const buttons = document.querySelectorAll('button');
                buttons.forEach(btn => {
                    const text = btn.innerText || '';
                    if (text.includes('▶️ Start')) {
                        btn.setAttribute('aria-label', 'Start Docker inference server');
                    } else if (text.includes('⏹️ Stop')) {
                        btn.setAttribute('aria-label', 'Stop Docker inference server');
                    } else if (text.includes('🔄 Recreate & Run')) {
                        btn.setAttribute('aria-label', 'Recreate and run Docker inference container');
                    } else if (text.includes('🚀 Start Batch Processing')) {
                        btn.setAttribute('aria-label', 'Start batch processing of uploaded PDF files');
                    } else if (text.includes('🛑 Stop Process')) {
                        btn.setAttribute('aria-label', 'Stop current batch processing pipeline');
                    } else if (text.includes('⬅️ Prev Page')) {
                        btn.setAttribute('aria-label', 'Go to previous page');
                    } else if (text.includes('Next Page ➡️')) {
                        btn.setAttribute('aria-label', 'Go to next page');
                    } else if (text.includes('📋 Copy')) {
                        btn.setAttribute('aria-label', 'Copy raw markdown text to clipboard');
                    } else if (text.includes('💾 Save Configuration')) {
                        btn.setAttribute('aria-label', 'Save pipeline configuration settings');
                    } else if (text.includes('🧹 Clean & Reset')) {
                        btn.setAttribute('aria-label', 'Perform cache and workspace cleanup');
                    }
                });

                // Gradio settings button at the bottom of the page
                const settingsBtn = document.querySelector('button.settings');
                if (settingsBtn) {
                    settingsBtn.setAttribute('aria-label', 'Settings drawer toggle');
                    const settingsImg = settingsBtn.querySelector('img');
                    if (settingsImg && !settingsImg.getAttribute('alt')) {
                        settingsImg.setAttribute('alt', 'Settings icon');
                    }
                }

                // Hide decorative SVGs inside file upload boxes and other zones from screen readers
                const svgs = document.querySelectorAll('svg');
                svgs.forEach(svg => {
                    if (!svg.getAttribute('aria-hidden') && !svg.querySelector('title')) {
                        svg.setAttribute('aria-hidden', 'true');
                    }
                });
            }

            // Run initial enhancement
            addAriaLabels();

            // Observe dynamic rendering changes to keep dynamic buttons/components compliant
            const accessibilityObserver = new MutationObserver(addAriaLabels);
            accessibilityObserver.observe(document.body, { childList: true, subtree: true });

            // 3. Original scroll-sync logic
            let activeScrollSource = null;
            let scrollTimeout = null;

            function findScrollableElement(el) {
                if (!el) return null;
                if (el.scrollHeight > el.clientHeight && (window.getComputedStyle(el).overflowY === 'auto' || window.getComputedStyle(el).overflowY === 'scroll')) {
                    return el;
                }
                for (let i = 0; i < el.children.length; i++) {
                    const found = findScrollableElement(el.children[i]);
                    if (found) return found;
                }
                return null;
            }

            window.addEventListener('scroll', function(event) {
                const syncCheckboxContainer = document.getElementById('sync-scroll-checkbox');
                const syncCheckbox = syncCheckboxContainer ? syncCheckboxContainer.querySelector('input[type="checkbox"]') : null;
                if (!syncCheckbox || !syncCheckbox.checked) return;

                const source = event.target;
                if (!source) return;

                const pdf = document.getElementById('pdf-scroll-container');
                const raw = document.getElementById('raw-scroll-container');
                const previewContainer = document.getElementById('preview-scroll-container');
                const preview = findScrollableElement(previewContainer) || previewContainer;

                let sourceWindow = null;
                if (pdf && (pdf === source || pdf.contains(source))) {
                    sourceWindow = pdf;
                } else if (raw && (raw === source || raw.contains(source))) {
                    sourceWindow = raw;
                } else if (preview && (preview === source || preview.contains(source))) {
                    sourceWindow = preview;
                }

                if (!sourceWindow) return;

                if (activeScrollSource && activeScrollSource !== sourceWindow) return;
                activeScrollSource = sourceWindow;

                const maxSourceScroll = sourceWindow.scrollHeight - sourceWindow.clientHeight;
                if (maxSourceScroll <= 0) return;
                const percentage = sourceWindow.scrollTop / maxSourceScroll;

                const targets = [pdf, raw, preview].filter(el => el && el !== sourceWindow);

                targets.forEach(target => {
                    const maxTargetScroll = target.scrollHeight - target.clientHeight;
                    target.scrollTop = percentage * maxTargetScroll;
                });

                clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(() => {
                    activeScrollSource = null;
                }, 100);
            }, true);
        }
        """
    )

if __name__ == "__main__":  # pragma: no cover
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css, theme=dark_theme, allowed_paths=["/home/owner"])
