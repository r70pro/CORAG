import gradio as gr

from docker_manager import (
    create_docker_container,
    get_docker_status_str,
    shutdown_docker_container,
    start_docker_container,
    stop_docker_container,
)
from html_utils import (
    make_backing_services_html,
    make_gpu_metrics_html,
    make_system_health_badge_html,
)
from settings_manager import load_settings, save_settings
from system_diagnostics import (
    check_backing_services_data,
    get_gpu_metrics_data,
)

# Service health check latency history (keeps last 8 data points)
service_history = {
    "postgres": [1.2, 1.1, 1.3, 1.2, 1.4, 1.2, 1.3],
    "redis": [0.8, 0.7, 0.9, 0.8, 0.8, 0.7, 0.8],
    "minio": [3.0, 3.2, 2.9, 3.1, 3.0, 2.8, 3.0],
    "qdrant": [2.1, 2.3, 2.0, 2.2, 2.1, 1.9, 2.1],
    "vllm": [15.8, 15.2, 16.1, 15.5, 15.9, 14.8, 15.8],
}


def get_app_fn(name, fallback):
    import sys

    app = sys.modules.get("app")
    return getattr(app, name, fallback) if app else fallback


def check_backing_services(vllm_port=8000):
    data = check_backing_services_data(service_history, vllm_port)
    return make_backing_services_html(data), make_system_health_badge_html(data)


def get_gpu_metrics():
    data = get_gpu_metrics_data()
    return make_gpu_metrics_html(data)


def select_view(active_view_idx):
    titles = [
        "<h1 class='inline-header-title'>Ingestion Pipeline</h1><p class='inline-header-subtitle'>Upload and process documents through the OCR pipeline</p>",
        "<h1 class='inline-header-title'>Layout Inspector</h1><p class='inline-header-subtitle'>Verify visual text extraction accuracy side-by-side</p>",
        "<h1 class='inline-header-title'>Embedding & Vector Pipeline</h1><p class='inline-header-subtitle'>Manage Stage 2 dense vector embedding, device acceleration, and Qdrant storage</p>",
        "<h1 class='inline-header-title'>Case Dashboard</h1><p class='inline-header-subtitle'>Overview of ingested case folders and databases</p>",
        "<h1 class='inline-header-title'>RAG Processing (Query & Cite)</h1><p class='inline-header-subtitle'>Query, summarize, and retrieve matching citations</p>",
        "<h1 class='inline-header-title'>System Diagnostics</h1><p class='inline-header-subtitle'>Service health, GPU telemetry & cleanup management.</p>",
    ]

    btn_updates = []
    for i in range(6):
        if i == active_view_idx:
            btn_updates.append(gr.update(elem_classes=["nav-btn", "active-nav-btn"]))
        else:
            btn_updates.append(gr.update(elem_classes=["nav-btn"]))

    view_updates = []
    for i in range(6):
        view_updates.append(gr.update(visible=(i == active_view_idx)))

    return [gr.update(value=titles[active_view_idx])] + btn_updates + view_updates


def trigger_save_settings(
    url, model, wrk, concat, dim, retries, guided, d_port, d_gpu, d_maxlen, d_token
):
    save_settings_fn = get_app_fn("save_settings", save_settings)
    settings = load_settings()
    settings.update(
        {
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
            "hf_token": d_token,
        }
    )
    return save_settings_fn(settings)


def go_prev_page(current_page):
    return max(1, current_page - 1)


def go_next_page(current_page, total_pages):
    return min(total_pages, current_page + 1)


def ui_start_container(port):
    start_fn = get_app_fn("start_docker_container", start_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    success, msg = start_fn()
    _, badge = status_fn(port)
    return msg, badge


def ui_stop_container(port):
    stop_fn = get_app_fn("stop_docker_container", stop_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    success, msg = stop_fn()
    _, badge = status_fn(port)
    return msg, badge


def ui_recreate_container(hf_token, port, model, gpu_mem, max_model_len, tensor_parallel_size=1):
    create_fn = get_app_fn("create_docker_container", create_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    save_settings_fn = get_app_fn("save_settings", save_settings)

    # Normalize model name if empty, invalid, or literally "model"
    if not model or not str(model).strip() or str(model).strip() == "model":
        model_str = "allenai/olmOCR-2-7B-1025-FP8"
    else:
        model_str = str(model).strip()

    # Coerce the port defensively
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        port_int = 8000

    try:
        tp_int = max(1, int(tensor_parallel_size))
    except (TypeError, ValueError):
        tp_int = 1

    try:
        gpu_mem_float = float(gpu_mem)
        if gpu_mem_float <= 0 or gpu_mem_float > 1.0:
            gpu_mem_float = 0.8
    except (TypeError, ValueError):
        gpu_mem_float = 0.8

    try:
        max_len_int = int(max_model_len)
        if max_len_int <= 0:
            max_len_int = 15360
    except (TypeError, ValueError):
        max_len_int = 15360

    success, msg = create_fn(hf_token, port_int, model_str, gpu_mem_float, max_len_int, tp_int)
    _, badge = status_fn(port_int)

    # Invalidate stale model resolution cache after container recreation
    try:
        from rag.analyzer import invalidate_model_cache

        invalidate_model_cache()
    except Exception:
        pass

    settings = load_settings()
    new_url = f"http://localhost:{port_int}/v1"
    new_settings = {
        "docker_port": port_int,
        "model_name": model_str,
        "docker_gpu_mem": gpu_mem_float,
        "docker_max_model_len": max_len_int,
        "docker_tensor_parallel": tp_int,
        "server_url": new_url,
        # Keep analysis settings in sync when model changes
        "analysis_model_name": model_str,
        "analysis_server_url": new_url,
    }
    # Only update hf_token if user provided a new non-masked token
    if hf_token and str(hf_token).strip() and str(hf_token).strip() != "********":
        new_settings["hf_token"] = str(hf_token).strip()

    settings.update(new_settings)
    save_settings_fn(settings)
    return msg, badge, new_url


def ui_header_start(port):
    start_fn = get_app_fn("start_docker_container", start_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    start_fn()
    _, badge = status_fn(port)
    return badge


def ui_header_stop(port):
    stop_fn = get_app_fn("stop_docker_container", stop_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    stop_fn()
    _, badge = status_fn(port)
    return badge


def periodic_status_check(port_val):
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    if port_val is None:
        port_val = 8000
    _, badge_html = status_fn(int(port_val))
    return badge_html


def periodic_diagnostics_check(port_val):
    check_backing = get_app_fn("check_backing_services", check_backing_services)
    get_gpu = get_app_fn("get_gpu_metrics", get_gpu_metrics)
    if port_val is None:
        port_val = 8000
    backing_services, header_health_badge = check_backing(vllm_port=int(port_val))
    gpu_stats = get_gpu()
    return backing_services, gpu_stats, header_health_badge


def ui_shutdown_all_containers(port):
    shutdown_fn = get_app_fn("shutdown_docker_container", shutdown_docker_container)
    status_fn = get_app_fn("get_docker_status_str", get_docker_status_str)
    success, msg1 = shutdown_fn()
    try:
        from rag_infra_manager import destroy_rag_infrastructure

        success2, msg2 = destroy_rag_infrastructure(remove_volumes=False)
        msg = f"{msg1} RAG: {msg2}"
    except Exception as e:
        msg = f"{msg1} RAG: Error destroying infrastructure: {e}"
    _, badge = status_fn(port)
    return msg, badge


def trigger_download_report(port_val):
    from system_diagnostics import generate_diagnostic_report_file

    if port_val is None:
        port_val = 8000
    report_path = generate_diagnostic_report_file(int(port_val))
    return gr.update(value=report_path, visible=True)


def handle_get_installed_models_ui():
    from system_diagnostics import get_installed_models_data

    data = get_installed_models_data()
    rows = []
    deletable_choices = []
    for m in data.get("models", []):
        status_text = "ACTIVE" if m.get("is_active") else "Available"
        rows.append(
            [
                m["id"],
                m["model_type"],
                f"{m['context_length']:,} tokens",
                m["human_size"],
                status_text,
                m["modified_at"],
            ]
        )
        if not m.get("is_active"):
            deletable_choices.append(m["id"])
    return rows, gr.update(choices=deletable_choices, value=None)


def handle_delete_installed_model_ui(selected_model_id):
    from system_diagnostics import delete_installed_models

    if not selected_model_id:
        rows, dropdown_update = handle_get_installed_models_ui()
        return "⚠️ No model selected for deletion.", rows, dropdown_update

    success, msg, deleted, reclaimed = delete_installed_models([selected_model_id])
    rows, dropdown_update = handle_get_installed_models_ui()
    status_msg = f"✓ {msg}" if success else f"❌ {msg}"
    return status_msg, rows, dropdown_update
