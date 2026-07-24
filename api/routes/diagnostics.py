"""
System diagnostics API routes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from api.models import CleanupRequest, CleanupResponse, GPUInfo, HealthResponse, ServiceHealth

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Full health check")
def health_check():
    """Return a full system health snapshot (services + GPU)."""
    from settings_manager import load_settings
    from system_diagnostics import check_backing_services_data, get_gpu_metrics_data

    settings = load_settings()
    port = settings.get("docker_port", 8000)

    backing = check_backing_services_data({}, vllm_port=port)
    gpu = get_gpu_metrics_data()

    services = []
    for name, info in backing.get("services", {}).items():
        services.append(
            ServiceHealth(
                name=name,
                is_up=info["is_up"],
                latency_ms=info["latency"],
                extra_info=info.get("extra_info"),
            )
        )

    gpu_info = GPUInfo(
        cuda_available=gpu.get("cuda_available", False),
        gpu_name=gpu.get("gpu_name", "N/A"),
        vram_used_mb=gpu.get("vram_used", 0.0),
        vram_total_mb=gpu.get("vram_total", 0.0),
        vram_pct=gpu.get("vram_pct", 0.0),
        vram_free_mb=gpu.get("vram_free", 0.0),
        vram_reclaimable_mb=gpu.get("vram_reclaimable", 0.0),
        processes=gpu.get("processes", []),
    )

    overall = "healthy" if backing.get("all_healthy") else "degraded"
    return HealthResponse(overall=overall, services=services, gpu=gpu_info)


@router.get("/gpu", response_model=GPUInfo, summary="GPU metrics")
def gpu_metrics():
    """Return GPU hardware metrics only."""
    from system_diagnostics import get_gpu_metrics_data

    gpu = get_gpu_metrics_data()
    return GPUInfo(
        cuda_available=gpu.get("cuda_available", False),
        gpu_name=gpu.get("gpu_name", "N/A"),
        vram_used_mb=gpu.get("vram_used", 0.0),
        vram_total_mb=gpu.get("vram_total", 0.0),
        vram_pct=gpu.get("vram_pct", 0.0),
        vram_free_mb=gpu.get("vram_free", 0.0),
        vram_reclaimable_mb=gpu.get("vram_reclaimable", 0.0),
        processes=gpu.get("processes", []),
    )


@router.get("/services", summary="Backing services status")
def services_status():
    """Return latency and status of each backing service."""
    from settings_manager import load_settings
    from system_diagnostics import check_backing_services_data

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    data = check_backing_services_data({}, vllm_port=port)
    return data


@router.get("/report", summary="Download diagnostic report")
def download_report():
    """Generate and return a downloadable diagnostic report file."""
    from settings_manager import load_settings
    from system_diagnostics import generate_diagnostic_report_file

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    report_path = generate_diagnostic_report_file(port)
    return FileResponse(
        report_path,
        media_type="text/markdown",
        filename="diagnostic_report.md",
    )


@router.post(
    "/cleanup", response_model=CleanupResponse, summary="Perform system reset and disk cleanup"
)
def execute_cleanup(req: CleanupRequest):
    """Execute cleanup of selected components to reclaim disk space."""
    from api.models import CleanupResponse
    from cleanup_manager import perform_reset_cleanup

    components = req.components or []
    clean_runs = "runs" in components or "workspace" in components
    clean_gradio = "temp" in components or "cache" in components or "gradio" in components
    clean_pycache = "pycache" in components or "bytecode" in components
    clean_hf = "hf" in components or "models" in components

    res = perform_reset_cleanup(clean_runs, clean_gradio, clean_pycache, clean_hf)
    if isinstance(res, str):
        clean_msg = res.replace("### ", "").replace("**", "").replace("`", "").strip()
    else:
        clean_msg = str(res)

    return CleanupResponse(
        success=True,
        message=clean_msg,
        reclaimed_bytes=0,
        reclaimed_str="",
    )
