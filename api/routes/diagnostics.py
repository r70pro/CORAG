"""
System diagnostics API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth import verify_admin_key
from api.models import (
    CleanupRequest,
    CleanupResponse,
    DeleteModelsRequest,
    DeleteModelsResponse,
    GPUInfo,
    HealthResponse,
    InstalledModelItem,
    InstalledModelsResponse,
    ServiceHealth,
)
from path_security import PathSecurityError, require_approved_file, resolve_under

logger = logging.getLogger(__name__)

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
    from settings_manager import WORKSPACE_DIR

    try:
        safe_report = require_approved_file(
            report_path, {resolve_under(WORKSPACE_DIR, "exports")}, {".md"}
        )
    except PathSecurityError as exc:
        raise HTTPException(status_code=500, detail="Unable to generate report") from exc
    return FileResponse(
        safe_report,
        media_type="text/markdown",
        filename="diagnostic_report.md",
    )


@router.post(
    "/cleanup",
    response_model=CleanupResponse,
    summary="Perform system reset and disk cleanup",
    dependencies=[Depends(verify_admin_key)],
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


@router.get(
    "/models", response_model=InstalledModelsResponse, summary="Get all installed/cached models"
)
def get_installed_models():
    """Return detailed metadata and disk space usage for all installed/cached models."""
    try:
        from system_diagnostics import get_installed_models_data

        data = get_installed_models_data()
        models = [InstalledModelItem(**m) for m in data.get("models", [])]
        return InstalledModelsResponse(
            models=models,
            total_count=data.get("total_count", len(models)),
            total_size_bytes=data.get("total_size_bytes", 0),
            total_human_size=data.get("total_human_size", "0 B"),
        )
    except Exception as e:
        logger.error(f"Error fetching installed models: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve installed models") from e


@router.delete(
    "/models",
    response_model=DeleteModelsResponse,
    summary="Delete selected installed models",
    dependencies=[Depends(verify_admin_key)],
)
def delete_models(req: DeleteModelsRequest):
    """Delete selected cached model directories from disk to reclaim storage space."""
    try:
        from system_diagnostics import delete_installed_models, format_bytes_human

        success, msg, deleted, reclaimed = delete_installed_models(req.model_ids)
        response = DeleteModelsResponse(
            success=success,
            message=msg,
            deleted_models=deleted,
            reclaimed_bytes=reclaimed,
            reclaimed_str=format_bytes_human(reclaimed),
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete models")
        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error deleting installed models: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete models") from e
