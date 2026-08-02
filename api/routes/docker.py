"""
Docker container management API routes.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_remote_lifecycle_enabled, verify_admin_key
from api.models import (
    AnalysisContextModeRequest,
    AnalysisSwitchRequest,
    DockerCreateRequest,
    DockerLogsResponse,
    DockerModelsResponse,
    DockerStatusResponse,
    MessageResponse,
    StartupModeRequest,
)

router = APIRouter()


@router.get("/analysis/profiles", response_model=dict, summary="Verified analysis profiles")
def get_analysis_profiles():
    from analysis_profiles import analysis_status

    return analysis_status()


@router.get("/analysis/status", response_model=dict, summary="Live analysis model state")
def get_analysis_model_status():
    from analysis_profiles import analysis_status

    return analysis_status()


@router.post(
    "/analysis/switch", response_model=dict, status_code=202,
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
    summary="Start a guarded analysis model switch",
)
def switch_analysis_model(req: AnalysisSwitchRequest):
    from analysis_profiles import start_switch

    try:
        return start_switch(req.target_model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/analysis/operations/{operation_id}", response_model=dict)
def get_analysis_switch_operation(operation_id: str):
    from analysis_profiles import get_operation

    operation = get_operation(operation_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Analysis switch operation not found")
    return operation


@router.post(
    "/analysis/context-mode",
    response_model=MessageResponse,
    summary="Switch analysis context mode",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def set_analysis_context_mode(req: AnalysisContextModeRequest):
    """Use the OCR model's GPU allocation for a 262K analysis context, or restore it."""
    from analysis_profiles import switch_in_progress
    from docker_manager import set_extended_analysis_context
    if switch_in_progress():
        raise HTTPException(status_code=409, detail="Analysis model switch is in progress")

    success, msg = await asyncio.to_thread(set_extended_analysis_context, req.extended)
    if not success:
        raise HTTPException(status_code=503, detail=msg)
    from settings_manager import load_settings, save_settings
    settings = load_settings()
    settings["startup_mode"] = "analysis_262k" if req.extended else "dual_32k"
    save_settings(settings)
    return MessageResponse(success=True, message=msg)


@router.post(
    "/startup-mode",
    response_model=MessageResponse,
    summary="Select and persist an inference operating mode",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def set_startup_mode(req: StartupModeRequest):
    """Apply a workflow-oriented model profile and restore it next launch."""
    from analysis_profiles import switch_in_progress
    from docker_manager import set_extended_analysis_context, set_vllm_role_running
    if switch_in_progress():
        raise HTTPException(status_code=409, detail="Analysis model switch is in progress")
    from settings_manager import load_settings, save_settings

    if req.mode == "analysis_262k":
        success, msg = await asyncio.to_thread(set_extended_analysis_context, True)
    elif req.mode == "dual_32k":
        success, msg = await asyncio.to_thread(set_extended_analysis_context, False)
    else:
        analysis_ok, analysis_msg = await asyncio.to_thread(set_vllm_role_running, "analysis", False)
        ocr_ok, ocr_msg = await asyncio.to_thread(set_vllm_role_running, "ocr", True)
        success, msg = analysis_ok and ocr_ok, f"{analysis_msg} {ocr_msg}"
    if not success:
        raise HTTPException(status_code=503, detail=msg)
    settings = load_settings()
    settings["startup_mode"] = req.mode
    save_settings(settings)
    return MessageResponse(success=True, message=f"Operating mode saved. {msg}")


@router.get("/models", response_model=DockerModelsResponse, summary="Get available/cached models")
def get_models():
    """Return list of cached and default preset model names with max context lengths."""
    from docker_manager import get_cached_models_info

    models, max_lengths = get_cached_models_info()
    return DockerModelsResponse(models=models, max_lengths=max_lengths)


@router.get("/status", response_model=DockerStatusResponse, summary="Get container status")
def get_status(role: str = Query("ocr", pattern="^(ocr|analysis)$")):
    """Return the current vLLM inference container status."""
    from docker_manager import get_docker_status_str
    from settings_manager import load_settings

    settings = load_settings()
    is_analysis = role == "analysis"
    port = 8002 if is_analysis else settings.get("docker_port", 8000)
    container = "kirag_vllm_analysis" if is_analysis else "olmocr"
    status_text, badge_html = get_docker_status_str(port, container)
    # Derive a machine-readable status from the badge
    status = "unknown"
    if "Ready" in badge_html or "badge-success" in badge_html:
        status = "ready"
    elif "Foreign Container" in badge_html:
        status = "foreign"
    elif "Running" in badge_html or "Starting" in badge_html or "badge-running" in badge_html:
        status = "starting"
    elif "Stopped" in badge_html or "badge-stopped" in badge_html:
        status = "stopped"
    elif "Not Found" in badge_html or "badge-idle" in badge_html:
        status = "not_found"
    elif "Error" in badge_html or "Failed" in badge_html or "badge-failed" in badge_html:
        status = "error"
    return DockerStatusResponse(status=status, message=status_text, badge_html=badge_html)


@router.get("/logs", response_model=DockerLogsResponse, summary="Get container logs")
def get_logs(
    tail: int = Query(200, ge=1, le=10_000),
    role: str = Query("ocr", pattern="^(ocr|analysis)$"),
):
    """Return stdout/stderr logs from the vLLM container."""
    from docker_manager import get_docker_logs, get_docker_status

    container = "kirag_vllm_analysis" if role == "analysis" else "olmocr"
    logs = get_docker_logs(tail=tail, container_name=container)
    status = get_docker_status(container)
    return DockerLogsResponse(logs=logs, container_status=status)


@router.post(
    "/start",
    response_model=MessageResponse,
    summary="Start container",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def start_container():
    """Start the existing vLLM inference container."""
    from docker_manager import start_docker_container

    success, msg = await asyncio.to_thread(start_docker_container)
    if not success:
        raise HTTPException(status_code=503, detail=msg or "Unable to start container")
    return MessageResponse(success=success, message=msg)


@router.post(
    "/roles/{role}/start",
    response_model=MessageResponse,
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def start_role_container(role: str):
    from analysis_profiles import switch_in_progress
    from docker_manager import set_vllm_role_running
    if role == "analysis" and switch_in_progress():
        raise HTTPException(status_code=409, detail="Analysis model switch is in progress")

    success, msg = await asyncio.to_thread(set_vllm_role_running, role, True)
    if not success:
        raise HTTPException(status_code=503, detail=msg)
    return MessageResponse(success=True, message=msg)


@router.post(
    "/roles/{role}/stop",
    response_model=MessageResponse,
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def stop_role_container(role: str):
    from analysis_profiles import switch_in_progress
    from docker_manager import set_vllm_role_running
    if role == "analysis" and switch_in_progress():
        raise HTTPException(status_code=409, detail="Analysis model switch is in progress")

    success, msg = await asyncio.to_thread(set_vllm_role_running, role, False)
    if not success:
        raise HTTPException(status_code=503, detail=msg)
    return MessageResponse(success=True, message=msg)


@router.post(
    "/stop",
    response_model=MessageResponse,
    summary="Stop container",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def stop_container():
    """Stop the running vLLM inference container."""
    from docker_manager import stop_docker_container

    success, msg = await asyncio.to_thread(stop_docker_container)
    if not success:
        raise HTTPException(status_code=500, detail=msg or "Unable to stop container")
    return MessageResponse(success=success, message=msg)


@router.post(
    "/create",
    response_model=MessageResponse,
    summary="Create/recreate container",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def create_container(req: DockerCreateRequest):
    """Create or recreate the vLLM inference container with the given parameters."""
    import os

    from docker_manager import create_docker_container
    from settings_manager import load_settings, save_settings

    # Keep environment-only credentials in memory. Model recreation must not
    # copy HF_TOKEN into the tracked settings file unless the request explicitly
    # asks to save a token.
    settings = load_settings(include_env_secrets=False)
    model = req.model
    if not model or model == "model":
        model = settings.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
        if not model or model == "model":
            model = "allenai/olmOCR-2-7B-1025-FP8"
    if model != "allenai/olmOCR-2-7B-1025-FP8":
        raise HTTPException(
            status_code=422,
            detail="The production OCR endpoint only accepts allenai/olmOCR-2-7B-1025-FP8; use /analysis/switch for analysis models",
        )

    explicit_hf_token = req.hf_token if req.hf_token and req.hf_token != "********" else ""
    hf_token = explicit_hf_token or settings.get("hf_token", "") or os.environ.get("HF_TOKEN", "")

    port = req.port if req.port else settings.get("docker_port", 8000)
    gpu_mem = req.gpu_mem if req.gpu_mem else settings.get("docker_gpu_mem", 0.8)
    max_model_len = (
        req.max_model_len if req.max_model_len else settings.get("docker_max_model_len", 15360)
    )
    tensor_parallel_size = (
        req.tensor_parallel_size
        if req.tensor_parallel_size
        else settings.get("docker_tensor_parallel", 1)
    )

    success, msg = await asyncio.to_thread(
        create_docker_container, hf_token, port, model, gpu_mem, max_model_len, tensor_parallel_size
    )

    # Invalidate stale model resolution cache after container recreation
    try:
        from rag.analyzer import invalidate_model_cache

        invalidate_model_cache()
    except Exception:
        pass

    # Persist the new settings
    if success:
        server_url = f"http://localhost:{port}/v1"
        new_settings = {
            "docker_port": port,
            "docker_gpu_mem": gpu_mem,
            "docker_max_model_len": max_model_len,
            "docker_tensor_parallel": tensor_parallel_size,
        }
        new_settings.update({"model_name": model, "server_url": server_url})
        if explicit_hf_token:
            new_settings["hf_token"] = explicit_hf_token
        settings.update(new_settings)
        save_settings(settings)
    else:
        raise HTTPException(status_code=503, detail=msg or "Unable to create container")
    return MessageResponse(success=success, message=msg)


@router.post(
    "/shutdown",
    response_model=MessageResponse,
    summary="Shutdown and remove",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def shutdown_container():
    """Stop and remove the vLLM inference container."""
    from docker_manager import shutdown_docker_container

    success, msg = await asyncio.to_thread(shutdown_docker_container)
    if not success:
        raise HTTPException(status_code=500, detail=msg or "Unable to remove container")
    return MessageResponse(success=success, message=msg)
