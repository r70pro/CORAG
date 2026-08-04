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
    VllmSwitchRequest,
)

router = APIRouter()


@router.get("/vllm/status", response_model=dict, summary="Single inference-slot status")
def get_vllm_status():
    from vllm_lifecycle import status

    return status()


@router.post(
    "/vllm/switch", response_model=dict, status_code=202,
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
    summary="Switch the exclusive vLLM inference slot",
)
async def switch_vllm_slot(req: VllmSwitchRequest):
    from settings_manager import load_settings, save_settings
    from vllm_lifecycle import switch_vllm

    settings = load_settings()
    model = req.model or settings.get("analysis_model_name") if req.role == "analysis" else None
    try:
        operation = await asyncio.to_thread(switch_vllm, req.role, model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    settings["startup_mode"] = req.role
    if req.role == "analysis" and model:
        settings["analysis_model_name"] = model
    save_settings(settings)
    return operation


@router.post(
    "/vllm/stop", response_model=MessageResponse,
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def stop_vllm_slot():
    from settings_manager import load_settings, save_settings
    from vllm_lifecycle import stop_vllm
    try:
        await asyncio.to_thread(stop_vllm)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    settings = load_settings()
    settings["startup_mode"] = "stopped"
    save_settings(settings)
    return MessageResponse(success=True, message="Inference slot stopped")


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
    """Reject the retired shared-GPU mode with an actionable response."""
    raise HTTPException(
        status_code=410,
        detail="Context mode was retired; switch the exclusive slot to OCR or analysis.",
    )


@router.post(
    "/startup-mode",
    response_model=MessageResponse,
    summary="Select and persist an inference operating mode",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def set_startup_mode(req: StartupModeRequest):
    """Apply a workflow-oriented model profile and restore it next launch."""
    from settings_manager import load_settings, save_settings
    from vllm_lifecycle import stop_vllm, switch_vllm
    settings = load_settings()

    try:
        if req.mode == "stopped":
            await asyncio.to_thread(stop_vllm)
            msg = "Inference slot stopped."
        else:
            model = settings.get("analysis_model_name") if req.mode == "analysis" else None
            await asyncio.to_thread(switch_vllm, req.mode, model)
            msg = f"{req.mode.upper()} inference is ready."
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    from vllm_lifecycle import status as slot_status
    slot = slot_status()
    available = bool(slot.get(role, {}).get("available"))
    status_text = f"{role.upper()} is {'ready' if available else 'inactive'}; active role: {slot.get('active_role', 'stopped')}"
    badge_html = "<span class='badge-success'>Inference Server: Ready</span>" if available else "<span class='badge-stopped'>Role: Inactive</span>"
    # Derive a machine-readable status from the badge
    status = "ready" if available else "stopped"
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

    container = "kirag_vllm"
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
    """Restore the persisted role in the exclusive inference slot."""
    from settings_manager import load_settings
    from vllm_lifecycle import switch_vllm
    settings = load_settings()
    role = settings.get("startup_mode", "analysis")
    role = {"analysis_262k": "analysis", "ocr_only": "ocr"}.get(role, role)
    if role not in {"ocr", "analysis"}:
        raise HTTPException(status_code=409, detail="No inference role is selected")
    model = settings.get("analysis_model_name") if role == "analysis" else None
    try:
        await asyncio.to_thread(switch_vllm, role, model)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MessageResponse(success=True, message=f"{role.upper()} inference is ready")


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
    """Stop the exclusive inference slot after active work drains."""
    return await stop_vllm_slot()


@router.post(
    "/create",
    response_model=MessageResponse,
    summary="Create/recreate container",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def create_container(req: DockerCreateRequest):
    """Compatibility endpoint that now performs a guarded OCR-role switch."""
    from settings_manager import load_settings, save_settings
    from vllm_lifecycle import switch_vllm
    from vllm_profiles import OCR_MODEL

    if req.model != OCR_MODEL or req.port != 8000:
        raise HTTPException(
            status_code=422,
            detail="The create endpoint is restricted to the pinned OCR profile on port 8000",
        )

    settings = load_settings(include_env_secrets=False)
    try:
        await asyncio.to_thread(switch_vllm, "ocr")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    settings.update({"startup_mode": "ocr", "model_name": OCR_MODEL, "server_url": "http://127.0.0.1:8000/v1"})
    save_settings(settings)
    return MessageResponse(success=True, message="OCR inference is ready in the exclusive slot")


@router.post(
    "/shutdown",
    response_model=MessageResponse,
    summary="Shutdown and remove",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
async def shutdown_container():
    """Stop and remove the exclusive vLLM inference slot."""
    return await stop_vllm_slot()
