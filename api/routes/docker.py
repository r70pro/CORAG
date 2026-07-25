"""
Docker container management API routes.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from api.models import (
    DockerCreateRequest,
    DockerLogsResponse,
    DockerModelsResponse,
    DockerStatusResponse,
    MessageResponse,
)

router = APIRouter()


@router.get("/models", response_model=DockerModelsResponse, summary="Get available/cached models")
def get_models():
    """Return list of cached and default preset model names with max context lengths."""
    from docker_manager import get_cached_models_info

    models, max_lengths = get_cached_models_info()
    return DockerModelsResponse(models=models, max_lengths=max_lengths)



@router.get("/status", response_model=DockerStatusResponse, summary="Get container status")

def get_status():
    """Return the current vLLM inference container status."""
    from docker_manager import get_docker_status_str
    from settings_manager import load_settings

    settings = load_settings()
    port = settings.get("docker_port", 8000)
    status_text, badge_html = get_docker_status_str(port)
    # Derive a machine-readable status from the badge
    status = "unknown"
    if "Ready" in badge_html or "badge-success" in badge_html:
        status = "ready"
    elif "Running" in badge_html or "Starting" in badge_html or "badge-running" in badge_html:
        status = "starting"
    elif "Stopped" in badge_html or "badge-stopped" in badge_html:
        status = "stopped"
    elif "Not Found" in badge_html or "badge-idle" in badge_html:
        status = "not_found"
    return DockerStatusResponse(status=status, message=status_text, badge_html=badge_html)


@router.get("/logs", response_model=DockerLogsResponse, summary="Get container logs")
def get_logs(tail: int = 200):
    """Return stdout/stderr logs from the vLLM container."""
    from docker_manager import get_docker_logs, get_docker_status

    logs = get_docker_logs(tail=tail)
    status = get_docker_status()
    return DockerLogsResponse(logs=logs, container_status=status)


@router.post("/start", response_model=MessageResponse, summary="Start container")
async def start_container():
    """Start the existing vLLM inference container."""
    from docker_manager import start_docker_container

    success, msg = await asyncio.to_thread(start_docker_container)
    return MessageResponse(success=success, message=msg)


@router.post("/stop", response_model=MessageResponse, summary="Stop container")
async def stop_container():
    """Stop the running vLLM inference container."""
    from docker_manager import stop_docker_container

    success, msg = await asyncio.to_thread(stop_docker_container)
    return MessageResponse(success=success, message=msg)


@router.post("/create", response_model=MessageResponse, summary="Create/recreate container")
async def create_container(req: DockerCreateRequest):
    """Create or recreate the vLLM inference container with the given parameters."""
    import os

    from docker_manager import create_docker_container
    from settings_manager import load_settings, save_settings

    settings = load_settings()
    model = req.model
    if not model or model == "model":
        model = settings.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
        if not model or model == "model":
            model = "allenai/olmOCR-2-7B-1025-FP8"

    hf_token = req.hf_token if req.hf_token and req.hf_token != "********" else settings.get("hf_token", "")
    if hf_token == "********":
        hf_token = os.environ.get("HF_TOKEN", "")

    port = req.port if req.port else settings.get("docker_port", 8000)
    gpu_mem = req.gpu_mem if req.gpu_mem else settings.get("docker_gpu_mem", 0.8)
    max_model_len = req.max_model_len if req.max_model_len else settings.get("docker_max_model_len", 15360)
    tensor_parallel_size = req.tensor_parallel_size if req.tensor_parallel_size else settings.get("docker_tensor_parallel", 1)

    success, msg = await asyncio.to_thread(
        create_docker_container,
        hf_token, port, model, gpu_mem, max_model_len, tensor_parallel_size
    )
    # Persist the new settings
    if success:
        new_settings = {
            "docker_port": port,
            "model_name": model,
            "docker_gpu_mem": gpu_mem,
            "docker_max_model_len": max_model_len,
            "docker_tensor_parallel": tensor_parallel_size,
            "server_url": f"http://localhost:{port}/v1",
        }
        if hf_token and hf_token != "********":
            new_settings["hf_token"] = hf_token
        settings.update(new_settings)
        save_settings(settings)
    return MessageResponse(success=success, message=msg)


@router.post("/shutdown", response_model=MessageResponse, summary="Shutdown and remove")
async def shutdown_container():
    """Stop and remove the vLLM inference container."""
    from docker_manager import shutdown_docker_container

    success, msg = await asyncio.to_thread(shutdown_docker_container)
    return MessageResponse(success=success, message=msg)


