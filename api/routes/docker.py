"""
Docker container management API routes.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.models import DockerCreateRequest, DockerStatusResponse, MessageResponse

router = APIRouter()


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


@router.post("/start", response_model=MessageResponse, summary="Start container")
def start_container():
    """Start the existing vLLM inference container."""
    from docker_manager import start_docker_container

    success, msg = start_docker_container()
    return MessageResponse(success=success, message=msg)


@router.post("/stop", response_model=MessageResponse, summary="Stop container")
def stop_container():
    """Stop the running vLLM inference container."""
    from docker_manager import stop_docker_container

    success, msg = stop_docker_container()
    return MessageResponse(success=success, message=msg)


@router.post("/create", response_model=MessageResponse, summary="Create/recreate container")
def create_container(req: DockerCreateRequest):
    """Create or recreate the vLLM inference container with the given parameters."""
    from docker_manager import create_docker_container
    from settings_manager import load_settings, save_settings

    success, msg = create_docker_container(
        req.hf_token, req.port, req.model, req.gpu_mem, req.max_model_len
    )
    # Persist the new settings
    if success:
        settings = load_settings()
        settings.update(
            {
                "hf_token": req.hf_token,
                "docker_port": req.port,
                "model_name": req.model,
                "docker_gpu_mem": req.gpu_mem,
                "docker_max_model_len": req.max_model_len,
                "server_url": f"http://localhost:{req.port}/v1",
            }
        )
        save_settings(settings)
    return MessageResponse(success=success, message=msg)


@router.post("/shutdown", response_model=MessageResponse, summary="Shutdown and remove")
def shutdown_container():
    """Stop and remove the vLLM inference container."""
    from docker_manager import shutdown_docker_container

    success, msg = shutdown_docker_container()
    return MessageResponse(success=success, message=msg)
