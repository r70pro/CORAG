"""
Settings management API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_admin_key
from api.models import MessageResponse, SettingsUpdateRequest
from settings_manager import load_settings, save_settings

router = APIRouter()


@router.get("", response_model=dict, summary="Get current settings")
def get_settings():
    """Return the full application settings dictionary."""
    settings = load_settings(include_env_secrets=False)
    # Mask the HF token for security
    if settings.get("hf_token"):
        settings["hf_token"] = "********"
    return settings


@router.put(
    "",
    response_model=MessageResponse,
    summary="Update settings",
    dependencies=[Depends(verify_admin_key)],
)
def update_settings(req: SettingsUpdateRequest):
    """Merge provided fields into the current settings and save."""
    # Environment-only secrets must not be materialized into the mutable
    # settings file as a side effect of changing an unrelated preference.
    settings = load_settings(include_env_secrets=False)
    update_data = req.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided to update")
    if "analysis_model_name" in update_data or "analysis_server_url" in update_data:
        requested_model = update_data.get("analysis_model_name", settings.get("analysis_model_name"))
        requested_url = update_data.get("analysis_server_url", settings.get("analysis_server_url"))
        if (
            requested_model != settings.get("analysis_model_name")
            or requested_url != settings.get("analysis_server_url")
        ):
            raise HTTPException(
                status_code=422,
                detail="Analysis endpoint identity is managed by the guarded /api/docker/analysis/switch workflow",
            )
    chunk_size = update_data.get("chunk_size", settings.get("chunk_size", 800))
    chunk_overlap = update_data.get("chunk_overlap", settings.get("chunk_overlap", 100))
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be smaller than chunk_size")
    settings.update(update_data)
    result = save_settings(settings)
    if "Error" in result:
        raise HTTPException(status_code=500, detail="Unable to save settings")
    return MessageResponse(success=True, message=result)
