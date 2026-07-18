"""
Settings management API routes.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.models import MessageResponse, SettingsUpdateRequest
from settings_manager import load_settings, save_settings

router = APIRouter()


@router.get("/", response_model=dict, summary="Get current settings")
def get_settings():
    """Return the full application settings dictionary."""
    settings = load_settings()
    # Mask the HF token for security
    if settings.get("hf_token"):
        settings["hf_token"] = "********"
    return settings


@router.put("/", response_model=MessageResponse, summary="Update settings")
def update_settings(req: SettingsUpdateRequest):
    """Merge provided fields into the current settings and save."""
    settings = load_settings()
    update_data = req.model_dump(exclude_none=True)
    if not update_data:
        return MessageResponse(success=False, message="No fields provided to update.")
    settings.update(update_data)
    result = save_settings(settings)
    return MessageResponse(success="Error" not in result, message=result)
