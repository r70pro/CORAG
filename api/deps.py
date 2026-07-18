"""
Shared dependencies and helpers for API route modules.
"""

from __future__ import annotations

from settings_manager import load_settings


def get_settings() -> dict:
    """Return the current application settings dict."""
    return load_settings()
