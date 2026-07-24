"""
KIRAG REST API Server.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload

OpenAPI docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

# All API endpoints are now consolidated in api.main
from api.main import app

__all__ = ["app"]
