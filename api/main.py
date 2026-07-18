"""
KIRAG REST API — FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

OpenAPI docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from settings_manager import VERSION


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    yield
    # Cleanup on shutdown
    try:
        from pipeline_manager import cleanup_active_runs

        cleanup_active_runs()
    except Exception:
        pass


app = FastAPI(
    title="KIRAG API",
    version=VERSION,
    description=(
        "Medicolegal RAG Workstation REST API.\n\n"
        "Provides programmatic access to OCR pipeline processing, "
        "Docker container management, RAG query/indexing, "
        "system diagnostics, and configuration management."
    ),
    lifespan=lifespan,
)

# CORS — allow all origins for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount route modules ──────────────────────────────────────────────────────

from api.routes import diagnostics, docker, documents, pipeline, rag, settings  # noqa: E402

app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(docker.router, prefix="/api/docker", tags=["Docker"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["Diagnostics"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


@app.get("/", tags=["Root"])
def root():
    """API root — returns version and available endpoints."""
    return {
        "service": "KIRAG API",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": {
            "pipeline": "/api/pipeline",
            "docker": "/api/docker",
            "rag": "/api/rag",
            "diagnostics": "/api/diagnostics",
            "settings": "/api/settings",
            "documents": "/api/documents",
        },
    }
