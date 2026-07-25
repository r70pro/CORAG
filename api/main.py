"""
KIRAG REST API — FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

OpenAPI docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from api.models import PipelineStartRequest, RAGQueryRequest
from settings_manager import VERSION

logger = logging.getLogger(__name__)

# API Key security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    key_from_header: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str | None:
    """Verify API authentication key if KIRAG_API_KEY environment variable is configured."""
    expected_key = os.environ.get("KIRAG_API_KEY", "").strip()
    if not expected_key:
        return None

    provided_key = key_from_header or (bearer.credentials if bearer else None)
    if not provided_key or provided_key != expected_key:
        logger.warning("Unauthorized API access attempt: invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided_key


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    logger.info(f"Starting KIRAG API v{VERSION}...")
    yield
    logger.info("Shutting down KIRAG API...")
    # Cleanup on shutdown
    try:
        from pipeline_manager import cleanup_active_runs

        cleanup_active_runs()
    except Exception as e:
        logger.error(f"Error during API shutdown cleanup: {e}")


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
    dependencies=[Depends(verify_api_key)],
)

# CORS — specific allowed origins to tighten cross-origin access control
cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
if cors_origins_env:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:7860",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:7860",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error processing request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )


# ── Mount route modules ──────────────────────────────────────────────────────

from api.routes import diagnostics, docker, documents, pipeline, rag, settings  # noqa: E402

app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(docker.router, prefix="/api/docker", tags=["Docker"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["Diagnostics"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


# ── Phase 1 Consolidated Endpoints ───────────────────────────────────────────


@app.post("/api/ingest", summary="Ingest documents via pipeline", tags=["Phase 1 Core Endpoints"])
def api_ingest(req: PipelineStartRequest):
    """Trigger OCR ingestion pipeline for specified file paths."""
    from api.routes.pipeline import start_pipeline

    return start_pipeline(req)


@app.post("/api/chat", summary="Query or chat with RAG engine", tags=["Phase 1 Core Endpoints"])
def api_chat(req: RAGQueryRequest):
    """Run RAG chat query with optional SSE streaming or full JSON output."""
    from api.routes.rag import rag_query

    return rag_query(req)


@app.get(
    "/health",
    summary="Top-level health check for load balancers/monitors",
    tags=["Health"],
    dependencies=[],
)
def health_check():
    """Simple, lightweight health check endpoint returning HTTP 200 for load balancers."""
    return {"status": "ok", "service": "KIRAG API", "version": VERSION}


@app.get(
    "/api/health", summary="System & backing services health check", tags=["Phase 1 Core Endpoints"]
)
def api_health():
    """Return health status of backing Docker services and GPU metrics."""
    from system_diagnostics import check_backing_services_data, get_gpu_metrics_data

    try:
        backing = check_backing_services_data()
        gpu = get_gpu_metrics_data()
        services = []
        if isinstance(backing, dict):
            raw_services = backing.get("services", backing)
            if isinstance(raw_services, dict):
                for name, info in raw_services.items():
                    if isinstance(info, dict):
                        services.append(
                            {
                                "name": str(name),
                                "is_up": bool(info.get("is_up", False)),
                                "latency_ms": float(
                                    info.get("latency") or info.get("latency_ms") or 0.0
                                ),
                                "extra_info": info.get("extra_info"),
                            }
                        )
                    else:
                        services.append(
                            {
                                "name": str(name),
                                "is_up": info == "healthy" or bool(info),
                                "latency_ms": 0.0,
                                "extra_info": str(info) if info is not None else None,
                            }
                        )
            elif isinstance(raw_services, list):
                services = raw_services
        elif isinstance(backing, list):
            services = backing

        all_healthy = backing.get("all_healthy", True) if isinstance(backing, dict) else True
        vllm_model = backing.get("vllm_model") if isinstance(backing, dict) else None
        vllm_progress = backing.get("vllm_progress") if isinstance(backing, dict) else None
        failed_services = backing.get("failed_services", []) if isinstance(backing, dict) else []

        status_str = (
            "healthy"
            if all_healthy
            else ("loading" if failed_services == ["vllm"] and vllm_progress else "degraded")
        )

        return {
            "status": status_str,
            "all_healthy": all_healthy,
            "failed_services": failed_services,
            "vllm_model": vllm_model,
            "vllm_progress": vllm_progress,
            "services": services,
            "gpu": gpu,
            "gpu_metrics": gpu,
        }
    except Exception as e:
        logger.error(f"Error in api_health check: {e}")
        return {"status": "error", "message": str(e), "services": []}


@app.get(
    "/api/case-summary",
    summary="Retrieve summary of indexed cases & corpus",
    tags=["Phase 1 Core Endpoints"],
)
def api_case_summary():
    """Return aggregate corpus statistics and list of indexed cases with rich metadata."""
    from rag.db import get_corpus_stats, get_runs_with_stats
    from rag.embedding import get_collection_info
    from rag.metadata_helper import get_all_cases_metadata, get_case_timeline

    try:
        db_stats = get_corpus_stats()
        runs_with_stats = get_runs_with_stats()
        run_ids = [r.get("run_id") for r in runs_with_stats if r.get("run_id")]
        cases_metadata = get_all_cases_metadata(run_ids)
        qdrant_info = get_collection_info()

        indexed_cases = []
        for r in runs_with_stats:
            rid = r.get("run_id", "")
            meta = cases_metadata.get(rid, {})
            names = meta.get("names", [])
            client_name = ", ".join(names) if names else f"Case {rid[:8]}"
            dob = meta.get("dob", "—")
            injuries = meta.get("injuries", [])

            earliest = r.get("earliest_date")
            latest = r.get("latest_date")
            date_range = "—"
            if earliest and latest:
                date_range = f"{earliest} → {latest}"
            elif earliest:
                date_range = f"{earliest} → ..."
            elif latest:
                date_range = f"... → {latest}"

            indexed_cases.append(
                {
                    "run_id": rid,
                    "display_name": client_name,
                    "client_name": client_name,
                    "dob": dob,
                    "injuries": injuries if injuries else ["No specific injury/diagnosis found"],
                    "documents_count": r.get("total_documents", 1),
                    "chunks_count": r.get("total_chunks", 0),
                    "authors_count": r.get("unique_authors", 0),
                    "date_range": date_range,
                    "created_at": str(r.get("created_at", "")),
                    "indexed_at": str(r.get("indexed_at", "")),
                    "timeline_events": get_case_timeline(rid),
                }
            )

        return {
            "stats": db_stats,
            "indexed_cases": indexed_cases,
            "vector_store": {
                "points_count": qdrant_info.get("points_count", 0),
                "status": qdrant_info.get("status", "unknown"),
            },
        }
    except Exception as e:
        logger.error(f"Error in api_case_summary: {e}")
        return {"error": str(e)}


@app.get("/api/cases/{run_id}/timeline", tags=["Phase 1 Core Endpoints"])
def api_case_timeline(run_id: str):
    """Return chronological medicolegal timeline events for a specific run ID."""
    from rag.metadata_helper import get_case_timeline

    try:
        events = get_case_timeline(run_id)
        return {"run_id": run_id, "events": events}
    except Exception as e:
        logger.error(f"Error in api_case_timeline: {e}")
        return {"run_id": run_id, "events": [], "error": str(e)}


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
            "ingest": "/api/ingest",
            "chat": "/api/chat",
            "health": "/api/health",
            "case_summary": "/api/case-summary",
        },
    }
