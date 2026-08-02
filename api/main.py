"""
KIRAG REST API — FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

OpenAPI docs:
    http://localhost:8001/docs
"""

from __future__ import annotations

import logging
import os
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.auth import requested_api_bind_host, require_safe_bind, verify_api_key
from api.errors import default_error_code, error_response
from api.models import ErrorEnvelope, PipelineStartRequest, RAGQueryRequest
from api.upload_security import UploadRequestLimitMiddleware
from runtime_logging import configure_runtime_logging
from settings_manager import VERSION

configure_runtime_logging("api")
logger = logging.getLogger(__name__)


def _inference_endpoint_ready(server_url: str) -> bool:
    try:
        with urllib.request.urlopen(server_url.rstrip("/") + "/models", timeout=2) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    require_safe_bind(requested_api_bind_host())
    logger.info(f"Starting KIRAG API v{VERSION}...")
    try:
        from analysis_profiles import resume_pending_switch
        if resume_pending_switch():
            logger.warning("Resumed an interrupted analysis model switch")
    except Exception as exc:
        logger.error("Unable to recover analysis switch state: %s", exc)
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
    responses={
        status_code: {
            "model": ErrorEnvelope,
            "description": description,
        }
        for status_code, description in {
            400: "Bad request",
            401: "Authentication required",
            403: "Insufficient authorization",
            404: "Resource not found",
            413: "Payload too large",
            415: "Unsupported media type",
            422: "Request validation failed",
            500: "Internal server error",
            502: "Upstream service error",
            503: "Service unavailable",
        }.items()
    },
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
app.add_middleware(UploadRequestLimitMiddleware)


def _authentication_error(request: Request):
    """Return a typed authentication error, or ``None`` when access is allowed."""

    if request.url.path in {"/health", "/livez", "/readyz", "/inference/ready"} or request.method == "OPTIONS":
        return None

    auth_header = request.headers.get("authorization", "")
    bearer = None
    if auth_header.lower().startswith("bearer "):
        bearer = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=auth_header.split(" ", 1)[1]
        )
    try:
        verify_api_key(
            key_from_header=(
                request.headers.get("x-api-key") or request.headers.get("x-admin-api-key")
            ),
            bearer=bearer,
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            detail = exc.detail
            message = detail if isinstance(detail, str) else "Authentication failed"
            return error_response(
                exc.status_code,
                default_error_code(exc.status_code),
                message,
                headers=exc.headers,
            )
        raise
    return None


async def require_authentication(request: Request, call_next):
    """Leave only the minimal liveness endpoint unauthenticated."""

    rejection = _authentication_error(request)
    if rejection is not None:
        return rejection
    return await call_next(request)


class AuthenticationMiddleware:
    """Pure ASGI authentication middleware.

    Avoiding ``BaseHTTPMiddleware`` keeps early authentication responses from
    waiting on a request-body task that was never started.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            rejection = _authentication_error(Request(scope))
            if rejection is not None:
                await rejection(scope, receive, send)
                return
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Apply browser hardening to successful and error responses."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Content-Security-Policy"] = (
                    "default-src 'none'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'none'"
                )
                headers["X-Frame-Options"] = "DENY"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = (
                    "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
                )
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Cross-Origin-Resource-Policy"] = "same-origin"
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


app.add_middleware(AuthenticationMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    code = default_error_code(exc.status_code)
    message = detail if isinstance(detail, str) else "Request failed"
    details = None
    if isinstance(detail, dict):
        code = str(detail.get("code") or code)
        message = str(detail.get("message") or message)
        details = detail.get("details")
    return error_response(
        exc.status_code,
        code,
        message,
        details=details,
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "location": [str(item) for item in error.get("loc", ())],
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return error_response(
        422,
        "validation_error",
        "Request validation failed",
        details=details,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server error processing request", exc_info=True)
    return error_response(500, "internal_error", "Internal Server Error")


# ── Mount route modules ──────────────────────────────────────────────────────

from api.routes import diagnostics, docker, documents, pipeline, rag, settings, system  # noqa: E402

app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(docker.router, prefix="/api/docker", tags=["Docker"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["Diagnostics"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(system.router, prefix="/api/system", tags=["System"])


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
    return {"status": "ok"}


@app.get("/livez", include_in_schema=False, dependencies=[])
def liveness_check():
    """Process liveness only; safe for a supervisor restart probe."""
    return {"status": "alive", "version": VERSION}


@app.get("/readyz", include_in_schema=False, dependencies=[])
def readiness_check():
    """Return 503 until core data services are usable; inference is feature-gated."""
    from system_diagnostics import check_backing_services_data

    backing = check_backing_services_data()
    failed_core = [
        name for name in backing.get("failed_services", []) if not name.startswith("vllm_")
    ]
    if failed_core:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "failed_services": sorted(
                    set(failed_core)
                ),
            },
        )
    return {"status": "ready"}


@app.get("/inference/ready", include_in_schema=False, dependencies=[])
def inference_readiness_check():
    """Report role-specific inference readiness without blocking the app shell."""
    from settings_manager import load_settings
    settings = load_settings()
    roles = {
        "ocr": _inference_endpoint_ready(settings.get("server_url", "http://127.0.0.1:8000/v1")),
        "analysis": _inference_endpoint_ready(settings.get("analysis_server_url", "http://127.0.0.1:8002/v1")),
    }
    return {"status": "ready" if all(roles.values()) else "degraded", "roles": roles}


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
        vllm_models = backing.get("vllm_models", {}) if isinstance(backing, dict) else {}
        vllm_progress = backing.get("vllm_progress") if isinstance(backing, dict) else None
        failed_services = backing.get("failed_services", []) if isinstance(backing, dict) else []

        status_str = (
            "healthy"
            if all_healthy
            else (
                "loading"
                if failed_services
                and all(name.startswith("vllm_") for name in failed_services)
                and any((vllm_progress or {}).values())
                else "degraded"
            )
        )

        return {
            "status": status_str,
            "all_healthy": all_healthy,
            "failed_services": failed_services,
            "vllm_model": vllm_model,
            "vllm_models": vllm_models,
            "vllm_progress": vllm_progress,
            "services": services,
            "gpu": gpu,
            "gpu_metrics": gpu,
        }
    except Exception as e:
        logger.error(f"Error in api_health check: {e}")
        raise HTTPException(status_code=503, detail="System health is unavailable") from e


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
            client_name = ", ".join(names) if names else "Patient name not present in source"
            dob = meta.get("dob", "Not present in source")
            injuries = meta.get("injuries", [])

            earliest = r.get("earliest_date")
            latest = r.get("latest_date")
            date_range = "Not present in source"
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
                    "dob_unparsed_raw": meta.get("dob_unparsed_raw", []),
                    "injuries": injuries if injuries else ["Not present in source"],
                    "documents_count": r.get("total_documents", 0),
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
        raise HTTPException(status_code=500, detail="Unable to retrieve case summary") from e


@app.get("/api/cases/{run_id}/timeline", tags=["Phase 1 Core Endpoints"])
def api_case_timeline(run_id: str):
    """Return chronological medicolegal timeline events for a specific run ID."""
    from rag.metadata_helper import get_case_timeline

    try:
        events = get_case_timeline(run_id)
        return {"run_id": run_id, "events": events}
    except Exception as e:
        logger.error(f"Error in api_case_timeline: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve timeline") from e


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
            "system": "/api/system",
            "ingest": "/api/ingest",
            "chat": "/api/chat",
            "health": "/api/health",
            "case_summary": "/api/case-summary",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    api_host = require_safe_bind(requested_api_bind_host())
    uvicorn.run("api.main:app", host=api_host, port=8001)
