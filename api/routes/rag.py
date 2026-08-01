"""
RAG query, indexing, and infrastructure API routes.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.auth import has_admin_access, require_remote_lifecycle_enabled, verify_admin_key
from api.errors import error_envelope
from api.models import (
    CaseInfo,
    CorpusStatsResponse,
    DeleteCasesRequest,
    EmbeddingConfigRequest,
    EmbeddingTelemetryResponse,
    ExportChatRequest,
    IndexRunRequest,
    InfraStatusResponse,
    MessageResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from api.upload_security import (
    UPLOAD_CHUNK_BYTES,
    LimitedUploadRoute,
    MarkdownValidator,
    close_uploads,
    escaped_original_name,
    markdown_upload_limits,
    require_content_type,
    unique_upload_name,
)
from path_security import (
    PathSecurityError,
    require_approved_file,
    resolve_file_under,
    resolve_run_under,
)

router = APIRouter(route_class=LimitedUploadRoute)
logger = logging.getLogger(__name__)


@router.delete("/chat/history/{session_id}", response_model=MessageResponse, summary="Delete RAG chat history")
def delete_chat_history(session_id: str):
    """Permanently remove the server-side cache for one browser chat thread."""
    if not session_id or len(session_id) > 128 or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id):
        raise HTTPException(status_code=422, detail="Invalid chat session ID")
    try:
        from rag.cache import clear_chat_history

        clear_chat_history(session_id)
        return MessageResponse(success=True, message="Chat thread deleted")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Unable to delete chat thread") from exc

# ── Case Management & Deletion ────────────────────────────────────────────────


@router.post(
    "/cases/delete",
    response_model=MessageResponse,
    summary="Delete indexed cases",
    dependencies=[Depends(verify_admin_key)],
)
def delete_cases(req: DeleteCasesRequest):
    """Delete selected or all cases from Qdrant vector store, PostgreSQL DB, and MinIO storage."""
    try:
        import logging

        from rag.cache import invalidate_query_cache
        from rag.db import delete_run_data, get_all_runs
        from rag.embedding import delete_collection, delete_run_vectors, init_collection
        from rag.storage import delete_run_objects

        logger = logging.getLogger(__name__)

        def _purge_run_dir_from_disk(run_id: str):
            from settings_manager import delete_run_directory

            delete_run_directory(run_id)

        def _purge_all_run_dirs_from_disk():
            import shutil

            from settings_manager import WORKSPACE_DIR

            workspace = Path(WORKSPACE_DIR).resolve()
            if not workspace.is_dir():
                return
            for entry in workspace.iterdir():
                try:
                    target = resolve_run_under(workspace, entry.name)
                except PathSecurityError:
                    continue
                if target.is_dir():
                    shutil.rmtree(target)

        def _valid_run_id(run_id: object) -> str | None:
            if not isinstance(run_id, str):
                return None
            if (
                not run_id
                or len(run_id) > 128
                or "\x00" in run_id
                or "/" in run_id
                or "\\" in run_id
                or Path(run_id).is_absolute()
                or not all(char.isalnum() or char in "_-" for char in run_id)
            ):
                return None
            return run_id

        if req.delete_all:
            runs = get_all_runs()
            count = 0
            for r in runs:
                rid = _valid_run_id(r.get("run_id"))
                if rid:
                    try:
                        delete_run_vectors(rid)
                    except Exception as e:
                        logger.warning(f"Vector delete error for {rid}: {e}")
                    try:
                        delete_run_data(rid)
                    except Exception as e:
                        logger.warning(f"DB delete error for {rid}: {e}")
                    try:
                        delete_run_objects(rid)
                    except Exception as e:
                        logger.warning(f"Storage delete error for {rid}: {e}")
                    try:
                        _purge_run_dir_from_disk(rid)
                    except Exception as e:
                        logger.warning(f"Disk purge error for {rid}: {e}")
                    count += 1

            # Sweep all remaining run_* directories from disk workspace
            try:
                _purge_all_run_dirs_from_disk()
            except Exception as e:
                logger.warning(f"Full disk sweep error: {e}")

            # Invalidate query cache
            try:
                invalidate_query_cache()
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")

            # Re-init vector collection to ensure complete wipe of orphaned points
            try:
                delete_collection()
                init_collection()
            except Exception as e:
                logger.warning(f"Collection wipe warning: {e}")

            return MessageResponse(
                success=True, message=f"Deleted all {count} case(s) successfully."
            )
        elif req.run_ids:
            validated_run_ids = []
            for raw_run_id in req.run_ids:
                run_id = _valid_run_id(raw_run_id)
                if run_id is None:
                    raise HTTPException(status_code=400, detail="Invalid run identifier")
                validated_run_ids.append(run_id)
            count = 0
            for rid in validated_run_ids:
                try:
                    delete_run_vectors(rid)
                except Exception as e:
                    logger.warning(f"Vector delete error for {rid}: {e}")
                try:
                    delete_run_data(rid)
                except Exception as e:
                    logger.warning(f"DB delete error for {rid}: {e}")
                try:
                    delete_run_objects(rid)
                except Exception as e:
                    logger.warning(f"Storage delete error for {rid}: {e}")
                try:
                    _purge_run_dir_from_disk(rid)
                except Exception as e:
                    logger.warning(f"Disk purge error for {rid}: {e}")
                count += 1

            try:
                invalidate_query_cache()
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")

            return MessageResponse(success=True, message=f"Deleted {count} case(s).")
        else:
            raise HTTPException(status_code=400, detail="No run_ids provided to delete")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Case deletion failed")
        raise HTTPException(status_code=500, detail="Error deleting cases")


# ── Embedding & Vector Store ──────────────────────────────────────────────────


@router.get(
    "/embedding/telemetry",
    response_model=EmbeddingTelemetryResponse,
    summary="Get embedding telemetry",
)
def get_embedding_telemetry():
    """Return live telemetry for Qdrant vector store and Redis embedding cache."""
    try:
        from embedding_pipeline_ui import get_embedding_telemetry_html
        from rag.embedding import get_collection_info, get_collection_name
        from settings_manager import load_settings

        settings = load_settings()
        model_name = settings.get("embedding_model", "BAAI/bge-large-en-v1.5")
        device = settings.get("embedding_device", "auto")
        col_name = get_collection_name(model_name)
        info = get_collection_info(model_name)

        active_device = device
        if device == "auto" or not device:
            try:
                import torch

                active_device = "CUDA GPU" if torch.cuda.is_available() else "CPU Mode"
            except Exception:
                active_device = "CPU Mode"

        cached_count = "N/A"
        try:
            import rag.cache as cache

            if cache.is_healthy():
                info_cache = cache.get_cache_info()
                cached_count = f"{info_cache.get('cached_embeddings', 0)} vectors"
        except Exception:
            pass

        html_val = get_embedding_telemetry_html()

        return EmbeddingTelemetryResponse(
            active_device=str(active_device),
            device_target=device,
            qdrant_points=info.get("points_count", 0),
            collection_name=col_name,
            vector_dim=1024,
            metric="Cosine Similarity",
            redis_cached_count=cached_count,
            telemetry_html=html_val,
        )
    except Exception as exc:
        logger.exception("Embedding telemetry failed")
        raise HTTPException(status_code=503, detail="Embedding telemetry is unavailable") from exc


@router.post(
    "/embedding/config", response_model=MessageResponse, summary="Save embedding configuration"
)
def save_embedding_config(req: EmbeddingConfigRequest):
    """Save chunking and embedding configuration."""
    from embedding_pipeline_ui import save_embedding_pipeline_settings

    msg = save_embedding_pipeline_settings(
        req.embedding_model,
        req.embedding_device,
        req.chunk_size,
        req.chunk_overlap,
        req.embedding_batch_size,
    )
    if "✅" not in msg:
        raise HTTPException(status_code=500, detail="Unable to save embedding configuration")
    return MessageResponse(success=True, message=msg)


@router.post(
    "/embedding/purge-cache",
    response_model=MessageResponse,
    summary="Purge Redis embedding cache",
    dependencies=[Depends(verify_admin_key)],
)
def purge_cache():
    """Purge Redis vector cache."""
    from embedding_pipeline_ui import purge_embedding_cache

    msg = purge_embedding_cache()
    if "✅" not in msg:
        raise HTTPException(status_code=500, detail="Unable to purge embedding cache")
    return MessageResponse(success=True, message=msg)


# ── Chat Export ───────────────────────────────────────────────────────────────


@router.post("/export", summary="Export chat session")
def export_chat_session(req: ExportChatRequest, is_admin: bool = Depends(has_admin_access)):
    """Export a chat session into MD, TXT, CSV, DOCX, or Timeline DOCX format."""
    from fastapi.responses import FileResponse

    from rag_export import (
        export_chat_docx,
        export_chat_markdown,
        export_chat_text,
        export_timeline_csv,
        export_timeline_docx,
    )

    if req.include_reasoning and not is_admin:
        raise HTTPException(status_code=403, detail="Administrative authorization required")
    history = req.history
    if not (req.include_reasoning and is_admin):
        history = [
            {key: value for key, value in item.items() if key != "reasoning"}
            if isinstance(item, dict) else item
            for item in history
        ]
    fmt = req.export_format.lower()
    case_label = req.case_id or "All Cases"

    if fmt == "md":
        path = export_chat_markdown(history, mode=req.mode, active_case=case_label, include_reasoning=req.include_reasoning)
    elif fmt == "txt":
        path = export_chat_text(history, mode=req.mode, active_case=case_label, include_reasoning=req.include_reasoning)
    elif fmt == "csv":
        path = export_timeline_csv(req.history, active_case=case_label)
    elif fmt == "docx":
        path = export_chat_docx(history, mode=req.mode, active_case=case_label, include_reasoning=req.include_reasoning)
    elif fmt == "timeline_docx":
        path = export_timeline_docx(req.history, active_case=case_label)
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported export format: {fmt}")

    if not path:
        raise HTTPException(status_code=500, detail="Failed to generate export file")

    from rag_export import EXPORT_DIR

    try:
        safe_path = require_approved_file(path, {EXPORT_DIR}, {".md", ".txt", ".csv", ".docx"})
    except PathSecurityError as exc:
        raise HTTPException(status_code=500, detail="Failed to generate export file") from exc
    if not safe_path.is_file():
        raise HTTPException(status_code=500, detail="Failed to generate export file")
    return FileResponse(safe_path, filename=safe_path.name)


# ── Infrastructure ────────────────────────────────────────────────────────────


@router.post(
    "/infra/start",
    response_model=MessageResponse,
    summary="Start RAG infrastructure",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
def start_infra():
    """Start PostgreSQL, Redis, MinIO, Qdrant via Docker Compose and initialize schemas."""
    from rag_infra_manager import start_and_init_rag

    success, msg = start_and_init_rag()
    if not success:
        raise HTTPException(status_code=503, detail=msg or "Unable to start RAG infrastructure")
    return MessageResponse(success=success, message=msg)


@router.post(
    "/infra/stop",
    response_model=MessageResponse,
    summary="Stop RAG infrastructure",
    dependencies=[Depends(verify_admin_key), Depends(require_remote_lifecycle_enabled)],
)
def stop_infra():
    """Stop all RAG infrastructure services."""
    from rag_infra_manager import stop_rag_infrastructure

    success, msg = stop_rag_infrastructure()
    if not success:
        raise HTTPException(status_code=500, detail=msg or "Unable to stop RAG infrastructure")
    return MessageResponse(success=success, message=msg)


@router.get("/infra/status", response_model=InfraStatusResponse, summary="Infrastructure status")
def infra_status():
    """Return the status of each RAG infrastructure service."""
    from rag_infra_manager import get_rag_service_status

    statuses = get_rag_service_status()
    return InfraStatusResponse(**statuses)


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream or complete JSON response",
        }
    },
    summary="Query the RAG system",
)
def rag_query(req: RAGQueryRequest, is_admin: bool = Depends(has_admin_access)):
    """Run a RAG analysis query.

    If ``stream=True`` (default), returns an SSE stream of text chunks.
    Otherwise returns the complete response as JSON.
    """
    from rag.analyzer import ANALYSIS_MODE_MAP, ContextWindowError, analyze

    mode_key = ANALYSIS_MODE_MAP.get(req.mode, req.mode)
    is_general_knowledge = mode_key == "general_knowledge"
    expose_reasoning = bool(is_admin and req.reasoning_audit)
    date_from = req.date_from.isoformat() if req.date_from else None
    date_to = req.date_to.isoformat() if req.date_to else None

    chat_history: list[dict] = []
    if req.session_id:
        try:
            from rag.cache import get_chat_history

            chat_history = get_chat_history(req.session_id)
            if is_general_knowledge:
                # Do not expose case-grounded answers from a previous mode to
                # the no-document conversation. Preserve only the latest
                # contiguous General Knowledge exchange.
                matching_history = []
                for message in reversed(chat_history):
                    if message.get("mode") != "general_knowledge":
                        break
                    matching_history.append(message)
                chat_history = list(reversed(matching_history))
            chat_history = [
                {"role": message["role"], "content": message["content"]}
                for message in chat_history
                if message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            ]
        except Exception:
            logger.exception("Unable to load RAG chat history")

    if req.stream:

        def event_generator():
            # The analyzer and local model are synchronous. Run them in a worker so
            # progress can be delivered immediately and keep-alives can cross every
            # proxy while a non-streaming model call is in progress.
            events: queue.Queue[tuple[str, object]] = queue.Queue()

            def report_progress(progress: float, message: str):
                normalized_message = message.lower()
                if "generating" in normalized_message:
                    stage = "generating"
                elif "preparing" in normalized_message:
                    stage = "preparing"
                else:
                    stage = "retrieving"
                events.put(
                    ("status", {"stage": stage, "progress": progress, "message": message})
                )

            def produce():
                try:
                    reasoning_audit_parts: list[str] = []
                    answer_parts: list[str] = []
                    def report_reasoning(chunk: str):
                        if expose_reasoning:
                            reasoning_audit_parts.append(chunk)
                            events.put(("reasoning", chunk))

                    for chunk in analyze(
                        query=req.query,
                        mode=mode_key,
                        server_url=req.model_url,
                        model_name=req.model_name,
                        top_k=req.top_k,
                        chat_history=chat_history,
                        run_id_filter=req.case_id,
                        doc_type_filter=req.doc_type,
                        author_filter=req.author,
                        date_from=date_from,
                        date_to=date_to,
                        stream=True,
                        progress_callback=report_progress,
                        reasoning_callback=report_reasoning,
                        use_reranker=req.use_reranker,
                        reranker_model=req.reranker_model,
                        reranker_device=req.reranker_device,
                    ):
                        answer_parts.append(chunk)
                        events.put(("content", chunk))
                    if req.session_id:
                        try:
                            from rag.cache import get_chat_history, save_chat_history
                            saved = get_chat_history(req.session_id)
                            saved.append(
                                {"role": "user", "content": req.query, "mode": mode_key}
                            )
                            assistant_message = {
                                "role": "assistant",
                                "content": "".join(answer_parts),
                                "mode": mode_key,
                            }
                            if expose_reasoning and reasoning_audit_parts:
                                assistant_message["reasoning"] = "".join(reasoning_audit_parts)
                            saved.append(assistant_message)
                            save_chat_history(req.session_id, saved)
                        except Exception:
                            logger.exception("Unable to persist RAG chat history")
                    if reasoning_audit_parts:
                        from audit_log import audit_event
                        audit_event(
                            "llm_reasoning_audit",
                            "success",
                            case_id=req.case_id,
                            mode=mode_key,
                            model=req.model_name,
                            reasoning="".join(reasoning_audit_parts),
                        )
                    events.put(("done", None))
                except Exception as exc:
                    events.put(("error", exc))

            threading.Thread(target=produce, name="rag-query-stream", daemon=True).start()
            starting_message = (
                "Starting general-knowledge chat without document retrieval…"
                if is_general_knowledge
                else "Starting RAG analysis…"
            )
            yield f"data: {json.dumps({'type': 'status', 'stage': 'starting', 'message': starting_message, 'progress': 0.0})}\n\n"

            while True:
                try:
                    event_type, payload = events.get(timeout=10.0)
                except queue.Empty:
                    # SSE comments are ignored by clients but prevent idle proxy and
                    # browser connections from being dropped during long inference.
                    yield ": keep-alive\n\n"
                    continue

                if event_type == "status":
                    status = payload
                    yield f"data: {json.dumps({'type': 'status', **status})}\n\n"
                elif event_type == "content":
                    yield f"data: {json.dumps({'type': 'content', 'chunk': payload})}\n\n"
                elif event_type == "reasoning":
                    yield f"data: {json.dumps({'type': 'reasoning', 'chunk': payload})}\n\n"
                elif event_type == "done":
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'complete', 'message': 'Analysis complete.', 'progress': 1.0})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                elif event_type == "error":
                    exc = payload
                    if isinstance(exc, ContextWindowError):
                        envelope = error_envelope("context_window_exceeded", str(exc))
                    else:
                        logger.error("Streaming RAG query failed", exc_info=(type(exc), exc, exc.__traceback__))
                        envelope = error_envelope("rag_query_failed", "RAG query failed")
                    yield f"data: {envelope.model_dump_json(exclude_none=True)}\n\n"
                    break

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )
    else:
        # Non-streaming: collect full response
        full_response = ""
        full_reasoning = ""
        def collect_reasoning(chunk: str):
            nonlocal full_reasoning
            if expose_reasoning:
                full_reasoning += chunk
        try:
            for chunk in analyze(
                query=req.query,
                mode=mode_key,
                server_url=req.model_url,
                model_name=req.model_name,
                top_k=req.top_k,
                chat_history=chat_history,
                run_id_filter=req.case_id,
                doc_type_filter=req.doc_type,
                author_filter=req.author,
                date_from=date_from,
                date_to=date_to,
                stream=False,
                use_reranker=req.use_reranker,
                reasoning_callback=collect_reasoning,
                reranker_model=req.reranker_model,
                reranker_device=req.reranker_device,
            ):
                full_response += chunk
        except ContextWindowError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "context_window_exceeded",
                    "message": str(exc),
                },
            ) from exc
        except Exception as exc:
            logger.exception("RAG query failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "rag_query_failed",
                    "message": "RAG query failed",
                },
            ) from exc
        if full_reasoning:
            from audit_log import audit_event
            audit_event(
                "llm_reasoning_audit", "success", case_id=req.case_id,
                mode=mode_key, model=req.model_name, reasoning=full_reasoning,
            )
        if req.session_id:
            try:
                from rag.cache import get_chat_history, save_chat_history
                saved = get_chat_history(req.session_id)
                saved.append({"role": "user", "content": req.query, "mode": mode_key})
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "mode": mode_key,
                }
                if expose_reasoning and full_reasoning:
                    assistant_message["reasoning"] = full_reasoning
                saved.append(assistant_message)
                save_chat_history(req.session_id, saved)
            except Exception:
                logger.exception("Unable to persist RAG chat history")
        return RAGQueryResponse(response=full_response, reasoning=full_reasoning or None)


# ── Indexing ──────────────────────────────────────────────────────────────────


def _indexing_event_stream(messages):
    """Serialize indexing generator updates as promptly flushed SSE events."""
    try:
        succeeded = False
        for message in messages:
            succeeded = succeeded or "✅" in message or "Done" in message
            yield f"data: {json.dumps({'message': message})}\n\n"
        if succeeded:
            yield "data: [DONE]\n\n"
        else:
            envelope = error_envelope("indexing_failed", "Unable to complete indexing")
            yield f"data: {envelope.model_dump_json(exclude_none=True)}\n\n"
    except Exception:
        logger.exception("Streaming indexing operation failed")
        envelope = error_envelope("indexing_failed", "Indexing operation failed")
        yield f"data: {envelope.model_dump_json(exclude_none=True)}\n\n"


@router.post("/index", summary="Index a specific run", dependencies=[Depends(verify_admin_key)])
def index_run(req: IndexRunRequest):
    """Index a single OCR run into the RAG corpus."""
    from indexing_service import CorpusIndexingService
    from settings_manager import WORKSPACE_DIR

    try:
        run_dir = resolve_run_under(WORKSPACE_DIR, req.run_dir)
    except PathSecurityError as exc:
        raise HTTPException(status_code=400, detail="Invalid run name") from exc
    if not run_dir.is_dir():
        raise HTTPException(status_code=400, detail="Run not found")
    messages = list(CorpusIndexingService.index_run(str(run_dir), force=True))
    success = any("✅" in m or "Done" in m for m in messages)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to index run")
    return MessageResponse(success=True, message="\n".join(messages))


@router.post(
    "/index/stream",
    summary="Index a specific run with progress streaming",
    dependencies=[Depends(verify_admin_key)],
)
def stream_index_run(req: IndexRunRequest):
    """Index one OCR run while streaming generator progress to the client."""
    from indexing_service import CorpusIndexingService
    from settings_manager import WORKSPACE_DIR

    try:
        run_dir = resolve_run_under(WORKSPACE_DIR, req.run_dir)
    except PathSecurityError as exc:
        raise HTTPException(status_code=400, detail="Invalid run name") from exc
    if not run_dir.is_dir():
        raise HTTPException(status_code=400, detail="Run not found")
    return StreamingResponse(
        _indexing_event_stream(CorpusIndexingService.index_run(str(run_dir), force=True)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/index-all", summary="Index all available runs", dependencies=[Depends(verify_admin_key)]
)
def index_all_runs():
    """Index all available OCR runs into the RAG corpus."""
    from indexing_service import CorpusIndexingService

    messages = list(CorpusIndexingService.index_all_runs(force=True))
    success = any("✅" in m or "Done" in m for m in messages)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to index runs")
    return MessageResponse(success=True, message="\n".join(messages))


@router.post(
    "/index-all/stream",
    summary="Index all available runs with progress streaming",
    dependencies=[Depends(verify_admin_key)],
)
def stream_index_all_runs():
    """Index all OCR runs while streaming generator progress to the client."""
    from indexing_service import CorpusIndexingService

    return StreamingResponse(
        _indexing_event_stream(CorpusIndexingService.index_all_runs(force=True)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/upload-markdown", summary="Upload external markdown files")
async def upload_markdown(
    files: list[UploadFile] | UploadFile | None = File(default=None),
    case_option: str = Form("new"),
    new_case_name: str = Form(""),
):
    """Upload and index external markdown files into the corpus."""
    import tempfile

    from indexing_service import CorpusIndexingService

    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    file_list = files if isinstance(files, list) else [files]
    limits = markdown_upload_limits()
    if len(file_list) > limits.max_files:
        await close_uploads(file_list)
        raise HTTPException(status_code=413, detail="Too many uploaded files")

    temp_dir = tempfile.mkdtemp(prefix="kirag-markdown-")
    saved_files = []
    aggregate_bytes = 0
    try:
        for upload in file_list:
            original_name = escaped_original_name(upload, ".md")
            require_content_type(
                upload,
                {"text/markdown", "text/plain", "application/octet-stream"},
            )
            stored_name = unique_upload_name(".md")
            path = resolve_file_under(temp_dir, stored_name, {".md"})
            validator = MarkdownValidator()
            file_bytes = 0
            with path.open("xb") as output:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    aggregate_bytes += len(chunk)
                    if file_bytes > limits.max_file_bytes:
                        raise HTTPException(
                            status_code=413, detail="Uploaded Markdown file is too large"
                        )
                    if aggregate_bytes > limits.max_total_bytes:
                        raise HTTPException(status_code=413, detail="Aggregate upload is too large")
                    validator.feed(chunk)
                    output.write(chunk)
            if file_bytes == 0:
                raise HTTPException(status_code=415, detail="Markdown upload is empty")
            validator.finish()
            saved_files.append(SimpleNamespace(name=str(path), original_filename=original_name))

        messages = list(
            CorpusIndexingService.add_markdown_to_case(saved_files, case_option, new_case_name)
        )
        success = any("✅" in m or "Done" in m for m in messages)
        if not success:
            raise HTTPException(status_code=500, detail="Unable to index Markdown files")
        return MessageResponse(success=True, message="\n".join(messages))
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        await close_uploads(file_list)


# ── Corpus ────────────────────────────────────────────────────────────────────


@router.get("/corpus/stats", response_model=CorpusStatsResponse, summary="Corpus statistics")
def corpus_stats():
    """Return aggregate statistics for the indexed corpus."""
    try:
        from rag.db import get_corpus_stats
        from rag.embedding import get_collection_info

        db_stats = get_corpus_stats()
        qdrant_info = get_collection_info()

        e_date = db_stats.get("earliest_date")
        l_date = db_stats.get("latest_date")
        return CorpusStatsResponse(
            indexed_runs=db_stats.get("indexed_runs", 0),
            indexed_documents=db_stats.get("indexed_documents", 0),
            total_chunks=db_stats.get("total_chunks", 0),
            unique_authors=db_stats.get("unique_authors", 0),
            earliest_date=str(e_date) if e_date else None,
            latest_date=str(l_date) if l_date else None,
            vectors_count=qdrant_info.get("points_count", 0),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Corpus statistics are unavailable") from exc


@router.get("/corpus/cases", response_model=list[CaseInfo], summary="List indexed cases")
def list_cases():
    """Return a list of indexed cases with their run IDs."""
    try:
        from rag.db import get_indexed_runs

        runs = get_indexed_runs()
        return [
            CaseInfo(label=r.get("display_name", r["run_id"]), run_id=r["run_id"]) for r in runs
        ]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Indexed cases are unavailable") from exc
