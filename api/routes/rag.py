"""
RAG query, indexing, and infrastructure API routes.
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

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
)

router = APIRouter()

# ── Case Management & Deletion ────────────────────────────────────────────────


@router.post("/cases/delete", response_model=MessageResponse, summary="Delete indexed cases")
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
            import shutil

            from settings_manager import WORKSPACE_DIR
            bundled_ws = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")
            fallback_ws = os.path.join(os.path.expanduser("~"), ".local", "share", "kirag", "workspace")
            for ws in [WORKSPACE_DIR, bundled_ws, fallback_ws]:
                if ws and os.path.exists(ws):
                    try:
                        for name in os.listdir(ws):
                            if name == run_id or (name.startswith("run_") and run_id in name):
                                target = os.path.join(ws, name)
                                if os.path.isdir(target):
                                    shutil.rmtree(target, ignore_errors=True)
                    except Exception:
                        pass

        if req.delete_all:
            runs = get_all_runs()
            count = 0
            for r in runs:
                rid = r.get("run_id")
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

            return MessageResponse(success=True, message=f"Deleted all {count} case(s) successfully.")
        elif req.run_ids:
            count = 0
            for rid in req.run_ids:
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

            try:
                invalidate_query_cache()
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")

            return MessageResponse(success=True, message=f"Deleted {count} case(s).")
        else:
            return MessageResponse(success=False, message="No run_ids provided to delete.")
    except Exception as e:
        return MessageResponse(success=False, message=f"Error deleting cases: {e}")



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
    except Exception as e:
        return EmbeddingTelemetryResponse(redis_cached_count=f"Error: {e}")


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
    return MessageResponse(success="✅" in msg, message=msg)


@router.post(
    "/embedding/purge-cache", response_model=MessageResponse, summary="Purge Redis embedding cache"
)
def purge_cache():
    """Purge Redis vector cache."""
    from embedding_pipeline_ui import purge_embedding_cache

    msg = purge_embedding_cache()
    return MessageResponse(success="✅" in msg, message=msg)


# ── Chat Export ───────────────────────────────────────────────────────────────


@router.post("/export", summary="Export chat session")
def export_chat_session(req: ExportChatRequest):
    """Export a chat session into MD, TXT, CSV, DOCX, or Timeline DOCX format."""
    from fastapi.responses import FileResponse

    from rag_export import (
        export_chat_csv,
        export_chat_docx,
        export_chat_markdown,
        export_chat_text,
        export_timeline_docx,
    )

    fmt = req.export_format.lower()
    case_label = req.case_id or "All Cases"

    if fmt == "md":
        path = export_chat_markdown(req.history, mode=req.mode, active_case=case_label)
    elif fmt == "txt":
        path = export_chat_text(req.history, mode=req.mode, active_case=case_label)
    elif fmt == "csv":
        path = export_chat_csv(req.history, active_case=case_label)
    elif fmt == "docx":
        path = export_chat_docx(req.history, mode=req.mode, active_case=case_label)
    elif fmt == "timeline_docx":
        path = export_timeline_docx(req.history, active_case=case_label)
    else:
        return MessageResponse(success=False, message=f"Unsupported format: {fmt}")

    if not path or not os.path.exists(path):
        return MessageResponse(success=False, message="Failed to generate export file.")

    filename = os.path.basename(path)
    return FileResponse(path, filename=filename)


# ── Infrastructure ────────────────────────────────────────────────────────────


@router.post("/infra/start", response_model=MessageResponse, summary="Start RAG infrastructure")
def start_infra():
    """Start PostgreSQL, Redis, MinIO, Qdrant via Docker Compose and initialize schemas."""
    from rag_infra_manager import start_and_init_rag

    success, msg = start_and_init_rag()
    return MessageResponse(success=success, message=msg)


@router.post("/infra/stop", response_model=MessageResponse, summary="Stop RAG infrastructure")
def stop_infra():
    """Stop all RAG infrastructure services."""
    from rag_infra_manager import stop_rag_infrastructure

    success, msg = stop_rag_infrastructure()
    return MessageResponse(success=success, message=msg)


@router.get("/infra/status", response_model=InfraStatusResponse, summary="Infrastructure status")
def infra_status():
    """Return the status of each RAG infrastructure service."""
    from rag_infra_manager import get_rag_service_status

    statuses = get_rag_service_status()
    return InfraStatusResponse(**statuses)


@router.post("/query", summary="Query the RAG system")
def rag_query(req: RAGQueryRequest):
    """Run a RAG analysis query.

    If ``stream=True`` (default), returns an SSE stream of text chunks.
    Otherwise returns the complete response as JSON.
    """
    from rag.analyzer import ANALYSIS_MODE_MAP, analyze

    mode_key = ANALYSIS_MODE_MAP.get(req.mode, req.mode)

    if req.stream:

        def event_generator():
            try:
                for chunk in analyze(
                    query=req.query,
                    mode=mode_key,
                    server_url=req.model_url,
                    model_name=req.model_name,
                    top_k=req.top_k,
                    run_id_filter=req.case_id,
                    doc_type_filter=req.doc_type,
                    author_filter=req.author,
                    date_from=req.date_from,
                    date_to=req.date_to,
                    stream=True,
                    use_reranker=req.use_reranker,
                    reranker_model=req.reranker_model,
                    reranker_device=req.reranker_device,
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Non-streaming: collect full response
        full_response = ""
        try:
            for chunk in analyze(
                query=req.query,
                mode=mode_key,
                server_url=req.model_url,
                model_name=req.model_name,
                top_k=req.top_k,
                run_id_filter=req.case_id,
                doc_type_filter=req.doc_type,
                author_filter=req.author,
                date_from=req.date_from,
                date_to=req.date_to,
                stream=True,
                use_reranker=req.use_reranker,
                reranker_model=req.reranker_model,
                reranker_device=req.reranker_device,
            ):
                full_response += chunk
        except Exception as e:
            return {"error": str(e)}
        return {"response": full_response}


# ── Indexing ──────────────────────────────────────────────────────────────────


@router.post("/index", summary="Index a specific run")
def index_run(req: IndexRunRequest):
    """Index a single OCR run into the RAG corpus."""
    from indexing_service import CorpusIndexingService

    messages = list(CorpusIndexingService.index_run(req.run_dir))
    return MessageResponse(
        success=any("✅" in m or "Done" in m for m in messages),
        message="\n".join(messages),
    )


@router.post("/index-all", summary="Index all available runs")
def index_all_runs():
    """Index all available OCR runs into the RAG corpus."""
    from indexing_service import CorpusIndexingService

    messages = list(CorpusIndexingService.index_all_runs())
    return MessageResponse(
        success=any("✅" in m or "Done" in m for m in messages),
        message="\n".join(messages),
    )


@router.post("/upload-markdown", summary="Upload external markdown files")
async def upload_markdown(
    files: list[UploadFile] = File(...),
    case_option: str = Form("new"),
    new_case_name: str = Form(""),
):
    """Upload and index external markdown files into the corpus."""
    import os
    import tempfile

    from indexing_service import CorpusIndexingService

    # Write uploaded files to a temp dir so the indexer can read them.
    # Sanitise the client-supplied filename via basename so a path like
    # "../../evil.md" cannot escape the temp dir and overwrite arbitrary files.
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for upload in files:
            safe_name = os.path.basename(upload.filename or "").strip()
            if not safe_name or "/" in safe_name or "\\" in safe_name or ".." in safe_name:
                continue
            path = os.path.join(temp_dir, safe_name)
            content = await upload.read()
            with open(path, "wb") as f:
                f.write(content)
            saved_paths.append(path)

        messages = list(
            CorpusIndexingService.add_markdown_to_case(saved_paths, case_option, new_case_name)
        )
        return MessageResponse(
            success=any("✅" in m or "Done" in m for m in messages),
            message="\n".join(messages),
        )
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


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
    except Exception:
        return CorpusStatsResponse(indexed_runs=-1, total_chunks=-1)


@router.get("/corpus/cases", response_model=list[CaseInfo], summary="List indexed cases")
def list_cases():
    """Return a list of indexed cases with their run IDs."""
    try:
        from rag.db import get_indexed_runs

        runs = get_indexed_runs()
        return [
            CaseInfo(label=r.get("display_name", r["run_id"]), run_id=r["run_id"]) for r in runs
        ]
    except Exception:
        return []
