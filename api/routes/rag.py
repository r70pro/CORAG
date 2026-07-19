"""
RAG query, indexing, and infrastructure API routes.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from api.models import (
    CaseInfo,
    CorpusStatsResponse,
    IndexRunRequest,
    InfraStatusResponse,
    MessageResponse,
    RAGQueryRequest,
)

router = APIRouter()


# ── Query ─────────────────────────────────────────────────────────────────────


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

        return CorpusStatsResponse(
            indexed_runs=db_stats.get("indexed_runs", 0),
            indexed_documents=db_stats.get("indexed_documents", 0),
            total_chunks=db_stats.get("total_chunks", 0),
            unique_authors=db_stats.get("unique_authors", 0),
            earliest_date=db_stats.get("earliest_date"),
            latest_date=db_stats.get("latest_date"),
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
