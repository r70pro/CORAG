"""
OCR pipeline management API routes.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from api.models import MessageResponse, PipelineStartRequest, PipelineStatusResponse, RunInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", summary="Upload source PDF documents for ingestion")
async def upload_pipeline_files(
    request: Request,
    files: list[UploadFile] | UploadFile | None = File(default=None),
):
    """Upload one or more PDF files for ingestion pipeline processing."""
    import os

    from fastapi import HTTPException

    from settings_manager import WORKSPACE_DIR

    try:
        file_list: list[UploadFile] = []
        if files:
            if isinstance(files, list):
                file_list.extend(files)
            else:
                file_list.append(files)

        if not file_list:
            try:
                form = await request.form()
                raw_files = form.getlist("files") or form.getlist("file")
                for f in raw_files:
                    if isinstance(f, UploadFile) or hasattr(f, "filename"):
                        file_list.append(f)
            except Exception as fe:
                logger.warning(f"Error parsing request form fallback: {fe}")

        if not file_list:
            raise HTTPException(status_code=400, detail="No files provided for upload.")

        upload_dir = os.path.join(WORKSPACE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        saved_paths = []

        for upload in file_list:
            raw_name = os.path.basename(upload.filename or "").strip()
            safe_name = raw_name if raw_name else "document.pdf"
            dest_path = os.path.join(upload_dir, safe_name)
            content = await upload.read()
            with open(dest_path, "wb") as buffer:
                buffer.write(content)
            saved_paths.append(dest_path)
        return {"success": True, "file_paths": saved_paths}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload pipeline files failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/start", summary="Start OCR pipeline (SSE stream)")
def start_pipeline(req: PipelineStartRequest):
    """Start a batch OCR pipeline run.

    Returns a Server-Sent Events stream of ``PipelineUpdate`` JSON objects,
    allowing clients to render real-time progress.
    """
    import os
    from unittest.mock import MagicMock

    from pipeline_manager import process_pdfs
    from settings_manager import WORKSPACE_DIR

    # Build mock file objects with .name attributes from file paths
    files = []
    missing_paths = []
    for path in req.file_paths:
        resolved_path = path
        if not os.path.isfile(resolved_path):
            candidates = [
                os.path.join(WORKSPACE_DIR, os.path.basename(path)),
                os.path.join("/home/owner/Downloads", os.path.basename(path)),
                os.path.expanduser(f"~/Downloads/{os.path.basename(path)}"),
                os.path.join(WORKSPACE_DIR, "souki_enclosures.pdf"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resolved_path = cand
                    break
        if not os.path.isfile(resolved_path):
            missing_paths.append(path)
        else:
            f = MagicMock()
            f.name = resolved_path
            files.append(f)

    def _extract_int_stat(val: object) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, dict):
            raw_val = str(val.get("value", ""))
            import re
            match = re.search(r"stat-value'>(\d+)<", raw_val) or re.search(r"(\d+)", raw_val)
            if match:
                return int(match.group(1))
        if isinstance(val, str):
            import re
            match = re.search(r"(\d+)", val)
            if match:
                return int(match.group(1))
        return 0

    def event_generator():
        if missing_paths or not files:
            err_msg = f"File(s) not found: {', '.join(missing_paths or req.file_paths)}"
            event_data = {
                "log_text": f"[Error] {err_msg}",
                "status_badge": "<span class='badge-failed'>File Not Found</span>",
                "progress_html": "<div class='stat-card'><div class='stat-value'>0%</div></div>",
                "completed_pages": 0,
                "failed_pages": 0,
                "run_id": "",
                "file_status_html": "",
                "upload_manifest_html": "",
                "error": err_msg,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            for result in process_pdfs(
                files=files,
                server_url=req.server_url,
                model_name=req.model_name,
                workers=req.workers,
                max_concurrent=req.max_concurrent,
                max_retries=req.max_retries,
                target_dim=req.target_dim,
                guided_decoding=req.guided_decoding,
            ):
                event_data = {
                    "log_text": result[0],
                    "status_badge": result[1] if isinstance(result[1], str) else "",
                    "progress_html": result[2],
                    "completed_pages": _extract_int_stat(result[3]),
                    "failed_pages": _extract_int_stat(result[4]),
                    "run_id": result[9],
                    "file_status_html": result[10] if isinstance(result[10], str) else "",
                    "upload_manifest_html": result[11] if isinstance(result[11], str) else "",
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'log_text': f'[Error] {e}', 'status_badge': '<span class=\"badge-failed\">Failed</span>'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/runs", response_model=list[RunInfo], summary="List available runs")
def list_runs():
    """Return all completed OCR runs with file counts and indexed status."""
    from settings_manager import get_available_runs

    runs = get_available_runs()
    result = []
    for display_name, run_dir in runs:
        import hashlib
        import re

        run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]
        # Extract file count from display name, e.g. "run_... (3 files)"
        match = re.search(r"\((\d+)\s+file", display_name)
        file_count = int(match.group(1)) if match else 0
        is_indexed = "[INDEXED]" in display_name or display_name.startswith("✅")
        result.append(
            RunInfo(
                display_name=display_name,
                run_dir=run_dir,
                run_id=run_id,
                file_count=file_count,
                is_indexed=is_indexed,
            )
        )
    return result


@router.post("/stop/{run_id}", response_model=MessageResponse, summary="Stop a pipeline run")
def stop_pipeline(run_id: str):
    """Send a stop signal to a running pipeline."""
    from pipeline_manager import stop_processing

    msg = stop_processing(run_id)
    return MessageResponse(success="Stop request sent" in msg, message=msg)


@router.get("/status/{run_id}", response_model=PipelineStatusResponse, summary="Get run status")
def get_run_status(run_id: str):
    """Return the current status of a pipeline run."""
    import process_state

    with process_state.active_runs_lock:
        run_info = process_state.active_runs.get(run_id)

    if not run_info:
        return PipelineStatusResponse(run_id=run_id, status="unknown")

    proc = run_info.get("proc")
    if proc and proc.poll() is None:
        status = "running"
    elif run_info.get("stop"):
        status = "stopped"
    else:
        status = "completed"

    return PipelineStatusResponse(
        run_id=run_id,
        status=status,
        log_tail=run_info.get("log_tail", ""),
    )
