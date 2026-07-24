"""
OCR pipeline management API routes.
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import MessageResponse, PipelineStartRequest, PipelineStatusResponse, RunInfo

router = APIRouter()


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
            return MessageResponse(success=False, message=f"File not found: {path}")
        f = MagicMock()
        f.name = resolved_path
        files.append(f)

    def event_generator():
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
                    "completed_pages": result[3] if isinstance(result[3], int | str) else 0,
                    "failed_pages": result[4] if isinstance(result[4], int | str) else 0,
                    "run_id": result[9],
                    "file_status_html": result[10] if isinstance(result[10], str) else "",
                    "upload_manifest_html": result[11] if isinstance(result[11], str) else "",
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/runs", response_model=list[RunInfo], summary="List available runs")
def list_runs():
    """Return all completed OCR runs with file counts."""
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
        result.append(
            RunInfo(
                display_name=display_name,
                run_dir=run_dir,
                run_id=run_id,
                file_count=file_count,
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
