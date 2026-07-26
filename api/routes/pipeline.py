"""
OCR pipeline management API routes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from api.auth import verify_admin_key
from api.errors import error_envelope
from api.models import MessageResponse, PipelineStartRequest, PipelineStatusResponse, RunInfo
from api.upload_security import (
    UPLOAD_CHUNK_BYTES,
    LimitedUploadRoute,
    close_uploads,
    escaped_original_name,
    pdf_upload_limits,
    require_content_type,
    unique_upload_name,
    validate_pdf_file,
)
from path_security import PathSecurityError, resolve_file_under, resolve_under, validate_filename

logger = logging.getLogger(__name__)
router = APIRouter(route_class=LimitedUploadRoute)


@router.post("/upload", summary="Upload source PDF documents for ingestion")
async def upload_pipeline_files(
    request: Request,
    files: list[UploadFile] | UploadFile | None = File(default=None),
):
    """Upload one or more PDF files for ingestion pipeline processing."""
    from settings_manager import WORKSPACE_DIR

    file_list: list[UploadFile] = []
    created_paths: list[Path] = []
    try:
        if files:
            file_list.extend(files if isinstance(files, list) else [files])

        if not file_list and request is not None:
            try:
                form = await request.form()
                raw_files = form.getlist("files") or form.getlist("file")
                file_list.extend(
                    f for f in raw_files if isinstance(f, UploadFile) or hasattr(f, "filename")
                )
            except Exception as exc:
                logger.warning("Error parsing request form fallback: %s", exc)

        if not file_list:
            raise HTTPException(status_code=400, detail="No files provided for upload.")

        limits = pdf_upload_limits()
        if len(file_list) > limits.max_files:
            raise HTTPException(status_code=413, detail="Too many uploaded files")

        upload_dir = resolve_under(WORKSPACE_DIR, "uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        metadata: list[dict[str, str]] = []
        aggregate_bytes = 0

        for upload in file_list:
            original_name = escaped_original_name(upload, ".pdf")
            require_content_type(upload, {"application/pdf", "application/octet-stream"})
            stored_name = unique_upload_name(".pdf")
            dest_path = resolve_file_under(upload_dir, stored_name, {".pdf"})
            created_paths.append(dest_path)

            file_bytes = 0
            prefix = bytearray()
            with dest_path.open("xb") as buffer:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    aggregate_bytes += len(chunk)
                    if file_bytes > limits.max_file_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded PDF is too large")
                    if aggregate_bytes > limits.max_total_bytes:
                        raise HTTPException(status_code=413, detail="Aggregate upload is too large")
                    if len(prefix) < 5:
                        prefix.extend(chunk[: 5 - len(prefix)])
                    buffer.write(chunk)

            if bytes(prefix) != b"%PDF-":
                raise HTTPException(status_code=415, detail="Upload is not a PDF")
            await run_in_threadpool(validate_pdf_file, dest_path)
            metadata_path = resolve_file_under(
                upload_dir, f"{stored_name}.metadata.json", {".json"}
            )
            metadata_path.write_text(
                json.dumps({"original_name": original_name}), encoding="utf-8"
            )
            created_paths.append(metadata_path)
            saved_paths.append(stored_name)
            metadata.append({"file_path": stored_name, "original_name": original_name})

        return {"success": True, "file_paths": saved_paths, "files": metadata}
    except HTTPException:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in created_paths:
            path.unlink(missing_ok=True)
        logger.error("Upload pipeline files failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed") from exc
    except BaseException:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        await close_uploads(file_list)


@router.post("/start", summary="Start OCR pipeline (SSE stream)")
def start_pipeline(req: PipelineStartRequest):
    """Start a batch OCR pipeline run.

    Returns a Server-Sent Events stream of ``PipelineUpdate`` JSON objects,
    allowing clients to render real-time progress.
    """
    from html import unescape

    from pipeline_manager import process_pdfs
    from settings_manager import WORKSPACE_DIR

    try:
        upload_dir = resolve_under(WORKSPACE_DIR, "uploads")
    except PathSecurityError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc

    # API clients may select only filenames previously stored by the upload route.
    files = []
    for filename in req.file_paths:
        try:
            validate_filename(filename, {".pdf"})
            resolved_path = resolve_file_under(upload_dir, filename, {".pdf"})
        except PathSecurityError as exc:
            raise HTTPException(status_code=400, detail="Invalid input file") from exc
        if not resolved_path.is_file():
            raise HTTPException(status_code=400, detail="Input file not found")
        original_filename = filename
        try:
            metadata_path = resolve_file_under(
                upload_dir, f"{filename}.metadata.json", {".json"}
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_name = unescape(str(metadata.get("original_name", "")))
            validate_filename(candidate_name, {".pdf"})
            original_filename = candidate_name
        except (OSError, ValueError, TypeError, json.JSONDecodeError, PathSecurityError):
            logger.warning("Upload provenance metadata is unavailable for %s", filename)
        file_ref = SimpleNamespace(
            name=str(resolved_path), original_filename=original_filename
        )
        files.append(file_ref)

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
        if not files:
            err_msg = "No valid input files"
            envelope = error_envelope("invalid_input_files", err_msg).model_dump(
                mode="json", exclude_none=True
            )
            event_data = {
                "log_text": f"[Error] {err_msg}",
                "status_badge": "<span class='badge-failed'>File Not Found</span>",
                "progress_html": "<div class='stat-card'><div class='stat-value'>0%</div></div>",
                "completed_pages": 0,
                "failed_pages": 0,
                "run_id": "",
                "file_status_html": "",
                "upload_manifest_html": "",
                **envelope,
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
        except Exception:
            logger.exception("Pipeline processing failed")
            envelope = error_envelope(
                "pipeline_processing_failed", "Pipeline processing failed"
            ).model_dump(mode="json", exclude_none=True)
            event_data = {
                **envelope,
                "log_text": "[Error] Pipeline processing failed",
                "status_badge": '<span class="badge-failed">Failed</span>',
            }
            yield f"data: {json.dumps(event_data)}\n\n"

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

        run_name = Path(run_dir).name
        run_id = hashlib.sha256(run_dir.encode()).hexdigest()[:16]
        # Extract file count from display name, e.g. "run_... (3 files)"
        match = re.search(r"\((\d+)\s+file", display_name)
        file_count = int(match.group(1)) if match else 0
        is_indexed = "[INDEXED]" in display_name or display_name.startswith("✅")
        result.append(
            RunInfo(
                display_name=display_name,
                run_dir=run_name,
                run_id=run_id,
                file_count=file_count,
                is_indexed=is_indexed,
            )
        )
    return result


@router.post(
    "/stop/{run_id}",
    response_model=MessageResponse,
    summary="Stop a pipeline run",
    dependencies=[Depends(verify_admin_key)],
)
def stop_pipeline(run_id: str):
    """Send a stop signal to a running pipeline."""
    from pipeline_manager import stop_processing

    msg = stop_processing(run_id)
    if "Stop request sent" not in msg:
        raise HTTPException(status_code=409, detail=msg or "Pipeline run cannot be stopped")
    return MessageResponse(success=True, message=msg)


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
