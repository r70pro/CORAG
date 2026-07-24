import datetime
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Generator
from typing import Any

import httpx
from pypdf import PdfReader

import process_state
from html_utils import make_file_status_html, make_progress_bar_html, make_upload_manifest_html
from pdf_manager import make_zip
from settings_manager import WORKSPACE_DIR

logger = logging.getLogger(__name__)


def enqueue_output(out: Any, q: queue.Queue) -> None:
    try:
        for line in iter(out.readline, ""):
            q.put(line)
    except Exception:
        pass
    finally:
        out.close()


class PipelineResult(tuple):
    """Plain-data result tuple yielded by process_pdfs().

    Holds raw Python values — no Gradio dependency.  The companion
    ``ui_adapters.pipeline_result_to_gradio()`` function converts these
    into ``gr.update()`` tuples for the Gradio UI layer.
    """

    __slots__ = ()

    def __new__(
        cls,
        log_text: str,
        status_badge: Any,
        progress_bar: str,
        completed_pages: Any,
        failed_pages: Any,
        file_selector: Any,
        download_zip: Any,
        download_individual: Any,
        start_btn: Any,
        active_run_id: str,
        file_status_table: str = "",
        upload_manifest_display: str = "",
        stop_btn: Any = None,
    ) -> "PipelineResult":
        return tuple.__new__(
            cls,
            (
                log_text,
                status_badge,
                progress_bar,
                completed_pages,
                failed_pages,
                file_selector,
                download_zip,
                download_individual,
                start_btn,
                active_run_id,
                file_status_table,
                upload_manifest_display,
                stop_btn,
            ),
        )

    @property
    def log_text(self) -> str:
        return self[0]

    @property
    def status_badge(self) -> Any:
        return self[1]

    @property
    def progress_bar(self) -> str:
        return self[2]

    @property
    def completed_pages(self) -> Any:
        return self[3]

    @property
    def failed_pages(self) -> Any:
        return self[4]

    @property
    def file_selector(self) -> Any:
        return self[5]

    @property
    def download_zip(self) -> Any:
        return self[6]

    @property
    def download_individual(self) -> Any:
        return self[7]

    @property
    def start_btn(self) -> Any:
        return self[8]

    @property
    def active_run_id(self) -> str:
        return self[9]

    @property
    def file_status_table(self) -> str:
        return self[10]

    @property
    def upload_manifest_display(self) -> str:
        return self[11]

    @property
    def stop_btn(self) -> Any:
        return self[12]


def _make_empty_yield(
    log_text: str,
    badge_html: str,
    progress_html: str,
    start_interactive: bool = True,
    run_id: str = "",
) -> PipelineResult:
    return PipelineResult(
        log_text,
        badge_html,
        progress_html,
        {"visible": False},
        {"visible": False},
        {"choices": [], "value": None},
        None,
        None,
        {"interactive": start_interactive},
        run_id,
        "",
        "",
        {"interactive": False},
    )


def process_pdfs(
    files: list[Any] | None,
    server_url: str,
    model_name: str,
    workers: int,
    max_concurrent: int,
    max_retries: int,
    target_dim: int,
    guided_decoding: bool,
) -> Generator[PipelineResult, None, None]:
    if not files:
        yield _make_empty_yield("No files uploaded.", "<span class='badge-idle'>Idle</span>", "")
        return

    pdf_paths = []
    for f in files:
        raw_p = None
        if isinstance(f, str):
            raw_p = f
        elif hasattr(f, "name"):
            raw_p = f.name
        elif isinstance(f, dict) and "path" in f:
            raw_p = f["path"]

        if raw_p:
            if os.path.isfile(raw_p):
                pdf_paths.append(raw_p)
            else:
                candidates = [
                    os.path.join(WORKSPACE_DIR, os.path.basename(raw_p)),
                    os.path.join("/home/owner/Downloads", os.path.basename(raw_p)),
                    os.path.expanduser(f"~/Downloads/{os.path.basename(raw_p)}"),
                    os.path.join(WORKSPACE_DIR, "souki_enclosures.pdf"),
                ]
                found = False
                for cand in candidates:
                    if os.path.isfile(cand):
                        pdf_paths.append(cand)
                        found = True
                        break
                if not found:
                    pdf_paths.append(raw_p)

    if not pdf_paths:
        yield _make_empty_yield("Invalid file uploads.", "<span class='badge-idle'>Idle</span>", "")
        return

    try:
        try:
            preflight = httpx.get(server_url.rstrip("/") + "/models", timeout=5.0)
        except Exception:
            if "localhost" in server_url:
                alt_url = server_url.replace("localhost", "127.0.0.1")
                preflight = httpx.get(alt_url.rstrip("/") + "/models", timeout=5.0)
                server_url = alt_url
            else:
                raise

        if preflight.status_code != 200:
            yield _make_empty_yield(
                f"Pre-flight check failed: server at {server_url} returned HTTP {preflight.status_code}.\n"
                "Please ensure the inference server is running before starting a batch.",
                "<span class='badge-failed'>Server Unreachable</span>",
                "",
            )
            return

        # Check if the requested model is actually loaded on the server
        try:
            res_json = preflight.json()
            if isinstance(res_json, dict) and "data" in res_json:
                models_data = res_json["data"]
                if isinstance(models_data, list):
                    loaded_models = [
                        m.get("id") for m in models_data if isinstance(m, dict) and m.get("id")
                    ]
                    # Flexible check: allow generic 'model', or exact match, or case-insensitive/prefix match
                    is_matched = (
                        not loaded_models
                        or model_name == "model"
                        or model_name in loaded_models
                        or any(model_name.lower() in lm.lower() or lm.lower() in model_name.lower() for lm in loaded_models)
                    )
                    if not is_matched:
                        yield _make_empty_yield(
                            f"Pre-flight check failed: The requested model '{model_name}' is not loaded on the server at {server_url}.\n"
                            f"Currently loaded model(s): {', '.join(loaded_models)}.\n\n"
                            "Please switch the model in the Settings or recreate the Docker container with the correct model.",
                            "<span class='badge-failed'>Model Mismatch</span>",
                            "",
                        )
                        return
        except Exception as e:
            # If JSON parsing or field access fails, log it but don't block the run
            logger.error(f"Error checking loaded models from server: {e}")
    except Exception as e:
        yield _make_empty_yield(
            f"Pre-flight check failed: cannot reach server at {server_url}.\n"
            f"Error: {e}\n\n"
            "Please ensure the inference server is running (check 🐳 Inference Status in the header).",
            "<span class='badge-failed'>Server Unreachable</span>",
            "",
        )
        return

    total_pages = 0
    file_mapping = {}
    file_page_counts = {}
    file_sizes = {}
    for idx, path in enumerate(pdf_paths):
        orig_name = os.path.basename(path)
        file_mapping[idx] = orig_name
        try:
            file_sizes[idx] = os.path.getsize(path)
        except Exception:
            file_sizes[idx] = 0
        try:
            reader = PdfReader(path)
            pc = len(reader.pages)
            file_page_counts[idx] = pc
            total_pages += pc
        except Exception as e:
            logger.error(f"Error reading PDF page count for {orig_name}: {e}")
            file_page_counts[idx] = 1
            total_pages += 1

    manifest_html = make_upload_manifest_html(file_mapping, file_page_counts, file_sizes)

    # Dynamic workspace dir lookup
    workspace_dir = WORKSPACE_DIR

    run_id = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(workspace_dir, f"run_{run_timestamp}_{run_id[:8]}")
    inputs_dir = os.path.join(run_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    copied_relative_paths = []
    for idx, path in enumerate(pdf_paths):
        orig_name = os.path.basename(path)
        safe_name = f"{idx}_{orig_name}"
        dest = os.path.join(inputs_dir, safe_name)
        shutil.copy(path, dest)
        copied_relative_paths.append(os.path.join("inputs", safe_name))

    with process_state.active_runs_lock:
        process_state.active_runs[run_id] = {
            "stop": False,
            "proc": None,
            "run_dir": run_dir,
            "file_mapping": file_mapping,
        }

    python_exe = sys.executable or "python"
    cmd = (
        [python_exe, "-u", "-m", "olmocr.pipeline", ".", "--pdfs"]
        + copied_relative_paths
        + [
            "--server",
            server_url,
            "--model",
            model_name,
            "--workers",
            str(int(workers)),
            "--max_concurrent_requests",
            str(int(max_concurrent)),
            "--target_longest_image_dim",
            str(int(target_dim)),
            "--max_page_retries",
            str(int(max_retries)),
            "--markdown",
        ]
    )

    if guided_decoding:
        cmd.append("--guided_decoding")

    try:
        proc = subprocess.Popen(
            cmd, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        with process_state.active_runs_lock:
            if run_id in process_state.active_runs:
                process_state.active_runs[run_id]["proc"] = proc
    except Exception as e:
        yield PipelineResult(
            f"Failed to start pipeline process: {e}",
            "<span class='badge-failed'>Failed</span>",
            "",
            {"visible": False},
            {"visible": False},
            {"choices": [], "value": None},
            None,
            None,
            {"interactive": True},
            "",
            "",
            manifest_html,
            {"interactive": False},
        )
        return

    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q))
    t.daemon = True
    t.start()

    accumulated_logs = ""
    completed_pages = 0
    failed_pages = 0
    completed_file_indices = set()
    failed_file_indices = set()
    start_time = time.monotonic()
    hundred_percent_start_time = None

    current_headers = None
    worker_states = {}

    pattern_completed = re.compile(r"completed_pages\s+([\d.]+)")
    pattern_failed = re.compile(r"failed_pages\s+([\d.]+)")

    file_status_html = make_file_status_html(
        file_mapping, file_page_counts, completed_file_indices, failed_file_indices
    )

    progress_bar_fn = make_progress_bar_html
    yield PipelineResult(
        "Initializing pipeline...",
        "<span class='badge-running'>Running</span>",
        progress_bar_fn(0, total_pages),
        {
            "visible": True,
            "value": "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Completed Pages</div></div>",
        },
        {
            "visible": True,
            "value": "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Failed Pages</div></div>",
        },
        {"choices": [], "value": None},
        None,
        None,
        {"interactive": False},
        run_id,
        file_status_html,
        manifest_html,
        {"interactive": True},
    )

    streaming_choices = []
    dropdown_value_set = False
    last_yield_time = time.monotonic()

    try:
        while t.is_alive() or not q.empty():
            if total_pages > 0 and completed_pages >= total_pages:
                if hundred_percent_start_time is None:
                    hundred_percent_start_time = time.monotonic()
                elif time.monotonic() - hundred_percent_start_time > 10.0:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            else:
                hundred_percent_start_time = None

            with process_state.active_runs_lock:
                if (
                    run_id in process_state.active_runs
                    and process_state.active_runs[run_id]["stop"]
                ):
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    elapsed = time.monotonic() - start_time
                    if not dropdown_value_set and streaming_choices:
                        dropdown_val_update = {
                            "choices": streaming_choices,
                            "value": streaming_choices[0][1],
                        }
                        dropdown_value_set = True
                    else:
                        dropdown_val_update = {"choices": streaming_choices}
                    yield PipelineResult(
                        accumulated_logs + "\n\n[PROCESS TERMINATED BY USER]\n",
                        "<span class='badge-stopped'>Stopped</span>",
                        progress_bar_fn(completed_pages, total_pages, elapsed),
                        {
                            "value": f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
                        },
                        {
                            "value": f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
                        },
                        dropdown_val_update,
                        None,
                        None,
                        {"interactive": True},
                        "",
                        make_file_status_html(
                            file_mapping,
                            file_page_counts,
                            completed_file_indices,
                            failed_file_indices,
                        ),
                        manifest_html,
                        {"interactive": False},
                    )
                    return

            try:
                line = q.get_nowait()
            except queue.Empty:
                threading.Event().wait(0.05)
                continue

            accumulated_logs += line

            if "failed_pages" in line:
                match_f = pattern_failed.search(line)
                if match_f:
                    try:
                        failed_pages = max(failed_pages, int(float(match_f.group(1))))
                    except (ValueError, TypeError):
                        pass

            parts = [p.strip() for p in line.split("|")]
            if "Worker ID" in line:
                current_headers = [h.strip() for h in line.split("|")]
            elif len(parts) >= 2 and all(p.isdigit() for p in parts):
                if not current_headers or len(parts) != len(current_headers):
                    if len(parts) == 3:
                        current_headers = ["Worker ID", "finished", "started"]
                    elif len(parts) == 4:
                        current_headers = ["Worker ID", "errored", "finished", "started"]

                worker_id = int(parts[0])
                if worker_id not in worker_states:
                    worker_states[worker_id] = {}
                for i in range(1, len(parts)):
                    if i < len(current_headers):
                        state_name = current_headers[i]
                        val = int(parts[i])
                        worker_states[worker_id][state_name] = val

                total_completed = sum(
                    states.get("finished", 0) for states in worker_states.values()
                )
                total_failed = sum(states.get("errored", 0) for states in worker_states.values())
                completed_pages = max(completed_pages, total_completed)
                failed_pages = max(failed_pages, total_failed)

            if "Completed pages:" in line or "completed_pages" in line:
                match_c = re.search(r"Completed pages:\s*([\d,]+)", line)
                if match_c:
                    completed_pages = max(completed_pages, int(match_c.group(1).replace(",", "")))
                else:
                    match_c = pattern_completed.search(line)
                    if match_c:
                        try:
                            completed_pages = max(completed_pages, int(float(match_c.group(1))))
                        except (ValueError, TypeError):
                            pass

            if "Failed pages:" in line:
                match_f = re.search(r"Failed pages:\s*([\d,]+)", line)
                if match_f:
                    failed_pages = max(failed_pages, int(match_f.group(1).replace(",", "")))

            md_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
            if os.path.exists(md_inputs_dir):
                completed_mds = [f for f in os.listdir(md_inputs_dir) if f.endswith(".md")]
                temp_completed_pages = 0
                for md_file in completed_mds:
                    match = re.match(r"^(\d+)_", md_file)
                    if match:
                        file_idx = int(match.group(1))
                        completed_file_indices.add(file_idx)
                        orig_name = file_mapping.get(file_idx, md_file)
                        choice_tuple = (orig_name, md_file)
                        streaming_choices.append(choice_tuple)
                    temp_completed_pages += file_page_counts.get(file_idx, 1)
                completed_pages = max(completed_pages, temp_completed_pages)

            now = time.monotonic()
            if now - last_yield_time >= 0.2:
                elapsed = now - start_time
                status_html = f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
                failed_html = f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
                file_status_html = make_file_status_html(
                    file_mapping, file_page_counts, completed_file_indices, failed_file_indices
                )

                if not dropdown_value_set and streaming_choices:
                    dropdown_val_update = {
                        "choices": streaming_choices,
                        "value": streaming_choices[0][1],
                    }
                    dropdown_value_set = True
                else:
                    dropdown_val_update = {"choices": streaming_choices}

                yield PipelineResult(
                    accumulated_logs,
                    "<span class='badge-running'>Running</span>",
                    progress_bar_fn(completed_pages, total_pages, elapsed),
                    {"value": status_html},
                    {"value": failed_html},
                    dropdown_val_update,
                    None,
                    None,
                    {"interactive": False},
                    run_id,
                    file_status_html,
                    manifest_html,
                    {"interactive": True},
                )
                last_yield_time = now

        proc.wait()
        exit_code = proc.returncode

        md_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
        choices = []
        dropdown_value = None
        zip_file_path = None

        if os.path.exists(md_inputs_dir):
            completed_mds = sorted([f for f in os.listdir(md_inputs_dir) if f.endswith(".md")])
            for md_file in completed_mds:
                match = re.match(r"^(\d+)_", md_file)
                if match:
                    idx = int(match.group(1))
                    orig_name = file_mapping.get(idx, md_file)
                    choices.append((orig_name, md_file))
                    completed_file_indices.add(idx)

            if choices:
                dropdown_value = choices[0][1]

            zip_file_path = os.path.join(run_dir, "all_markdown_results.zip")
            try:
                make_zip(md_inputs_dir, zip_file_path)
            except Exception as e:
                logger.error(f"Error creating ZIP archive: {e}")

        final_completed = sum(file_page_counts.get(idx, 1) for idx in completed_file_indices)
        completed_pages = max(completed_pages, final_completed)

        elapsed = time.monotonic() - start_time

        if exit_code == 0 or (total_pages > 0 and completed_pages >= total_pages):
            status_text = "<span class='badge-success'>Success</span>"
        else:
            status_text = "<span class='badge-failed'>Failed</span>"

        file_status_html = make_file_status_html(
            file_mapping, file_page_counts, completed_file_indices, failed_file_indices
        )

        if not dropdown_value_set and choices:
            dropdown_val_update = {"choices": choices, "value": dropdown_value}
            dropdown_value_set = True
        else:
            dropdown_val_update = {"choices": choices}

        yield PipelineResult(
            accumulated_logs + f"\n\n[PROCESS EXITED WITH CODE {exit_code}]\n",
            status_text,
            progress_bar_fn(completed_pages, total_pages, elapsed),
            {
                "value": f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
            },
            {
                "value": f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
            },
            dropdown_val_update,
            zip_file_path,
            None,
            {"interactive": True},
            run_id,
            file_status_html,
            manifest_html,
            {"interactive": False},
        )

    except Exception as e:
        elapsed = time.monotonic() - start_time
        yield PipelineResult(
            accumulated_logs + f"\n\nException during processing: {e}\n",
            "<span class='badge-failed'>Error</span>",
            progress_bar_fn(completed_pages, total_pages, elapsed),
            {
                "value": f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
            },
            {
                "value": f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
            },
            {"choices": [], "value": None},
            None,
            None,
            {"interactive": True},
            "",
            "",
            manifest_html,
            {"interactive": False},
        )
    finally:
        with process_state.active_runs_lock:
            if run_id in process_state.active_runs:
                process_state.active_runs[run_id]["completed"] = True


def stop_processing(run_id: str) -> str:
    if not run_id:
        return "<span class='badge-idle'>No active process to stop.</span>"
    with process_state.active_runs_lock:
        if run_id in process_state.active_runs:
            process_state.active_runs[run_id]["stop"] = True
            proc = process_state.active_runs[run_id]["proc"]
            if proc:
                proc.terminate()
            return f"<span class='badge-stopped'>Stop request sent for run {run_id[:8]}.</span>"
    return "<span class='badge-idle'>Process not found or already ended.</span>"


def cleanup_active_runs() -> None:
    if os.environ.get("TESTING") == "true":
        return
    if os.environ.get("KEEP_CONTAINERS_ON_EXIT") == "true":
        return
    with process_state.active_runs_lock:
        for run_id, run_info in process_state.active_runs.items():
            proc = run_info.get("proc")
            if proc and proc.poll() is None:
                logger.info(f"Terminating running pipeline process for run {run_id[:8]}...")
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass
