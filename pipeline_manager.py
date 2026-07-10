import os
import re
import uuid
import time
import queue
import shutil
import threading
import datetime
import subprocess
import httpx
import gradio as gr
import state

from settings_manager import WORKSPACE_DIR
from html_utils import make_progress_bar_html, make_file_status_html, make_upload_manifest_html
from pdf_manager import make_zip
from pypdf import PdfReader

def enqueue_output(out, q):
    try:
        for line in iter(out.readline, ''):
            q.put(line)
    except Exception:
        pass
    finally:
        out.close()

def _make_empty_yield(log_text, badge_html, progress_html, start_interactive=True, run_id=""):
    return (
        log_text,
        gr.update(value=badge_html),
        progress_html,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        None,
        None,
        gr.update(interactive=start_interactive),
        run_id,
        "",  # file_status_table
        "",  # upload_manifest
    )

def process_pdfs(files, server_url, model_name, workers, max_concurrent, max_retries, target_dim, guided_decoding):
    if not files:
        yield _make_empty_yield("No files uploaded.", "<span class='badge-idle'>Idle</span>", "")
        return

    pdf_paths = []
    for f in files:
        if isinstance(f, str):
            pdf_paths.append(f)
        elif hasattr(f, "name"):
            pdf_paths.append(f.name)
        elif isinstance(f, dict) and "path" in f:
            pdf_paths.append(f["path"])

    if not pdf_paths:
        yield _make_empty_yield("Invalid file uploads.", "<span class='badge-idle'>Idle</span>", "")
        return

    try:
        preflight = httpx.get(server_url.rstrip("/") + "/models", timeout=3.0)
        if preflight.status_code != 200:
            yield _make_empty_yield(
                f"Pre-flight check failed: server at {server_url} returned HTTP {preflight.status_code}.\n"
                "Please ensure the inference server is running before starting a batch.",
                "<span class='badge-failed'>Server Unreachable</span>",
                ""
            )
            return
    except Exception as e:
        yield _make_empty_yield(
            f"Pre-flight check failed: cannot reach server at {server_url}.\n"
            f"Error: {e}\n\n"
            "Please ensure the inference server is running (check 🐳 Inference Status in the header).",
            "<span class='badge-failed'>Server Unreachable</span>",
            ""
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
            print(f"Error reading PDF page count for {orig_name}: {e}")
            file_page_counts[idx] = 1
            total_pages += 1

    # Dynamic getter for HTML helper
    manifest_fn = state.get_fn('make_upload_manifest_html', make_upload_manifest_html)
    manifest_html = manifest_fn(file_mapping, file_page_counts, file_sizes)

    # Dynamic workspace dir lookup
    workspace_dir = state.get_val('WORKSPACE_DIR', WORKSPACE_DIR)

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

    with state.active_runs_lock:
        state.active_runs[run_id] = {
            "stop": False,
            "proc": None,
            "run_dir": run_dir,
            "file_mapping": file_mapping
        }

    cmd = [
        "/home/owner/olmocr-env/bin/python", "-u",
        "-m", "olmocr.pipeline",
        ".",
        "--pdfs"
    ] + copied_relative_paths + [
        "--server", server_url,
        "--model", model_name,
        "--workers", str(int(workers)),
        "--max_concurrent_requests", str(int(max_concurrent)),
        "--target_longest_image_dim", str(int(target_dim)),
        "--max_page_retries", str(int(max_retries)),
        "--markdown"
    ]
    
    if guided_decoding:
        cmd.append("--guided_decoding")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        with state.active_runs_lock:
            if run_id in state.active_runs:
                state.active_runs[run_id]["proc"] = proc
    except Exception as e:
        yield (
            f"Failed to start pipeline process: {e}",
            gr.update(value="<span class='badge-failed'>Failed</span>"),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            None,
            None,
            gr.update(interactive=True),
            "",
            "",
            manifest_html,
        )
        return

    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q))
    t.daemon = True
    t.start()

    accumulated_logs = ""
    completed_pages = 0
    failed_pages = 0
    vllm_running = 0
    vllm_queued = 0
    completed_file_indices = set()
    failed_file_indices = set()
    start_time = time.monotonic()
    
    current_headers = None
    worker_states = {}

    pattern_completed = re.compile(r"completed_pages\s+([\d.]+)")
    pattern_failed = re.compile(r"failed_pages\s+([\d.]+)")
    pattern_vllm_queue = re.compile(r"vllm running req:\s*(\d+)\s+queue req:\s*(\d+)")
    pattern_vllm_standalone_queue = re.compile(r"Running:\s*(\d+).*?(?:Waiting|Pending):\s*(\d+)")

    status_table_fn = state.get_fn('make_file_status_html', make_file_status_html)
    file_status_html = status_table_fn(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)

    progress_bar_fn = state.get_fn('make_progress_bar_html', make_progress_bar_html)
    yield (
        "Initializing pipeline...",
        gr.update(value="<span class='badge-running'>Running</span>"),
        progress_bar_fn(0, total_pages),
        gr.update(visible=True, value=f"<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Completed Pages</div></div>"),
        gr.update(visible=True, value=f"<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Failed Pages</div></div>"),
        gr.update(choices=[], value=None),
        None,
        None,
        gr.update(interactive=False),
        run_id,
        file_status_html,
        manifest_html,
    )

    streaming_choices = []
    dropdown_value_set = False
    last_yield_time = time.monotonic()

    try:
        while t.is_alive() or not q.empty():
            with state.active_runs_lock:
                if run_id in state.active_runs and state.active_runs[run_id]["stop"]:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    elapsed = time.monotonic() - start_time
                    if not dropdown_value_set and streaming_choices:
                        dropdown_val_update = gr.update(choices=streaming_choices, value=streaming_choices[0][1])
                        dropdown_value_set = True
                    else:
                        dropdown_val_update = gr.update(choices=streaming_choices)
                    yield (
                        accumulated_logs + "\n\n[PROCESS TERMINATED BY USER]\n",
                        gr.update(value="<span class='badge-stopped'>Stopped</span>"),
                        progress_bar_fn(completed_pages, total_pages, elapsed),
                        gr.update(value=f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"),
                        gr.update(value=f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"),
                        dropdown_val_update,
                        None,
                        None,
                        gr.update(interactive=True),
                        "",
                        status_table_fn(file_mapping, file_page_counts, completed_file_indices, failed_file_indices),
                        manifest_html,
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
                        try:
                            val = int(parts[i])
                            worker_states[worker_id][state_name] = val
                        except (ValueError, TypeError):
                            pass
                
                total_completed = sum(states.get("finished", 0) for states in worker_states.values())
                total_failed = sum(states.get("errored", 0) for states in worker_states.values())
                completed_pages = max(completed_pages, total_completed)
                failed_pages = max(failed_pages, total_failed)

            if "Completed pages:" in line:
                match_c = re.search(r"Completed pages:\s*([\d,]+)", line)
                if match_c:
                    try:
                        val = int(match_c.group(1).replace(",", ""))
                        completed_pages = max(completed_pages, val)
                    except (ValueError, TypeError):
                        pass

            if "Failed pages:" in line:
                match_f = re.search(r"Failed pages:\s*([\d,]+)", line)
                if match_f:
                    try:
                        val = int(match_f.group(1).replace(",", ""))
                        failed_pages = max(failed_pages, val)
                    except (ValueError, TypeError):
                        pass
            
            md_inputs_dir = os.path.join(run_dir, "markdown", "inputs")
            if os.path.exists(md_inputs_dir):
                completed_mds = [f for f in os.listdir(md_inputs_dir) if f.endswith(".md")]
                temp_completed_pages = 0
                for md_file in completed_mds:
                    match = re.match(r"^(\d+)_", md_file)
                    if match:
                        file_idx = int(match.group(1))
                        if file_idx not in completed_file_indices:
                            completed_file_indices.add(file_idx)
                            orig_name = file_mapping.get(file_idx, md_file)
                            choice_tuple = (orig_name, md_file)
                            if choice_tuple not in streaming_choices:
                                streaming_choices.append(choice_tuple)
                        temp_completed_pages += file_page_counts.get(file_idx, 1)
                completed_pages = max(completed_pages, temp_completed_pages)

            vllm_match = pattern_vllm_queue.search(line)
            if vllm_match:
                vllm_running = int(vllm_match.group(1))
                vllm_queued = int(vllm_match.group(2))
            else:
                vllm_match_standalone = pattern_vllm_standalone_queue.search(line)
                if vllm_match_standalone:
                    vllm_running = int(vllm_match_standalone.group(1))
                    vllm_queued = int(vllm_match_standalone.group(2))

            now = time.monotonic()
            if now - last_yield_time >= 0.2:
                elapsed = now - start_time
                status_html = f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
                failed_html = f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
                file_status_html = status_table_fn(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)
                
                if not dropdown_value_set and streaming_choices:
                    dropdown_val_update = gr.update(choices=streaming_choices, value=streaming_choices[0][1])
                    dropdown_value_set = True
                else:
                    dropdown_val_update = gr.update(choices=streaming_choices)
                
                yield (
                    accumulated_logs,
                    gr.update(value="<span class='badge-running'>Running</span>"),
                    progress_bar_fn(completed_pages, total_pages, elapsed),
                    gr.update(value=status_html),
                    gr.update(value=failed_html),
                    dropdown_val_update,
                    None,
                    None,
                    gr.update(interactive=False),
                    run_id,
                    file_status_html,
                    manifest_html,
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
                state.get_fn('make_zip', make_zip)(md_inputs_dir, zip_file_path)
            except Exception as e:
                print(f"Error creating ZIP archive: {e}")

        final_completed = sum(file_page_counts.get(idx, 1) for idx in completed_file_indices)
        completed_pages = max(completed_pages, final_completed)

        elapsed = time.monotonic() - start_time

        if exit_code == 0:
            status_text = "<span class='badge-success'>Success</span>"
        else:
            status_text = "<span class='badge-failed'>Failed</span>"

        file_status_html = status_table_fn(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)

        if not dropdown_value_set and choices:
            dropdown_val_update = gr.update(choices=choices, value=dropdown_value)
            dropdown_value_set = True
        else:
            dropdown_val_update = gr.update(choices=choices)

        yield (
            accumulated_logs + f"\n\n[PROCESS EXITED WITH CODE {exit_code}]\n",
            gr.update(value=status_text),
            progress_bar_fn(completed_pages, total_pages, elapsed),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"),
            dropdown_val_update,
            zip_file_path,
            None,
            gr.update(interactive=True),
            run_id,
            file_status_html,
            manifest_html,
        )

    except Exception as e:
        elapsed = time.monotonic() - start_time
        yield (
            accumulated_logs + f"\n\nException during processing: {e}\n",
            gr.update(value="<span class='badge-failed'>Error</span>"),
            progress_bar_fn(completed_pages, total_pages, elapsed),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"),
            gr.update(choices=[], value=None),
            None,
            None,
            gr.update(interactive=True),
            "",
            "",
            manifest_html,
        )
    finally:
        with state.active_runs_lock:
            if run_id in state.active_runs:
                state.active_runs[run_id]["completed"] = True

def stop_processing(run_id):
    if not run_id:
        return "<span class='badge-idle'>No active process to stop.</span>"
    with state.active_runs_lock:
        if run_id in state.active_runs:
            state.active_runs[run_id]["stop"] = True
            proc = state.active_runs[run_id]["proc"]
            if proc:
                proc.terminate()
            return f"<span class='badge-stopped'>Stop request sent for run {run_id[:8]}.</span>"
    return "<span class='badge-idle'>Process not found or already ended.</span>"

def cleanup_active_runs():
    if os.environ.get("TESTING") == "true":
        return
    with state.active_runs_lock:
        for run_id, run_info in state.active_runs.items():
            proc = run_info.get("proc")
            if proc and proc.poll() is None:
                print(f"Terminating running pipeline process for run {run_id[:8]}...")
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass
