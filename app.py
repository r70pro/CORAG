import os
import re
import json
import uuid
import time
import queue
import shutil
import zipfile
import threading
import subprocess
import datetime
import atexit
import httpx
import gradio as gr
from pypdf import PdfReader

# Constants
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

# Thread-safe storage for active runs
active_runs = {}
active_runs_lock = threading.Lock()

def load_settings():
    defaults = {
        "server_url": "http://localhost:8000/v1",
        "model_name": "allenai/olmOCR-2-7B-1025-FP8",
        "workers": 4,
        "max_concurrent_requests": 20,
        "target_longest_image_dim": 1288,
        "max_page_retries": 8,
        "guided_decoding": True,
        "docker_port": 8000,
        "docker_gpu_mem": 0.80,
        "docker_max_model_len": 15360,
        "hf_token": os.environ.get("HF_TOKEN", "")
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                user_settings = json.load(f)
                defaults.update(user_settings)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return defaults

def save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {e}"

def enqueue_output(out, q):
    try:
        for line in iter(out.readline, ''):
            q.put(line)
    except Exception:
        pass
    finally:
        out.close()

def get_docker_status():
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", "olmocr"],
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            return res.stdout.strip().lower()
        elif "no such object" in res.stderr.lower() or "no such inspect object" in res.stderr.lower():
            return "not_found"
        else:
            return "error"
    except Exception as e:
        return "error"

def check_server_ready(port):
    try:
        response = httpx.get(f"http://localhost:{port}/v1/models", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False

def get_docker_status_str(port):
    status = get_docker_status()
    if status == "not_found":
        return "not_found", "<span class='badge-idle'>Docker: Not Created</span>"
    elif status == "exited":
        return "stopped", "<span class='badge-stopped'>Docker: Stopped</span>"
    elif status == "running":
        if check_server_ready(port):
            return "ready", "<span class='badge-success'>Inference Server: Ready</span>"
        else:
            return "starting", "<span class='badge-running'>Server: Starting / Loading Model</span>"
    else:
        return "error", "<span class='badge-failed'>Docker: Error</span>"

def start_docker_container():
    status = get_docker_status()
    if status == "exited":
        try:
            subprocess.run(["docker", "start", "olmocr"], check=True, capture_output=True)
            return True, "Container started successfully."
        except subprocess.CalledProcessError as e:
            return False, f"Failed to start container: {e.stderr.decode().strip()}"
    elif status == "running":
        return True, "Container is already running."
    elif status == "not_found":
        return False, "Container 'olmocr' not found. Please recreate/provision it first."
    return False, f"Container status is {status}, cannot start."

def stop_docker_container():
    status = get_docker_status()
    if status == "running":
        try:
            subprocess.run(["docker", "stop", "olmocr"], check=True, capture_output=True)
            return True, "Container stopped successfully."
        except subprocess.CalledProcessError as e:
            return False, f"Failed to stop container: {e.stderr.decode().strip()}"
    return True, "Container is not running."

def create_docker_container(hf_token, port, model, gpu_mem, max_model_len):
    status = get_docker_status()
    if status in ["running", "exited"]:
        try:
            if status == "running":
                subprocess.run(["docker", "stop", "olmocr"], check=True, capture_output=True)
            subprocess.run(["docker", "rm", "olmocr"], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return False, f"Failed to remove existing container: {e.stderr.decode().strip()}"

    hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    cmd = [
        "docker", "run", "-d",
        "--name", "olmocr",
        "--restart", "unless-stopped",
        "--gpus", "all",
        "--ipc=host",
        "-p", f"{int(port)}:8000",
        "-v", f"{hf_cache_dir}:/root/.cache/huggingface",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "vllm/vllm-openai:v0.24.0",
        "--model", model,
        "--gpu_memory_utilization", f"{float(gpu_mem):.2f}",
        "--max_model_len", str(int(max_model_len)),
        "--enforce-eager"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, "Container created and started successfully."
    except subprocess.CalledProcessError as e:
        return False, f"Failed to create container: {e.stderr.strip()}"

def cleanup_docker():
    print("Application shutting down. Stopping local OLMOCR Docker container to release VRAM...")
    try:
        subprocess.run(["docker", "stop", "olmocr"], capture_output=True)
        print("Docker container stopped successfully.")
    except Exception as e:
        print(f"Error stopping container on shutdown: {e}")

atexit.register(cleanup_docker)


# ─── Helper: Build progress bar HTML ────────────────────────────────
def make_progress_bar_html(completed, total, elapsed_secs=0):
    pct = int((completed / total) * 100) if total > 0 else 0
    # ETA calculation
    eta_str = ""
    if completed > 0 and elapsed_secs > 0 and completed < total:
        rate = completed / elapsed_secs
        remaining = (total - completed) / rate
        if remaining < 60:
            eta_str = f"{int(remaining)}s remaining"
        elif remaining < 3600:
            eta_str = f"{int(remaining // 60)}m {int(remaining % 60)}s remaining"
        else:
            eta_str = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m remaining"
    elif completed >= total and total > 0:
        eta_str = "Complete"
    
    elapsed_str = ""
    if elapsed_secs > 0:
        if elapsed_secs < 60:
            elapsed_str = f"{int(elapsed_secs)}s elapsed"
        elif elapsed_secs < 3600:
            elapsed_str = f"{int(elapsed_secs // 60)}m {int(elapsed_secs % 60)}s elapsed"
        else:
            elapsed_str = f"{int(elapsed_secs // 3600)}h {int((elapsed_secs % 3600) // 60)}m elapsed"
    
    time_info = ""
    if elapsed_str and eta_str:
        time_info = f"<div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-top:4px;'><span>{elapsed_str}</span><span>{eta_str}</span></div>"
    elif elapsed_str:
        time_info = f"<div style='font-size:0.8rem; color:#94a3b8; margin-top:4px;'>{elapsed_str}</div>"
    
    return f"""<div style='width:100%;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
            <span style='font-size:0.9rem; color:#e2e8f0; font-weight:600;'>{completed}/{total} Pages</span>
            <span style='font-size:0.9rem; color:#818cf8; font-weight:600;'>{pct}%</span>
        </div>
        <div style='width:100%; background:#1e293b; border-radius:8px; height:12px; overflow:hidden;'>
            <div style='width:{pct}%; height:100%; background:linear-gradient(90deg, #6366f1, #3b82f6); border-radius:8px; transition:width 0.4s ease;'></div>
        </div>
        {time_info}
    </div>"""


# ─── Helper: Build file status table HTML ────────────────────────────
def make_file_status_html(file_mapping, file_page_counts, completed_files_set, failed_files_set=None):
    if failed_files_set is None:
        failed_files_set = set()
    
    rows = ""
    for idx in sorted(file_mapping.keys()):
        name = file_mapping[idx]
        pages = file_page_counts.get(idx, "?")
        if idx in failed_files_set:
            status = "<span style='color:#fca5a5;'>✗ Failed</span>"
        elif idx in completed_files_set:
            status = "<span style='color:#34d399;'>✓ Done</span>"
        else:
            status = "<span style='color:#94a3b8;'>⏳ Pending</span>"
        rows += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:6px 10px; color:#e2e8f0; font-size:0.85rem;'>{name}</td><td style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{pages}</td><td style='padding:6px 10px; text-align:center; font-size:0.85rem;'>{status}</td></tr>"
    
    return f"""<table style='width:100%; border-collapse:collapse;'>
        <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
            <th style='padding:6px 10px; color:#94a3b8; text-align:left; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>File</th>
            <th style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Pages</th>
            <th style='padding:6px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ─── Helper: Build upload manifest HTML ─────────────────────────────
def make_upload_manifest_html(file_mapping, file_page_counts, file_sizes):
    rows = ""
    total_pages = 0
    total_size = 0
    for idx in sorted(file_mapping.keys()):
        name = file_mapping[idx]
        pages = file_page_counts.get(idx, "?")
        size_bytes = file_sizes.get(idx, 0)
        if isinstance(pages, int):
            total_pages += pages
        total_size += size_bytes
        
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        rows += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'><td style='padding:5px 10px; color:#e2e8f0; font-size:0.85rem;'>{name}</td><td style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{pages}</td><td style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.85rem;'>{size_str}</td></tr>"
    
    if total_size < 1024 * 1024:
        total_size_str = f"{total_size / 1024:.1f} KB"
    else:
        total_size_str = f"{total_size / (1024 * 1024):.1f} MB"
    
    rows += f"<tr style='border-top:1px solid rgba(255,255,255,0.1);'><td style='padding:5px 10px; color:#818cf8; font-size:0.85rem; font-weight:600;'>Total ({len(file_mapping)} files)</td><td style='padding:5px 10px; color:#818cf8; text-align:center; font-size:0.85rem; font-weight:600;'>{total_pages}</td><td style='padding:5px 10px; color:#818cf8; text-align:center; font-size:0.85rem; font-weight:600;'>{total_size_str}</td></tr>"
    
    return f"""<table style='width:100%; border-collapse:collapse;'>
        <thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
            <th style='padding:5px 10px; color:#94a3b8; text-align:left; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>File</th>
            <th style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Pages</th>
            <th style='padding:5px 10px; color:#94a3b8; text-align:center; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;'>Size</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# Custom CSS for modern dark UI
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap');

body {
    background-color: #090d16 !important;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    background-color: #090d16 !important;
    max-width: 1800px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

.gradient-title {
    background: linear-gradient(135deg, #818cf8, #3b82f6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    text-align: left;
    margin-bottom: 2px !important;
    line-height: 1.2 !important;
}

.gradient-subtitle {
    color: #9ca3af !important;
    text-align: center;
    font-size: 1.1rem !important;
    margin-bottom: 30px !important;
}

.log-console textarea, .log-console code {
    background-color: #020617 !important;
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #1e293b !important;
    font-size: 0.85rem !important;
}

.stat-card {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    text-align: center;
}

.stat-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f3f4f6 !important;
}

.stat-label {
    font-size: 0.85rem !important;
    color: #9ca3af !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 5px;
}

.badge-idle { background-color: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-running { background-color: #1e3a8a; color: #60a5fa; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; animation: pulse 2s infinite; }
.badge-success { background-color: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-stopped { background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-failed { background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.preview-container {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    background: rgba(15, 23, 42, 0.3) !important;
    min-height: 350px;
}

/* Height containment: rendered preview scrolls internally */
.preview-scroll .prose,
.preview-scroll .md,
.preview-scroll .markdown-body,
.preview-scroll > div > div {
    max-height: 70vh !important;
    overflow-y: auto !important;
}

/* Height containment: raw markdown code editor */
.raw-md-scroll .cm-editor {
    max-height: 70vh !important;
}

/* Height containment: log viewer capped at 250px */
.log-console .cm-editor {
    max-height: 250px !important;
}

/* Compact download file components */
.compact-download {
    min-height: 0 !important;
}
.compact-download .file-preview,
.compact-download .upload-button,
.compact-download .wrap {
    min-height: 0 !important;
    padding: 6px 10px !important;
}

.status-container {
    padding: 10px 15px !important;
    margin: 0 !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 8px 0 16px 0;
}

.section-header {
    background: linear-gradient(90deg, rgba(99,102,241,0.15), transparent) !important;
    border-left: 3px solid #818cf8 !important;
    padding: 10px 16px !important;
    border-radius: 0 12px 12px 0 !important;
    margin-bottom: 12px !important;
}

.section-header h3 {
    margin: 0 !important;
    color: #c7d2fe !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

@media (max-width: 1200px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 0 12px !important;
    }
}

input, textarea, select, .wrap input {
    color: #e2e8f0 !important;
}

.code-wrap, .cm-editor, .cm-content {
    background-color: #020617 !important;
    color: #38bdf8 !important;
}

.accordion {
    border-color: rgba(255, 255, 255, 0.06) !important;
}

.file-preview {
    background: rgba(15, 23, 42, 0.5) !important;
}

/* File status table scrollable container */
.file-status-wrap {
    max-height: 200px;
    overflow-y: auto;
}
"""

def make_zip(markdown_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(markdown_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, markdown_dir)
                    zipf.write(file_path, arcname)


# ─── Yield tuple order ────────────────────────────────────────────────
# The process_pdfs generator yields a tuple with these outputs (in order):
#  0: log_viewer           - accumulated log text
#  1: status_badge         - HTML badge (Idle/Running/Success/Failed/Stopped)
#  2: progress_bar         - HTML progress bar
#  3: completed_pages_card - HTML stat card
#  4: failed_pages_card    - HTML stat card
#  5: file_selector        - dropdown choices/value
#  6: download_zip_btn     - file path or None
#  7: download_individual  - file path or None
#  8: start_btn            - interactive state
#  9: active_run_id        - run ID string
# 10: file_status_table    - HTML file status table
# 11: upload_manifest      - HTML manifest table

def _make_empty_yield(log_text, badge_html, progress_html, start_interactive=True, run_id=""):
    """Return a tuple for early-exit/error cases."""
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

    # Load file paths
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

    # ── Pre-flight server connectivity check ─────────────────────────
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

    # Count total pages and gather file info
    total_pages = 0
    file_mapping = {}   # index -> original name
    file_page_counts = {}  # index -> page count
    file_sizes = {}     # index -> size in bytes
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
            total_pages += 1  # Fallback

    # Build upload manifest
    manifest_html = make_upload_manifest_html(file_mapping, file_page_counts, file_sizes)

    # Create run directory
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(WORKSPACE_DIR, f"run_{run_timestamp}_{run_id[:8]}")
    inputs_dir = os.path.join(run_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    # Copy files with index prefixes to prevent duplicates
    copied_relative_paths = []
    for idx, path in enumerate(pdf_paths):
        orig_name = os.path.basename(path)
        safe_name = f"{idx}_{orig_name}"
        dest = os.path.join(inputs_dir, safe_name)
        shutil.copy(path, dest)
        copied_relative_paths.append(os.path.join("inputs", safe_name))

    # Save run info
    with active_runs_lock:
        active_runs[run_id] = {
            "stop": False,
            "proc": None,
            "run_dir": run_dir,
            "file_mapping": file_mapping
        }

    # Build command
    cmd = [
        "/home/owner/olmocr-env/bin/python", "-u",
        "-m", "olmocr.pipeline",
        ".",  # positional workspace argument
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

    # Start subprocess
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        with active_runs_lock:
            if run_id in active_runs:
                active_runs[run_id]["proc"] = proc
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

    # Start stdout listener thread
    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q))
    t.daemon = True
    t.start()

    # Streaming state variables
    accumulated_logs = ""
    completed_pages = 0
    failed_pages = 0
    vllm_running = 0
    vllm_queued = 0
    completed_file_indices = set()
    failed_file_indices = set()
    start_time = time.monotonic()

    # Regex patterns
    pattern_completed = re.compile(r"completed_pages\s+([\d.]+)")
    pattern_failed = re.compile(r"failed_pages\s+([\d.]+)")
    pattern_vllm_queue = re.compile(r"vllm running req:\s*(\d+)\s+queue req:\s*(\d+)")
    pattern_vllm_standalone_queue = re.compile(r"Running:\s*(\d+).*?(?:Waiting|Pending):\s*(\d+)")

    # Initial file status
    file_status_html = make_file_status_html(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)

    # Yield initial progress
    yield (
        "Initializing pipeline...",
        gr.update(value="<span class='badge-running'>Running</span>"),
        make_progress_bar_html(0, total_pages),
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

    # Track incremental choices
    streaming_choices = []
    streaming_dropdown_value = None

    try:
        while t.is_alive() or not q.empty():
            # Check for user termination
            with active_runs_lock:
                if run_id in active_runs and active_runs[run_id]["stop"]:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    elapsed = time.monotonic() - start_time
                    yield (
                        accumulated_logs + "\n\n[PROCESS TERMINATED BY USER]\n",
                        gr.update(value="<span class='badge-stopped'>Stopped</span>"),
                        make_progress_bar_html(completed_pages, total_pages, elapsed),
                        gr.update(value=f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"),
                        gr.update(value=f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"),
                        gr.update(choices=streaming_choices, value=streaming_dropdown_value),
                        None,
                        None,
                        gr.update(interactive=True),
                        "",
                        make_file_status_html(file_mapping, file_page_counts, completed_file_indices, failed_file_indices),
                        manifest_html,
                    )
                    return

            try:
                line = q.get_nowait()
            except queue.Empty:
                # No new log line yet
                threading.Event().wait(0.05)
                continue

            accumulated_logs += line
            
            # Parse failed_pages from logs
            if "failed_pages" in line:
                match_f = pattern_failed.search(line)
                if match_f:
                    try:
                        failed_pages = max(failed_pages, int(float(match_f.group(1))))
                    except (ValueError, TypeError):
                        pass
            
            # Scan markdown output directory for completed files
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
                            temp_completed_pages += file_page_counts.get(file_idx, 1)
                            # Add to incremental dropdown
                            orig_name = file_mapping.get(file_idx, md_file)
                            choice_tuple = (orig_name, md_file)
                            if choice_tuple not in streaming_choices:
                                streaming_choices.append(choice_tuple)
                                if streaming_dropdown_value is None:
                                    streaming_dropdown_value = md_file
                        else:
                            temp_completed_pages += file_page_counts.get(file_idx, 1)
                completed_pages = max(completed_pages, temp_completed_pages)

            # Check vLLM queue logs
            vllm_match = pattern_vllm_queue.search(line)
            if vllm_match:
                vllm_running = int(vllm_match.group(1))
                vllm_queued = int(vllm_match.group(2))
            else:
                vllm_match_standalone = pattern_vllm_standalone_queue.search(line)
                if vllm_match_standalone:
                    vllm_running = int(vllm_match_standalone.group(1))
                    vllm_queued = int(vllm_match_standalone.group(2))

            elapsed = time.monotonic() - start_time
            
            status_html = f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"
            failed_html = f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"
            file_status_html = make_file_status_html(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)
            
            yield (
                accumulated_logs,
                gr.update(value="<span class='badge-running'>Running</span>"),
                make_progress_bar_html(completed_pages, total_pages, elapsed),
                gr.update(value=status_html),
                gr.update(value=failed_html),
                gr.update(choices=streaming_choices, value=streaming_dropdown_value),
                None,
                None,
                gr.update(interactive=False),
                run_id,
                file_status_html,
                manifest_html,
            )

        # Process ended
        proc.wait()
        exit_code = proc.returncode

        # Final filesystem check for results
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
                
            # Create ZIP
            zip_file_path = os.path.join(run_dir, "all_markdown_results.zip")
            make_zip(md_inputs_dir, zip_file_path)

        # Recount completed pages from final file set
        final_completed = sum(file_page_counts.get(idx, 1) for idx in completed_file_indices)
        completed_pages = max(completed_pages, final_completed)

        elapsed = time.monotonic() - start_time

        if exit_code == 0:
            status_text = "<span class='badge-success'>Success</span>"
        else:
            status_text = "<span class='badge-failed'>Failed</span>"

        file_status_html = make_file_status_html(file_mapping, file_page_counts, completed_file_indices, failed_file_indices)

        yield (
            accumulated_logs + f"\n\n[PROCESS EXITED WITH CODE {exit_code}]\n",
            gr.update(value=status_text),
            make_progress_bar_html(completed_pages, total_pages, elapsed),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{completed_pages}</div><div class='stat-label'>Completed Pages</div></div>"),
            gr.update(value=f"<div class='stat-card'><div class='stat-value'>{failed_pages}</div><div class='stat-label'>Failed Pages</div></div>"),
            gr.update(choices=choices, value=dropdown_value),
            zip_file_path,
            None, # Individual file will be set when user selects dropdown
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
            make_progress_bar_html(completed_pages, total_pages, elapsed),
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
        with active_runs_lock:
            if run_id in active_runs:
                active_runs[run_id]["completed"] = True

def stop_processing(run_id):
    if not run_id:
        return "<span class='badge-idle'>No active process to stop.</span>"
    with active_runs_lock:
        if run_id in active_runs:
            active_runs[run_id]["stop"] = True
            proc = active_runs[run_id]["proc"]
            if proc:
                proc.terminate()
            return f"<span class='badge-stopped'>Stop request sent for run {run_id[:8]}.</span>"
    return "<span class='badge-idle'>Process not found or already ended.</span>"

def load_markdown_content(selected_file, run_id_state):
    # selected_file is the dropdown choice (value is the filename inside run_dir/markdown/inputs)
    if not selected_file or not run_id_state:
        return "", "", None

    with active_runs_lock:
        run_info = active_runs.get(run_id_state)
        if not run_info:
            return "Run info not found.", "Run info not found.", None
        run_dir = run_info["run_dir"]

    file_path = os.path.join(run_dir, "markdown", "inputs", selected_file)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, content, file_path
        except Exception as e:
            return f"Error reading file: {e}", f"Error reading file: {e}", None
    return "File not found.", "File not found.", None

# GUI layout construction
settings = load_settings()

dark_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    block_background_fill="rgba(17, 24, 39, 0.7)",
    block_background_fill_dark="rgba(17, 24, 39, 0.7)",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    block_label_text_color="#e2e8f0",
    block_label_text_color_dark="#e2e8f0",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#9ca3af",
    body_text_color_subdued_dark="#9ca3af",
    input_background_fill="#1e293b",
    input_background_fill_dark="#1e293b",
    input_border_color="rgba(255, 255, 255, 0.1)",
    input_border_color_dark="rgba(255, 255, 255, 0.1)",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #3b82f6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #3b82f6)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="rgba(30, 41, 59, 0.8)",
    button_secondary_background_fill_dark="rgba(30, 41, 59, 0.8)",
    button_secondary_text_color="#e2e8f0",
    button_secondary_text_color_dark="#e2e8f0",
    border_color_accent="rgba(99, 102, 241, 0.5)",
    border_color_accent_dark="rgba(99, 102, 241, 0.5)",
    shadow_drop="0 4px 24px rgba(0,0,0,0.4)",
    shadow_drop_lg="0 8px 32px rgba(0,0,0,0.5)",
)

with gr.Blocks(title="OLMOCR PDF Suite") as demo:
    # State tracking
    active_run_id = gr.State("")

    # ── Header ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(
                "<h1 class='gradient-title' style='margin:0;'>OLMOCR PDF-to-Markdown Suite</h1>"
                "<p style='color:#9ca3af; margin:4px 0 0 0; font-size:1rem;'>High-performance layout-aware PDF OCR pipeline using vision-language models</p>"
            )
        with gr.Column(scale=1, elem_classes=["glass-panel", "status-container"]):
            gr.HTML("<h3 style='margin:0 0 8px 0; color:#e2e8f0; font-size:1rem; font-weight:600; text-align:center;'>🐳 Inference Status</h3>")
            backend_status_badge = gr.HTML("<span class='badge-idle'>Checking Backend...</span>")
            with gr.Row():
                header_docker_start_btn = gr.Button("▶️ Start", variant="secondary", size="sm")
                header_docker_stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="sm")

    # ── Main Content ────────────────────────────────────────────────
    with gr.Row():
        # Left sidebar — Settings (collapsible)
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Pipeline Settings", open=False, elem_classes=["glass-panel"]):
                server_url_input = gr.Textbox(
                    label="vLLM OpenAI Server URL", 
                    value=settings["server_url"], 
                    placeholder="http://localhost:8000/v1"
                )
                model_name_input = gr.Textbox(
                    label="Model Name", 
                    value=settings["model_name"], 
                    placeholder="allenai/olmOCR-2-7B-1025-FP8"
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    workers_input = gr.Slider(
                        label="Workers", 
                        minimum=1, maximum=64, step=1, 
                        value=settings["workers"]
                    )
                    max_concurrent_input = gr.Slider(
                        label="Max Concurrent Requests", 
                        minimum=1, maximum=2000, step=10, 
                        value=settings["max_concurrent_requests"]
                    )
                    target_dim_input = gr.Slider(
                        label="Target Longest Image Dimension", 
                        minimum=512, maximum=2048, step=64, 
                        value=settings["target_longest_image_dim"]
                    )
                    max_retries_input = gr.Slider(
                        label="Max Page Retries", 
                        minimum=1, maximum=20, step=1, 
                        value=settings["max_page_retries"]
                    )
                    guided_decoding_input = gr.Checkbox(
                        label="Enable Guided Decoding (YAML structure)", 
                        value=settings["guided_decoding"]
                    )
                
                save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
                config_status = gr.Markdown()
                
            with gr.Accordion("🐳 Local Inference Server (Docker)", open=False, elem_classes=["glass-panel"]):
                docker_status_info = gr.Markdown("Manage the local GPU inference container.")
                
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token", 
                    value=settings["hf_token"], 
                    type="password"
                )
                docker_port_input = gr.Number(
                    label="Docker Host Port", 
                    value=settings["docker_port"], 
                    precision=0
                )
                docker_gpu_mem_input = gr.Slider(
                    label="GPU Memory Utilization", 
                    minimum=0.1, maximum=1.0, step=0.05, 
                    value=settings["docker_gpu_mem"]
                )
                docker_max_model_len_input = gr.Slider(
                    label="Max Model Length", 
                    minimum=2048, maximum=32768, step=1024, 
                    value=settings["docker_max_model_len"]
                )
                
                with gr.Row():
                    docker_start_btn = gr.Button("▶️ Start", variant="secondary")
                    docker_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                
                docker_recreate_btn = gr.Button("🔄 Recreate & Run", variant="primary")
                docker_action_status = gr.Markdown()

        # Center — Upload, Processing Controls, Monitoring, Log
        with gr.Column(scale=3):
            with gr.Row():
                # Upload area
                with gr.Column(scale=2, elem_classes=["glass-panel"]):
                    gr.Markdown("### 📥 Source Documents")
                    pdf_uploader = gr.File(
                        label="Upload / Drag-and-drop PDFs", 
                        file_count="multiple", 
                        file_types=[".pdf"]
                    )
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Batch Processing", variant="primary")
                        stop_btn = gr.Button("🛑 Stop Process", variant="stop", interactive=False)

                # Monitoring cards
                with gr.Column(scale=1, elem_classes=["glass-panel"]):
                    gr.Markdown("### 📊 Monitoring")
                    status_badge = gr.HTML("<span class='badge-idle'>Idle</span>", label="Status")
                    progress_bar = gr.HTML(
                        make_progress_bar_html(0, 0),
                        label="Batch Progress"
                    )
                    with gr.Row():
                        completed_pages_card = gr.HTML(
                            "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Completed Pages</div></div>",
                            visible=True
                        )
                        failed_pages_card = gr.HTML(
                            "<div class='stat-card'><div class='stat-value'>0</div><div class='stat-label'>Failed Pages</div></div>",
                            visible=True
                        )

            # Upload manifest + File status (side by side)
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["glass-panel"]):
                    gr.Markdown("### 📋 Upload Manifest")
                    upload_manifest_display = gr.HTML("", elem_classes=["file-status-wrap"])
                with gr.Column(scale=1, elem_classes=["glass-panel"]):
                    gr.Markdown("### 📁 Per-File Status")
                    file_status_table = gr.HTML("", elem_classes=["file-status-wrap"])

            # Log viewer — full width under the upload + monitoring row
            with gr.Row(elem_classes=["glass-panel"]):
                with gr.Column():
                    gr.Markdown("### 📜 System Output Log")
                    log_viewer = gr.Code(
                        label="Logs",
                        language="shell",
                        value="",
                        interactive=False,
                        lines=10,
                        elem_classes=["log-console"]
                    )

    # ── Results Section (full-width, below main) ───────────────────
    gr.HTML("<hr class='section-divider'>")
    with gr.Row(elem_classes=["glass-panel"]):
        with gr.Column(scale=3):
            file_selector = gr.Dropdown(
                label="📄 Select Processed Document", 
                choices=[], 
                interactive=True
            )
        with gr.Column(scale=1):
            download_individual_btn = gr.File(
                label="Download Markdown", 
                interactive=False,
                elem_classes=["compact-download"]
            )
        with gr.Column(scale=1):
            download_zip_btn = gr.File(
                label="Download All (ZIP)", 
                interactive=False,
                elem_classes=["compact-download"]
            )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            with gr.Row():
                gr.Markdown("### ✍️ Raw Markdown Output")
                copy_btn = gr.Button("📋 Copy", variant="secondary", size="sm")
            raw_markdown = gr.Code(
                label="Raw Text",
                language="markdown",
                interactive=False,
                lines=22,
                elem_classes=["raw-md-scroll"]
            )
        with gr.Column(scale=1, elem_classes=["glass-panel", "preview-container", "preview-scroll"]):
            gr.Markdown("### 👁️ Rendered Preview")
            rendered_markdown = gr.Markdown(value="Select a processed document to preview.")

    # ── Event handlers ─────────────────────────────────────────────

    # Copy to clipboard via JS
    copy_btn.click(
        None,
        inputs=[raw_markdown],
        js="(text) => { navigator.clipboard.writeText(text || ''); }"
    )

    def trigger_save_settings(url, model, wrk, concat, dim, retries, guided, d_port, d_gpu, d_maxlen, d_token):
        user_config = {
            "server_url": url,
            "model_name": model,
            "workers": int(wrk),
            "max_concurrent_requests": int(concat),
            "target_longest_image_dim": int(dim),
            "max_page_retries": int(retries),
            "guided_decoding": guided,
            "docker_port": int(d_port),
            "docker_gpu_mem": float(d_gpu),
            "docker_max_model_len": int(d_maxlen),
            "hf_token": d_token
        }
        return save_settings(user_config)

    save_config_btn.click(
        trigger_save_settings,
        inputs=[
            server_url_input, model_name_input, 
            workers_input, max_concurrent_input, 
            target_dim_input, max_retries_input, 
            guided_decoding_input,
            docker_port_input, docker_gpu_mem_input,
            docker_max_model_len_input, hf_token_input
        ],
        outputs=[config_status]
    )

    start_btn.click(
        process_pdfs,
        inputs=[
            pdf_uploader, server_url_input, model_name_input,
            workers_input, max_concurrent_input, max_retries_input,
            target_dim_input, guided_decoding_input
        ],
        outputs=[
            log_viewer, status_badge, progress_bar, 
            completed_pages_card, failed_pages_card,
            file_selector, download_zip_btn, download_individual_btn,
            start_btn, active_run_id,
            file_status_table, upload_manifest_display,
        ]
    )
    
    # Update stop button interactivity when run starts
    start_btn.click(
        lambda: gr.update(interactive=True),
        outputs=[stop_btn]
    )

    # Stop button: route feedback to status_badge (not config_status)
    stop_btn.click(
        stop_processing,
        inputs=[active_run_id],
        outputs=[status_badge]
    )
    
    # Disable stop button after stopped/completed
    stop_btn.click(
        lambda: gr.update(interactive=False),
        outputs=[stop_btn]
    )

    file_selector.change(
        load_markdown_content,
        inputs=[file_selector, active_run_id],
        outputs=[raw_markdown, rendered_markdown, download_individual_btn]
    )

    # Docker event handlers
    def ui_start_container(port):
        success, msg = start_docker_container()
        _, badge = get_docker_status_str(port)
        return msg, badge

    def ui_stop_container(port):
        success, msg = stop_docker_container()
        _, badge = get_docker_status_str(port)
        return msg, badge

    def ui_recreate_container(hf_token, port, model, gpu_mem, max_model_len):
        success, msg = create_docker_container(hf_token, port, model, gpu_mem, max_model_len)
        _, badge = get_docker_status_str(port)
        
        settings = load_settings()
        settings.update({
            "hf_token": hf_token,
            "docker_port": int(port),
            "model_name": model,
            "docker_gpu_mem": float(gpu_mem),
            "docker_max_model_len": int(max_model_len),
            "server_url": f"http://localhost:{int(port)}/v1"
        })
        save_settings(settings)
        new_url = f"http://localhost:{int(port)}/v1"
        return msg, badge, new_url

    # Sidebar Docker buttons
    docker_start_btn.click(
        ui_start_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge]
    )

    docker_stop_btn.click(
        ui_stop_container,
        inputs=[docker_port_input],
        outputs=[docker_action_status, backend_status_badge]
    )

    docker_recreate_btn.click(
        ui_recreate_container,
        inputs=[
            hf_token_input, docker_port_input, model_name_input,
            docker_gpu_mem_input, docker_max_model_len_input
        ],
        outputs=[docker_action_status, backend_status_badge, server_url_input]
    )

    # Header Docker buttons (same handlers, no status text output)
    def ui_header_start(port):
        start_docker_container()
        _, badge = get_docker_status_str(port)
        return badge

    def ui_header_stop(port):
        stop_docker_container()
        _, badge = get_docker_status_str(port)
        return badge

    header_docker_start_btn.click(
        ui_header_start,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    header_docker_stop_btn.click(
        ui_header_stop,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    # Periodic Backend Status Check
    status_timer = gr.Timer(value=5)
    
    def periodic_status_check(port_val):
        if port_val is None:
            port_val = 8000
        _, badge_html = get_docker_status_str(int(port_val))
        return badge_html

    status_timer.tick(
        periodic_status_check,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

    demo.load(
        periodic_status_check,
        inputs=[docker_port_input],
        outputs=[backend_status_badge]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css, theme=dark_theme, allowed_paths=["/home/owner"])
