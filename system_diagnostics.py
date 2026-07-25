import logging
import os
import re
import socket
import subprocess
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def get_service_latency(
    service_name: str, host: str = "127.0.0.1", port: int | None = None
) -> tuple[bool, float, str | None]:
    ports = {"postgres": 5432, "redis": 6379, "minio": 9000, "qdrant": 6333, "vllm": 8000}
    if port is None:
        port = ports.get(service_name)

    start_time = time.time()
    extra_info = None
    try:
        if service_name == "postgres":
            import psycopg2

            from secrets_config import get_db_password

            conn = psycopg2.connect(
                dbname=os.environ.get("OLMOCR_PG_DB", "olmocr_rag"),
                user=os.environ.get("OLMOCR_PG_USER", "olmocr"),
                password=get_db_password(),
                host=host,
                port=port,
                connect_timeout=1,
            )
            conn.close()
        elif service_name == "redis":
            import redis

            r = redis.Redis(host=host, port=port, socket_connect_timeout=1)
            r.ping()
        elif service_name == "minio":
            res = requests.get(f"http://{host}:{port}/minio/health/live", timeout=1)
            if res.status_code != 200:
                raise Exception()
        elif service_name == "qdrant":
            res = requests.get(f"http://{host}:{port}/readyz", timeout=1)
            if res.status_code != 200:
                raise Exception()
        elif service_name == "vllm":
            res = requests.get(f"http://{host}:{port}/v1/models", timeout=1)
            if res.status_code != 200:
                raise Exception()
            try:
                data = res.json()
                models = data.get("data", [])
                if models:
                    extra_info = models[0].get("id")
                else:
                    extra_info = "None Loaded"
            except Exception:
                extra_info = "Unknown"
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((host, port))
            s.close()

        latency = (time.time() - start_time) * 1000.0
        return True, latency, extra_info
    except Exception:
        return False, 0.0, None


def get_vllm_loading_progress() -> dict[str, Any] | None:
    # Fatal error patterns that indicate vLLM has crashed or failed to load.
    # Checked before positive progress indicators so failures are surfaced
    # instead of showing "loading" indefinitely.
    _FAILURE_PATTERNS = [
        (r"CUDA out of memory", "CUDA Out of Memory"),
        (r"torch\.OutOfMemoryError", "GPU Out of Memory"),
        (r"No available memory for the cache blocks", "Insufficient VRAM for KV Cache"),
        (r"Cannot re-initialize CUDA in forked subprocess", "CUDA Fork Error"),
        (r"max seq len .+ is larger than the maximum", "Max Sequence Length Exceeded"),
        (r"The model's max seq len", "Max Sequence Length Exceeded"),
        (r"not enough memory", "Insufficient Memory"),
        (r"Killed\s*$", "Process Killed (OOM)"),
    ]

    try:
        res = subprocess.run(
            ["docker", "logs", "--tail", "50", "olmocr"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout:
            lines = res.stdout.splitlines()

            # ── Failure detection ─────────────────────────────
            # Scan from the end of the log for fatal error patterns.
            # Only flag a failure if no positive "server ready" line follows it
            # (to avoid false positives from prior container runs).
            has_ready_line = any(
                "Application startup complete" in ln
                or "Uvicorn running on" in ln
                or "Started server process" in ln
                for ln in lines
            )
            if not has_ready_line:
                for line in reversed(lines):
                    for pattern, label in _FAILURE_PATTERNS:
                        if re.search(pattern, line, re.IGNORECASE):
                            return {
                                "pct": -1,
                                "shards_loaded": "FAILED",
                                "shards_total": "ERROR",
                                "eta": label,
                                "failed": True,
                                "error_line": line.strip()[:200],
                            }
                    # Also detect Python tracebacks
                    if "Traceback (most recent call last)" in line:
                        # Find the last line of the traceback for error summary
                        error_summary = "Python Exception"
                        for tl in reversed(lines):
                            if tl.strip() and not tl.startswith(" ") and "Traceback" not in tl:
                                error_summary = tl.strip()[:120]
                                break
                        return {
                            "pct": -1,
                            "shards_loaded": "FAILED",
                            "shards_total": "ERROR",
                            "eta": error_summary,
                            "failed": True,
                            "error_line": error_summary,
                        }

            # ── Positive progress detection ───────────────────
            # Check for CUDA graphs capture
            for line in reversed(lines):
                if "Capturing CUDA graphs" in line:
                    match = re.search(
                        r"Capturing CUDA graphs.*:\s*(\d+)%\s*\|\s*.*?\|\s*(\d+)/(\d+)", line
                    )
                    if match:
                        return {
                            "pct": 100,
                            "shards_loaded": "Graph",
                            "shards_total": "Active",
                            "eta": f"Capturing CUDA graphs ({match.group(2)}/{match.group(3)})",
                        }

            # Check for autotuning
            for line in reversed(lines):
                if "Tuning fp4_gemm:" in line:
                    match = re.search(r"Tuning fp4_gemm:\s*(\d+)%\s*\|\s*.*?\|\s*(\d+)/(\d+)", line)
                    if match:
                        return {
                            "pct": 100,
                            "shards_loaded": "Tuning",
                            "shards_total": "Active",
                            "eta": f"Autotuning FP4 GEMM ({match.group(2)}/{match.group(3)})",
                        }
                if "Autotuning process starts" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Tuning",
                        "shards_total": "Active",
                        "eta": "Autotuning FP4 GEMM starting...",
                    }

            # Check for profiling/warmup
            for line in reversed(lines):
                if "Profiling CUDA graph memory" in line or "Estimated CUDA graph memory" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Warmup",
                        "shards_total": "Active",
                        "eta": "Profiling CUDA graph memory...",
                    }
                if "Initial profiling/warmup run" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Warmup",
                        "shards_total": "Active",
                        "eta": "Warming up engine...",
                    }

            # Check for torch compilation
            for line in reversed(lines):
                if "torch.compile took" in line or "saved AOT compiled function" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Compile",
                        "shards_total": "Active",
                        "eta": "Compilation complete! Initializing engine & KV cache...",
                    }
                if "Compiling a graph for compile range" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Compile",
                        "shards_total": "Active",
                        "eta": "Compiling model graphs & CUDA PTX (torch.compile)...",
                    }
                if "for vLLM's torch.compile" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Compile",
                        "shards_total": "Active",
                        "eta": "Compiling model graphs & CUDA PTX (torch.compile)...",
                    }

            # Check for loading weights / model loading
            for line in reversed(lines):
                if "Loading weights took" in line or "Model loading took" in line:
                    return {
                        "pct": 100,
                        "shards_loaded": "Init",
                        "shards_total": "Active",
                        "eta": "Initializing engine...",
                    }

            # Fallback to safetensors loading shards
            for line in reversed(lines):
                if "Loading safetensors checkpoint shards:" in line:
                    match = re.search(
                        r"Loading safetensors checkpoint shards:\s*(\d+)%\s*Completed\s*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]",
                        line,
                    )
                    if match:
                        pct = int(match.group(1))
                        shards_loaded = int(match.group(2))
                        shards_total = int(match.group(3))
                        time_info = match.group(4)

                        eta = "Unknown"
                        if "<" in time_info:
                            remaining = time_info.split("<")[1].split(",")[0].strip()
                            if remaining == "?":
                                eta = "Calculating..."
                            elif ":" in remaining:
                                parts = remaining.split(":")
                                if len(parts) == 2:
                                    eta = f"{int(parts[0])}m {int(parts[1])}s"
                                elif len(parts) == 3:
                                    eta = f"{int(parts[0])}h {int(parts[1])}m {int(parts[2])}s"
                            else:
                                eta = remaining

                        return {
                            "pct": pct,
                            "shards_loaded": shards_loaded,
                            "shards_total": shards_total,
                            "eta": eta,
                        }
    except Exception:
        pass
    return None


def check_backing_services_data(
    service_history: dict[str, list[float]] | None = None, vllm_port: int = 8000
) -> dict[str, Any]:
    if service_history is None:
        service_history = {}
    services_data = {}
    all_healthy = True
    failed_services = []
    vllm_model = None

    services = ["postgres", "redis", "minio", "qdrant", "vllm"]
    for s in services:
        if s == "vllm":
            is_up, latency, extra_info = get_service_latency(s, port=vllm_port)
        else:
            is_up, latency, extra_info = get_service_latency(s)

        if is_up:
            if s not in service_history:
                service_history[s] = []
            service_history[s].append(latency)
            if len(service_history[s]) > 8:
                service_history[s].pop(0)

            if s == "vllm":
                vllm_model = extra_info
        else:
            all_healthy = False
            failed_services.append(s)

        services_data[s] = {
            "is_up": is_up,
            "latency": latency,
            "extra_info": extra_info,
            "latency_history": list(service_history.get(s, [])),
        }

    vllm_progress = None
    if not services_data["vllm"]["is_up"]:
        vllm_progress = get_vllm_loading_progress()

    return {
        "all_healthy": all_healthy,
        "services": services_data,
        "failed_services": failed_services,
        "vllm_model": vllm_model,
        "vllm_progress": vllm_progress,
    }


def get_docker_containers() -> dict[str, str]:
    containers = {}
    try:
        res = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}}|{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            for line in res.stdout.strip().split("\n"):
                if "|" in line:
                    cid, name = line.split("|", 1)
                    containers[cid] = name
    except Exception:
        pass
    return containers


def resolve_process_details(pid: int, name_from_smi: str) -> tuple[str, bool, str]:
    cmdline = ""
    is_docker = False
    container_name = ""
    try:
        with open(f"/proc/{pid}/cmdline") as f:
            raw = f.read()
            parts = [p for p in raw.split("\x00") if p]
            if parts:
                cmdline = " ".join(parts)
    except Exception:
        pass
    if not cmdline:
        cmdline = name_from_smi
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            cgroup_content = f.read()
            if "docker" in cgroup_content or "containerd" in cgroup_content:
                is_docker = True
                for line in cgroup_content.splitlines():
                    if "docker-" in line:
                        parts = line.split("docker-")
                        if len(parts) > 1:
                            container_name = parts[1].split(".")[0][:12]
                            break
                    elif "docker/" in line:
                        parts = line.split("docker/")
                        if len(parts) > 1:
                            container_name = parts[1].split("/")[0][:12]
                            break
    except Exception:
        pass
    return cmdline, is_docker, container_name


def get_display_name(cmdline: str, default_name: str) -> str:
    if not cmdline:
        return default_name
    parts = cmdline.split(" ")
    first = parts[0]
    basename = os.path.basename(first)
    if basename:
        if basename.startswith("python") and len(parts) > 1:
            script_name = os.path.basename(parts[1])
            return f"python: {script_name}"
        return basename
    return default_name


def get_gpu_metrics_data() -> dict[str, Any]:
    cuda_available = False
    gpu_name = "N/A"
    vram_used = 0.0
    vram_total = 0.0
    vram_pct = 0.0
    local_torch_vram = 0.0

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            local_torch_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
            vram_used = local_torch_vram
    except Exception:
        pass

    smi_vram_used = 0.0
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            cuda_available = True
            lines = res.stdout.strip().split("\n")
            parts = lines[0].split(",")
            gpu_name = parts[0].strip()

            raw_used = parts[1].strip() if len(parts) > 1 else ""
            raw_total = parts[2].strip() if len(parts) > 2 else ""

            if raw_used and raw_used.upper() != "[N/A]":
                try:
                    smi_vram_used = float(raw_used)
                    vram_used = smi_vram_used
                except (ValueError, TypeError):
                    pass

            if raw_total and raw_total.upper() != "[N/A]":
                try:
                    vram_total = float(raw_total)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass

    if cuda_available and vram_total == 0.0:
        try:
            import psutil

            vram_total = psutil.virtual_memory().total / (1024 * 1024)
        except Exception:
            pass

    if cuda_available and vram_total > 0:
        processes = []
        try:
            res_proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3)
            if res_proc.returncode == 0 and res_proc.stdout:
                lines = res_proc.stdout.splitlines()
                in_process_section = False
                for line in lines:
                    if "Processes:" in line:
                        in_process_section = True
                        continue
                    if in_process_section:
                        if "=====" in line:
                            continue
                        if "+-----" in line:
                            if len(processes) > 0:
                                break
                            continue
                        match = re.search(
                            r"\|\s+(\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+([CG]|[C\+G])\s+(.+?)\s+(\d+)(?:\.\.\.|\s*MiB)?\s*\|",
                            line,
                        )
                        if match:
                            gpu_id = int(match.group(1))
                            pid = int(match.group(2))
                            proc_type = match.group(3)
                            proc_name = match.group(4).strip()
                            vram_used_proc = int(match.group(5))
                            processes.append(
                                {
                                    "gpu_id": gpu_id,
                                    "pid": pid,
                                    "type": proc_type,
                                    "name": proc_name,
                                    "vram": vram_used_proc,
                                }
                            )
        except Exception:
            pass

        # Use system-wide process sum if global nvidia-smi query didn't give a positive value
        proc_vram_sum = float(sum(p["vram"] for p in processes))
        if smi_vram_used > 0.0:
            vram_used = smi_vram_used
        elif proc_vram_sum > 0.0:
            vram_used = max(proc_vram_sum, local_torch_vram)
        else:
            vram_used = local_torch_vram

        vram_pct = min(100.0, (vram_used / vram_total) * 100.0)

        docker_map = get_docker_containers()
        essential_keywords = ["xorg", "gnome-shell", "gdm", "kwin", "wayland", "systemd", "lightdm"]

        resolved_procs = []
        vram_reclaimable = 0.0

        for p in processes:
            cmdline, is_docker, container_id = resolve_process_details(p["pid"], p["name"])
            container_name = docker_map.get(container_id, container_id) if container_id else ""

            is_essential = any(
                k in cmdline.lower() or k in p["name"].lower() for k in essential_keywords
            )
            reclaimable = not is_essential

            display_name = get_display_name(cmdline, p["name"])

            if is_docker:
                type_text = f"Docker: {container_name}" if container_name else "Docker"
                type_badge_style = "background: rgba(245, 158, 11, 0.15); color: #f59e0b;"
                action_text = (
                    f"Stop container '{container_name}'" if container_name else "Stop container"
                )
                action_color = "#f59e0b"
            elif is_essential:
                type_text = "System Graphics"
                type_badge_style = "background: rgba(148, 163, 184, 0.15); color: #94a3b8;"
                action_text = "System process (essential)"
                action_color = "#94a3b8"
            else:
                type_text = "Application"
                type_badge_style = "background: rgba(59, 130, 246, 0.15); color: #60a5fa;"
                action_text = "Close application / process"
                action_color = "#60a5fa"

            if reclaimable:
                vram_reclaimable += p["vram"]

            resolved_procs.append(
                {
                    "display_name": display_name,
                    "cmdline": cmdline,
                    "pid": p["pid"],
                    "vram": p["vram"],
                    "type_text": type_text,
                    "type_badge_style": type_badge_style,
                    "action_text": action_text,
                    "action_color": action_color,
                    "is_essential": is_essential,
                }
            )

        vram_free = max(0.0, vram_total - vram_used)
        vram_potential_free = min(vram_total, vram_free + vram_reclaimable)

        return {
            "cuda_available": True,
            "gpu_name": gpu_name,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "vram_pct": vram_pct,
            "vram_free": vram_free,
            "vram_reclaimable": vram_reclaimable,
            "vram_potential_free": vram_potential_free,
            "processes": resolved_procs,
        }
    else:
        return {
            "cuda_available": False,
            "gpu_name": "N/A",
            "vram_used": 0.0,
            "vram_total": 0.0,
            "vram_pct": 0.0,
            "vram_free": 0.0,
            "vram_reclaimable": 0.0,
            "vram_potential_free": 0.0,
            "processes": [],
        }


def generate_diagnostic_report_file(port_val: int = 8000) -> str:
    from settings_manager import load_settings

    settings = load_settings()
    backing_data = check_backing_services_data({}, port_val)
    gpu_data = get_gpu_metrics_data()

    report = []
    report.append("# IQ-RAG System Diagnostics Report")
    report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 1. System Health Status")
    all_healthy = backing_data["all_healthy"]
    report.append(f"Overall Status: {'✓ HEALTHY' if all_healthy else '✗ DEGRADED'}")

    report.append("\n## 2. Backing Services Latency & Connection")
    for s, info in backing_data["services"].items():
        is_up = info["is_up"]
        lat = info["latency"]
        extra = info["extra_info"]
        report.append(
            f"- **{s.capitalize()}**: {'UP' if is_up else 'DOWN'} | Latency: {lat:.2f}ms | Extra: {extra}"
        )

    report.append("\n## 3. GPU Hardware & VRAM Telemetry")
    if gpu_data["cuda_available"]:
        report.append(f"- **GPU Model**: {gpu_data['gpu_name']}")
        report.append(f"- **VRAM Allocated**: {gpu_data['vram_used']:.1f} MB")
        report.append(f"- **VRAM Free**: {gpu_data['vram_free']:.1f} MB")
        report.append(f"- **VRAM Total**: {gpu_data['vram_total']:.1f} MB")
        report.append(f"- **VRAM Usage %**: {gpu_data['vram_pct']:.2f}%")
        report.append(f"- **Reclaimable VRAM**: {gpu_data['vram_reclaimable']:.1f} MB")
        report.append(f"- **Potential Max Free**: {gpu_data['vram_potential_free']:.1f} MB")

        report.append("\n### Active GPU Processes")
        for p in gpu_data["processes"]:
            report.append(
                f"- PID: {p['pid']} | {p['display_name']} | VRAM: {p['vram']} MB | Type: {p['type_text']}"
            )
    else:
        report.append("- CUDA is unavailable. Running on Host CPU.")

    report.append("\n## 4. Application Configuration Settings")
    for k, v in settings.items():
        if k == "hf_token" and v:
            v = "********"
        report.append(f"- **{k}**: {v}")

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(workspace_dir, "workspace")
    os.makedirs(target_dir, exist_ok=True)
    report_path = os.path.join(target_dir, "diagnostic_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return report_path


def format_bytes_human(num_bytes: int) -> str:
    """Format byte count into human-readable string (e.g. 19.1 GB, 450 MB)."""
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(num_bytes)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    if i == 0:
        return f"{int(size)} B"
    return f"{size:.2f} {units[i]}"


def get_installed_models_data() -> dict[str, Any]:
    """
    Scan all HuggingFace cache locations for installed models, computing total disk size,
    context lengths, model categories, and active status.
    """
    import json

    from settings_manager import MODEL_MAX_CONTENT_LENGTHS, load_settings

    settings = load_settings()
    active_models = {
        settings.get("model_name", ""),
        settings.get("analysis_model_name", ""),
        settings.get("embedding_model", ""),
        settings.get("reranker_model", ""),
    }
    active_models.discard("")

    cache_dirs = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dirs.append(os.path.join(hf_home, "hub"))

    home_dir = os.path.expanduser("~")
    cache_dirs.extend(
        [
            os.path.join(home_dir, ".cache", "huggingface", "hub"),
            "/home/owner/KIRAG/workspace/huggingface/hub",
            "/home/owner/IQRAG/.hf_cache/hub",
        ]
    )

    models_list = []
    visited_paths = set()
    total_size_bytes = 0
    seen_real_files = set()

    for raw_c_dir in cache_dirs:
        if not raw_c_dir:
            continue
        c_dir = os.path.realpath(raw_c_dir)
        if c_dir in visited_paths or not os.path.exists(c_dir) or not os.path.isdir(c_dir):
            continue
        visited_paths.add(c_dir)

        # Determine cache source label
        if "workspace" in c_dir:
            cache_source = "KIRAG Workspace"
        elif "IQRAG" in c_dir:
            cache_source = "IQRAG Cache"
        else:
            cache_source = "User HF Cache"

        try:
            for item in os.listdir(c_dir):
                if item.startswith("models--"):
                    model_path = os.path.join(c_dir, item)
                    if not os.path.isdir(model_path):
                        continue

                    parts = item[len("models--") :].split("--", 1)
                    if len(parts) == 2:
                        model_id = f"{parts[0]}/{parts[1]}"
                    else:
                        model_id = item[len("models--") :]

                    # Calculate model folder size on disk (avoid double counting symlinks)
                    size_bytes = 0
                    modified_timestamp = 0
                    has_real_blobs = False

                    try:
                        for root, _, files in os.walk(model_path):
                            for f in files:
                                fp = os.path.join(root, f)
                                try:
                                    real_fp = os.path.realpath(fp)
                                    st = os.stat(fp, follow_symlinks=False)
                                    file_size = st.st_size
                                    if not os.path.islink(fp):
                                        size_bytes += file_size
                                        if file_size > 1048576:  # > 1 MB weight blob
                                            has_real_blobs = True
                                    if real_fp not in seen_real_files:
                                        seen_real_files.add(real_fp)
                                        if not os.path.islink(fp):
                                            total_size_bytes += file_size

                                    if st.st_mtime > modified_timestamp:
                                        modified_timestamp = st.st_mtime
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning(f"Error calculating size for {model_path}: {e}")

                    # Determine context length
                    context_len = MODEL_MAX_CONTENT_LENGTHS.get(model_id)
                    if not context_len:
                        # Try reading from config.json in snapshot directory
                        try:
                            snapshots_dir = os.path.join(model_path, "snapshots")
                            if os.path.exists(snapshots_dir):
                                for s in os.listdir(snapshots_dir):
                                    cfg_path = os.path.join(snapshots_dir, s, "config.json")
                                    if os.path.exists(cfg_path):
                                        with open(cfg_path, encoding="utf-8") as cf:
                                            cfg_data = json.load(cf)
                                            context_len = (
                                                cfg_data.get("max_position_embeddings")
                                                or cfg_data.get("seq_length")
                                                or cfg_data.get("max_sequence_length")
                                            )
                                            if context_len:
                                                break
                        except Exception:
                            pass
                    if not context_len:
                        context_len = 131072 if "35B" in model_id or "70B" in model_id else 8192

                    # Determine model category/type
                    lower_id = model_id.lower()
                    if "reranker" in lower_id:
                        model_type = "Reranker"
                    elif any(kw in lower_id for kw in ["bge-", "embedding", "minilm"]):
                        model_type = "Embedding"
                    elif any(kw in lower_id for kw in ["olmocr", "vision", "-vl", "gemma-4"]):
                        model_type = "Vision LLM"
                    else:
                        model_type = "LLM"

                    is_stub = size_bytes < 1048576 and not has_real_blobs
                    mod_time_str = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(modified_timestamp))
                        if modified_timestamp > 0
                        else "N/A"
                    )

                    # Format human size with stub/empty status if incomplete
                    formatted_size = format_bytes_human(size_bytes)
                    if is_stub:
                        formatted_size = f"{formatted_size} (Stub/Empty)"

                    models_list.append(
                        {
                            "id": model_id,
                            "name": parts[1] if len(parts) == 2 else model_id,
                            "folder": item,
                            "path": model_path,
                            "cache_source": cache_source,
                            "size_bytes": size_bytes,
                            "human_size": formatted_size,
                            "context_length": int(context_len),
                            "model_type": model_type,
                            "is_active": False,  # Evaluated post-sort below for primary non-stub model
                            "is_stub": is_stub,
                            "modified_at": mod_time_str,
                        }
                    )
        except Exception as e:
            logger.error(f"Error scanning models in cache directory {c_dir}: {e}")

    # Sort models by size descending so complete models rank higher than stubs
    models_list.sort(key=lambda x: x["size_bytes"], reverse=True)

    # Assign is_active only to the primary non-stub installation for each active model ID
    active_marked = set()
    for m in models_list:
        if m["id"] in active_models and not m["is_stub"] and m["id"] not in active_marked:
            m["is_active"] = True
            active_marked.add(m["id"])

    return {
        "models": models_list,
        "total_count": len(models_list),
        "total_size_bytes": total_size_bytes,
        "total_human_size": format_bytes_human(total_size_bytes),
    }


def delete_installed_models(model_ids: list[str]) -> tuple[bool, str, list[str], int]:
    """
    Delete selected model directories from HuggingFace cache.
    Returns (success, message, list of deleted model IDs, reclaimed bytes).
    """
    import shutil

    if not model_ids:
        return False, "No models specified for deletion.", [], 0

    data = get_installed_models_data()

    deleted_models = []
    skipped_active = []
    reclaimed_bytes = 0

    for m_id in model_ids:
        target_info = None
        # Match by exact path first, then folder, ID, or partial name
        for m in data["models"]:
            if m["path"] == m_id or m["folder"] == m_id or m["id"] == m_id or m["name"] == m_id:
                target_info = m
                break

        if not target_info:
            continue

        model_key = target_info["id"]
        if target_info.get("is_active"):
            skipped_active.append(model_key)
            continue

        m_path = target_info["path"]
        if os.path.exists(m_path) or os.path.islink(m_path):
            try:
                if os.path.islink(m_path) or os.path.isfile(m_path):
                    os.unlink(m_path)
                elif os.path.isdir(m_path):
                    shutil.rmtree(m_path)
                deleted_models.append(model_key)
                reclaimed_bytes += target_info["size_bytes"]
                logger.info(f"Successfully deleted cached model path: {m_path}")
            except Exception as e:
                logger.error(f"Error removing model path {m_path}: {e}")

    reclaimed_str = format_bytes_human(reclaimed_bytes)
    msg_parts = []
    if deleted_models:
        msg_parts.append(
            f"Successfully deleted {len(deleted_models)} model(s) ({reclaimed_str} reclaimed)."
        )
    if skipped_active:
        msg_parts.append(
            f"Skipped {len(skipped_active)} active model(s) ({', '.join(skipped_active)})."
        )

    if not deleted_models and skipped_active:
        return False, " ".join(msg_parts), [], 0

    return (
        True,
        " ".join(msg_parts) if msg_parts else "No models were deleted.",
        deleted_models,
        reclaimed_bytes,
    )
