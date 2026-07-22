import os
import re
import socket
import subprocess
import time
from typing import Any

import requests


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

            # Check for CUDA graphs capture
            for line in reversed(lines):
                if "Capturing CUDA graphs" in line:
                    match = re.search(r"Capturing CUDA graphs.*:\s*(\d+)%\s*\|\s*.*?\|\s*(\d+)/(\d+)", line)
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
    service_history: dict[str, list[float]], vllm_port: int = 8000
) -> dict[str, Any]:
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
                            if "..." in line:
                                vram_used_proc *= 10
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

        vram_pct = (vram_used / vram_total) * 100.0

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
