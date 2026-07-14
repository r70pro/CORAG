import time
import requests
import socket
import subprocess
import re
import os

def get_service_latency(service_name, host="127.0.0.1", port=None):
    ports = {
        "postgres": 5432,
        "redis": 6379,
        "minio": 9000,
        "qdrant": 6333,
        "vllm": 8000
    }
    if port is None:
        port = ports.get(service_name)
        
    start_time = time.time()
    extra_info = None
    try:
        if service_name == "postgres":
            import psycopg2
            conn = psycopg2.connect(
                dbname="olmocr_rag",
                user="olmocr",
                password="pg_pass_5c6d3284f18b90a6e2d8",
                host=host,
                port=port,
                connect_timeout=1
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

def get_vllm_loading_progress():
    try:
        res = subprocess.run(
            ["docker", "logs", "--tail", "50", "olmocr"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout:
            lines = res.stdout.splitlines()
            for line in reversed(lines):
                if "Loading safetensors checkpoint shards:" in line:
                    match = re.search(
                        r"Loading safetensors checkpoint shards:\s*(\d+)%\s*Completed\s*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]",
                        line
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
                            "eta": eta
                        }
    except Exception:
        pass
    return None

def check_backing_services_data(service_history, vllm_port=8000):
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
            "latency_history": list(service_history.get(s, []))
        }
        
    vllm_progress = None
    if not services_data["vllm"]["is_up"]:
        vllm_progress = get_vllm_loading_progress()
        
    return {
        "all_healthy": all_healthy,
        "services": services_data,
        "failed_services": failed_services,
        "vllm_model": vllm_model,
        "vllm_progress": vllm_progress
    }

def get_gpu_metrics_data():
    cuda_available = False
    gpu_name = "N/A"
    vram_used = 0.0
    vram_total = 0.0
    vram_pct = 0.0
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            vram_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
    except Exception:
        pass
        
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            cuda_available = True
            lines = res.stdout.strip().split("\n")
            parts = lines[0].split(",")
            gpu_name = parts[0].strip()
            vram_used = float(parts[1].strip())
            vram_total = float(parts[2].strip())
    except Exception:
        pass
        
    if cuda_available and vram_total > 0:
        vram_pct = (vram_used / vram_total) * 100.0
        
        def get_docker_containers():
            containers = {}
            try:
                res = subprocess.run(
                    ["docker", "ps", "-a", "--format", "{{.ID}}|{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0:
                    for line in res.stdout.strip().split("\n"):
                        if "|" in line:
                            cid, name = line.split("|", 1)
                            containers[cid] = name
            except Exception:
                pass
            return containers

        def resolve_process_details(pid, name_from_smi):
            cmdline = ""
            is_docker = False
            container_name = ""
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    raw = f.read()
                    parts = [p for p in raw.split("\x00") if p]
                    if parts:
                        cmdline = " ".join(parts)
            except Exception:
                pass
            if not cmdline:
                cmdline = name_from_smi
            try:
                with open(f"/proc/{pid}/cgroup", "r") as f:
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

        def get_display_name(cmdline, default_name):
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

        processes = []
        try:
            res_proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=3
            )
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
                        match = re.search(r'\|\s+(\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+([CG]|[C\+G])\s+(.+?)\s+(\d+)\s*MiB\s*\|', line)
                        if match:
                            gpu_id = int(match.group(1))
                            pid = int(match.group(2))
                            proc_type = match.group(3)
                            proc_name = match.group(4).strip()
                            vram_used_proc = int(match.group(5))
                            processes.append({
                                "gpu_id": gpu_id,
                                "pid": pid,
                                "type": proc_type,
                                "name": proc_name,
                                "vram": vram_used_proc
                            })
        except Exception:
            pass

        docker_map = get_docker_containers()
        essential_keywords = ["xorg", "gnome-shell", "gdm", "kwin", "wayland", "systemd", "lightdm"]
        
        resolved_procs = []
        vram_reclaimable = 0.0
        
        for p in processes:
            cmdline, is_docker, container_id = resolve_process_details(p["pid"], p["name"])
            container_name = docker_map.get(container_id, container_id) if container_id else ""
            
            is_essential = any(k in cmdline.lower() or k in p["name"].lower() for k in essential_keywords)
            reclaimable = not is_essential
            
            display_name = get_display_name(cmdline, p["name"])
            
            if is_docker:
                type_text = f"Docker: {container_name}" if container_name else "Docker"
                type_badge_style = "background: rgba(245, 158, 11, 0.15); color: #f59e0b;"
                action_text = f"Stop container '{container_name}'" if container_name else "Stop container"
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
                
            resolved_procs.append({
                "display_name": display_name,
                "cmdline": cmdline,
                "pid": p["pid"],
                "vram": p["vram"],
                "type_text": type_text,
                "type_badge_style": type_badge_style,
                "action_text": action_text,
                "action_color": action_color,
                "is_essential": is_essential
            })
            
        vram_free = max(0.0, vram_total - vram_used)
        vram_potential_free = vram_free + vram_reclaimable
        
        return {
            "cuda_available": True,
            "gpu_name": gpu_name,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "vram_pct": vram_pct,
            "vram_free": vram_free,
            "vram_reclaimable": vram_reclaimable,
            "vram_potential_free": vram_potential_free,
            "processes": resolved_procs
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
            "processes": []
        }
