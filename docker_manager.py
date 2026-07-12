import os
import subprocess
import httpx
from state import get_fn

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
    except Exception:
        return "error"

def check_server_ready(port):
    try:
        response = httpx.get(f"http://localhost:{port}/v1/models", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False

def get_docker_status_str(port):
    status = get_fn('get_docker_status', get_docker_status)()
    if status == "not_found":
        return "not_found", "<span class='badge-idle'>Docker: Not Created</span>"
    elif status == "exited":
        return "stopped", "<span class='badge-stopped'>Docker: Stopped</span>"
    elif status == "running":
        if get_fn('check_server_ready', check_server_ready)(port):
            return "ready", "<span class='badge-success'>Inference Server: Ready</span>"
        else:
            return "starting", "<span class='badge-running'>Server: Starting / Loading Model</span>"
    else:
        return "error", "<span class='badge-failed'>Docker: Error</span>"

def start_docker_container():
    status = get_fn('get_docker_status', get_docker_status)()
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
    status = get_fn('get_docker_status', get_docker_status)()
    if status == "running":
        try:
            subprocess.run(["docker", "stop", "olmocr"], check=True, capture_output=True)
            return True, "Container stopped successfully."
        except subprocess.CalledProcessError as e:
            return False, f"Failed to stop container: {e.stderr.decode().strip()}"
    return True, "Container is not running."

def create_docker_container(hf_token, port, model, gpu_mem, max_model_len):
    status = get_fn('get_docker_status', get_docker_status)()
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
        "-p", f"127.0.0.1:{int(port)}:8000",
        "-v", f"{hf_cache_dir}:/root/.cache/huggingface",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "vllm/vllm-openai:cu130-nightly",
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
    if os.environ.get("TESTING") == "true":
        return
    print("Application shutting down. Stopping local OLMOCR Docker container to release VRAM...")
    try:
        subprocess.run(["docker", "stop", "olmocr"], capture_output=True)
        print("Docker container stopped successfully.")
    except Exception as e:
        print(f"Error stopping container on shutdown: {e}")
