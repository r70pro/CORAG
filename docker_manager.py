import logging
import os
import socket
import subprocess
import time

import httpx

logger = logging.getLogger(__name__)



def get_docker_status():
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", "olmocr"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip().lower()
        elif (
            "no such object" in res.stderr.lower() or "no such inspect object" in res.stderr.lower()
        ):
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
    # Defensively coerce the port so an empty/non-numeric value (e.g. a cleared
    # Gradio Number widget) cannot raise TypeError in check_server_ready.
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 8000
    status = get_docker_status()
    if status == "not_found":
        return "not_found", "<span class='badge-idle'>Docker: Not Created</span>"
    elif status == "exited":
        return "stopped", "<span class='badge-stopped'>Docker: Stopped</span>"
    elif status in ("running", "restarting"):
        if check_server_ready(port):
            return "ready", "<span class='badge-success'>Inference Server: Ready</span>"
        else:
            return "starting", "<span class='badge-running'>Server: Starting / Loading Model</span>"
    else:
        return "error", "<span class='badge-failed'>Docker: Error</span>"


def get_docker_logs(tail: int = 200) -> str:
    """Fetch stdout/stderr logs from the vLLM docker container."""
    try:
        res = subprocess.run(
            ["docker", "logs", "--tail", str(int(tail)), "olmocr"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            output = res.stdout or ""
            if res.stderr:
                output = (output + "\n" + res.stderr) if output else res.stderr
            return output.strip() or "No log output available from container."
        elif (
            "no such object" in res.stderr.lower()
            or "no such container" in res.stderr.lower()
        ):
            return "Container 'olmocr' not found."
        else:
            return f"Error reading logs: {res.stderr.strip()}"
    except Exception as e:
        return f"Failed to retrieve container logs: {e}"


def is_vllm_compatible_model(model_id: str) -> bool:
    """Filter out embeddings, rerankers, and non-LLM models that cannot be served by vLLM."""
    lower = model_id.lower()
    excluded_keywords = ["bge-", "reranker", "minilm", "embedding", "docling-", "layout-heron"]
    for kw in excluded_keywords:
        if kw in lower:
            return False
    return True


def get_cached_models() -> list[str]:
    """Scan HuggingFace cache directories and standard presets for available models."""
    presets = [
        "allenai/olmOCR-2-7B-1025-FP8",
        "nvidia/Phi-4-reasoning-plus-NVFP4",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]
    models = list(presets)

    cache_dirs = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dirs.append(os.path.join(hf_home, "hub"))

    home_dir = os.path.expanduser("~")
    cache_dirs.append(os.path.join(home_dir, ".cache", "huggingface", "hub"))
    cache_dirs.append("/home/owner/.cache/huggingface/hub")
    cache_dirs.append("/home/owner/IQRAG/.hf_cache/hub")

    visited = set()
    for c_dir in cache_dirs:
        if not c_dir or c_dir in visited:
            continue
        visited.add(c_dir)
        if os.path.exists(c_dir) and os.path.isdir(c_dir):
            try:
                for item in os.listdir(c_dir):
                    if item.startswith("models--"):
                        parts = item[len("models--") :].split("--", 1)
                        if len(parts) == 2:
                            model_id = f"{parts[0]}/{parts[1]}"
                            if is_vllm_compatible_model(model_id) and model_id not in models:
                                models.append(model_id)
            except Exception as e:
                logger.warning(f"Error scanning cache dir {c_dir}: {e}")

    return models



def get_cached_models_info() -> tuple[list[str], dict[str, int]]:
    """Return available model names and their context window length mapping."""
    from settings_manager import MODEL_MAX_CONTENT_LENGTHS

    models = get_cached_models()
    max_lengths = dict(MODEL_MAX_CONTENT_LENGTHS)
    for m in models:
        if m not in max_lengths:
            max_lengths[m] = 131072
    return models, max_lengths




def start_docker_container():
    msg_parts = []
    success = True

    # 1. Start RAG Infrastructure containers (postgres, redis, minio, qdrant)
    try:
        from rag_infra_manager import start_rag_infrastructure
        rag_ok, rag_msg = start_rag_infrastructure()
        if not rag_ok:
            logger.warning(f"RAG Infrastructure start warning: {rag_msg}")
        msg_parts.append(f"RAG Infra: {rag_msg}")
    except Exception as e:
        logger.error(f"Error starting RAG infrastructure: {e}")
        msg_parts.append(f"RAG Infra error: {e}")

    # 2. Start or provision vLLM container ('olmocr')
    status = get_docker_status()
    if status == "exited":
        try:
            subprocess.run(["docker", "start", "olmocr"], check=True, capture_output=True)
            msg_parts.append("Container 'olmocr' started successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            msg_parts.append(f"Failed to start container 'olmocr': {e.stderr.decode().strip()}")
    elif status in ("running", "restarting"):
        msg_parts.append("Container 'olmocr' is already running.")
    elif status == "not_found":
        try:
            from settings_manager import load_settings
            settings = load_settings()
            hf_token = settings.get("hf_token", os.environ.get("HF_TOKEN", ""))
            port = settings.get("docker_port", 8000)
            model = settings.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
            gpu_mem = settings.get("docker_gpu_mem", 0.8)
            max_len = settings.get("docker_max_model_len", 15360)
            tp = settings.get("docker_tensor_parallel", 1)

            create_ok, create_msg = create_docker_container(hf_token, port, model, gpu_mem, max_len, tp)
            if not create_ok:
                success = False
            msg_parts.append(f"Provisioned 'olmocr': {create_msg}")
        except Exception as e:
            success = False
            msg_parts.append(f"Failed to provision 'olmocr' container: {e}")
    else:
        success = False
        msg_parts.append(f"Container status is '{status}', cannot start.")

    return success, " ".join(msg_parts)


def stop_docker_container():
    status = get_docker_status()
    msg_parts = []
    success = True
    if status in ["running", "restarting"]:
        try:
            subprocess.run(["docker", "stop", "olmocr"], check=True, capture_output=True)
            msg_parts.append("Container 'olmocr' stopped successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            msg_parts.append(f"Failed to stop container 'olmocr': {e.stderr.decode().strip()}")
    else:
        msg_parts.append("Container 'olmocr' is not running.")

    try:
        from rag_infra_manager import stop_rag_infrastructure
        rag_ok, rag_msg = stop_rag_infrastructure()
        if not rag_ok:
            success = False
        msg_parts.append(f"RAG Infra: {rag_msg}")
    except Exception as e:
        msg_parts.append(f"RAG Infra stop error: {e}")

    return success, " ".join(msg_parts)



def resolve_vllm_image() -> str:
    env_img = os.environ.get("OLMOCR_VLLM_IMAGE")
    if env_img:
        if os.environ.get("TESTING") == "true":
            return env_img
        try:
            res = subprocess.run(["docker", "inspect", env_img], capture_output=True, check=False)
            if res.returncode == 0:
                return env_img
            logger.warning(
                f"Configured OLMOCR_VLLM_IMAGE '{env_img}' not found locally via docker inspect. "
                "Falling back to available cached local image."
            )
        except Exception:
            pass
    for img in [
        "nvcr.io/nvidia/vllm:26.04-py3",
        "vllm/vllm-openai:v0.20.0",
    ]:
        try:
            res = subprocess.run(["docker", "inspect", img], capture_output=True, check=False)
            if res.returncode == 0:
                return img
        except Exception:
            pass
    return "vllm/vllm-openai:v0.20.0"


def free_host_port(port: int):
    """Remove any Docker containers bound to host port before starting a new container."""
    if os.environ.get("TESTING") == "true":
        return
    try:
        res = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if f":{port}->" in line or f":{port} " in line:
                    parts = line.split("\t")
                    cid = parts[0].strip()
                    if cid:
                        logger.info(f"Removing container {cid} ({parts[1] if len(parts)>1 else ''}) bound to port {port}")
                        subprocess.run(["docker", "rm", "-f", cid], check=False, capture_output=True)
    except Exception as e:
        logger.warning(f"Error checking/clearing containers on port {port}: {e}")




def wait_for_port_free(port: int, timeout: float = 5.0) -> bool:
    """Wait for host port to be released by Docker proxy or OS sockets."""
    if os.environ.get("TESTING") == "true":
        return True
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                res = s.connect_ex(("127.0.0.1", port))
                if res != 0:
                    return True
        except Exception:
            return True
        time.sleep(0.3)
    return False




def create_docker_container(hf_token, port, model, gpu_mem, max_model_len, tensor_parallel_size=1):
    status = get_docker_status()
    # Fall back to default model if model is invalid, empty, or literally "model"
    if not model or not str(model).strip() or str(model).strip() == "model":
        model = "allenai/olmOCR-2-7B-1025-FP8"
    else:
        model = str(model).strip()

    # Coerce the port to a sane integer default before any int() conversion
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        port_int = 8000

    # Coerce tensor_parallel_size to an integer >= 1
    try:
        tp_int = max(1, int(tensor_parallel_size))
    except (TypeError, ValueError):
        tp_int = 1

    # Coerce gpu_mem defensively
    try:
        gpu_mem_float = float(gpu_mem)
        if gpu_mem_float <= 0 or gpu_mem_float > 1.0:
            gpu_mem_float = 0.8
    except (TypeError, ValueError):
        gpu_mem_float = 0.8

    # Coerce max_model_len defensively
    try:
        max_model_len_int = int(max_model_len)
        if max_model_len_int <= 0:
            max_model_len_int = 15360
    except (TypeError, ValueError):
        max_model_len_int = 15360

    if status != "not_found" and status != "error":
        try:
            subprocess.run(["docker", "rm", "-f", "olmocr"], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err_msg = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else str(e.stderr or "")
            ).strip()
            return False, f"Failed to remove existing container: {err_msg}"

    # Clear any orphan docker containers listening on host port and wait for socket release
    free_host_port(port_int)
    wait_for_port_free(port_int, timeout=2.0)
    time.sleep(1.0)







    hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)

    # Resolve HF_TOKEN: filter out dummy/masked tokens ("********", "tok", empty strings)
    token_str = str(hf_token).strip() if hf_token else ""
    if not token_str or token_str in ("********", "tok"):
        token_str = os.environ.get("HF_TOKEN", "").strip()
    if not token_str or token_str in ("********", "tok"):
        dotenv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(dotenv_file):
            try:
                with open(dotenv_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("HF_TOKEN="):
                            val = line.strip().split("=", 1)[1].strip().strip("'\"")
                            if val and val not in ("********", "tok"):
                                token_str = val
                                os.environ["HF_TOKEN"] = val
                                break
            except Exception:
                pass

    # Pass secrets via a temporary env-file rather than `-e HF_TOKEN=...`.
    # Only write HF_TOKEN if non-empty and not masked.
    env_file = None
    try:
        import tempfile

        fd, env_path = tempfile.mkstemp(prefix="olmocr_env_", suffix=".env")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if token_str and token_str != "********":
                fh.write(f"HF_TOKEN={token_str}\n")
            fh.write("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")
            # Disable PyTorch AOT Autograd caching to prevent pickling launcher errors
            fh.write("TORCH_AOT_AUTOGRAD_CACHE=0\n")
            # Newer vLLM releases reject max_model_len values above the model's
            # derived maximum. Allow the app's configured value to override it.
            fh.write("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\n")
        env_file = env_path

        target_image = resolve_vllm_image()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            "olmocr",
            "--restart",
            "unless-stopped",
            "--gpus",
            "all",
            "--ipc=host",
            "-e",
            "NVIDIA_DISABLE_REQUIRE=1",
            "-p",
            f"127.0.0.1:{port_int}:8000",
            "-v",
            f"{hf_cache_dir}:/root/.cache/huggingface",
            "--env-file",
            env_path,
            "--entrypoint",
            "vllm",
            target_image,
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--enforce-eager",
            "--gpu-memory-utilization",
            f"{gpu_mem_float:.2f}",
            "--max-model-len",
            str(max_model_len_int),
            "--max-num-batched-tokens",
            str(max(max_model_len_int, 4096)),
            "--tensor-parallel-size",
            str(tp_int),
            "--trust-remote-code",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, "Container created and started successfully."
        except subprocess.CalledProcessError as e:
            err_msg = str(e.stderr or "").strip()
            if "port is already allocated" in err_msg.lower() or "address already in use" in err_msg.lower():
                logger.info("Port conflict detected. Waiting 1.5s for Docker proxy socket release and retrying...")
                time.sleep(1.5)
                subprocess.run(["docker", "rm", "-f", "olmocr"], check=False, capture_output=True)
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    return True, "Container created and started successfully (retry)."
                except subprocess.CalledProcessError as retry_e:
                    err_msg = str(retry_e.stderr or "").strip()
            return False, f"Failed to create container: {err_msg}"
        finally:

            # Always remove the temp env-file (best-effort; ignore cleanup errors).
            try:
                os.remove(env_path)
            except OSError:
                pass
    finally:
        # Defensive cleanup in case an exception escaped the inner try/finally.
        if env_file is not None and os.path.exists(env_file):
            try:
                os.remove(env_path)
            except OSError:
                pass


def shutdown_docker_container():
    status = get_docker_status()
    msg_parts = []
    success = True
    if status in ["running", "exited"]:
        try:
            if status == "running":
                subprocess.run(["docker", "stop", "olmocr"], check=True, capture_output=True)
            subprocess.run(["docker", "rm", "olmocr"], check=True, capture_output=True)
            msg_parts.append("Container 'olmocr' shutdown successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            msg_parts.append(f"Failed to shutdown container 'olmocr': {e.stderr.decode().strip()}")
    else:
        msg_parts.append("Container 'olmocr' is not running.")

    try:
        from rag_infra_manager import stop_rag_infrastructure
        rag_ok, rag_msg = stop_rag_infrastructure()
        if not rag_ok:
            success = False
        msg_parts.append(f"RAG Infra: {rag_msg}")
    except Exception as e:
        msg_parts.append(f"RAG Infra shutdown error: {e}")

    return success, " ".join(msg_parts)


def cleanup_docker():
    if os.environ.get("TESTING") == "true":
        return
    # Allow operators to keep the GPU container running across app restarts
    # (e.g. in a persistent deployment) by setting KEEP_CONTAINERS_ON_EXIT.
    if os.environ.get("KEEP_CONTAINERS_ON_EXIT") == "true":
        return
    logger.info(
        "Application shutting down. Stopping local OLMOCR Docker container & RAG infra to release resources..."
    )
    try:
        subprocess.run(["docker", "stop", "olmocr"], capture_output=True)
        logger.info("Docker container 'olmocr' stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping container on shutdown: {e}")

    try:
        from rag_infra_manager import stop_rag_infrastructure
        stop_rag_infrastructure()
        logger.info("RAG infrastructure stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping RAG infrastructure on shutdown: {e}")

