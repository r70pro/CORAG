import hashlib
import logging
import os
import re
import socket
import subprocess
import time

import httpx

from audit_log import audit_event

logger = logging.getLogger(__name__)

CONTAINER_NAME = "kirag_vllm"
MANAGED_LABEL = "com.kirag.managed"
MANAGED_LABEL_VALUE = "true"
DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
DEFAULT_VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d"
)
_DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
_DOCKER_NETWORK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _decode_stderr(error: subprocess.CalledProcessError) -> str:
    if isinstance(error.stderr, bytes):
        return error.stderr.decode("utf-8", errors="replace").strip()
    return str(error.stderr or "").strip()


def get_docker_status(container_name: str = CONTAINER_NAME):
    try:
        res = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                f'{{{{.State.Status}}}}\t{{{{index .Config.Labels "{MANAGED_LABEL}"}}}}',
                container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            parts = res.stdout.strip().lower().split("\t", 1)
            if len(parts) != 2 or parts[1] != MANAGED_LABEL_VALUE:
                return "foreign"
            return parts[0]
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


def get_docker_restart_count(container_name: str = CONTAINER_NAME) -> int:
    """Return the managed container's restart count, or zero when unavailable."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.RestartCount}}", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return max(0, int(result.stdout.strip()))
    except (OSError, ValueError):
        pass
    return 0


def get_docker_status_str(port, container_name: str = CONTAINER_NAME):
    # Defensively coerce the port so an empty/non-numeric value (e.g. a cleared
    # Gradio Number widget) cannot raise TypeError in check_server_ready.
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 8000
    status = get_docker_status(container_name)
    if status == "not_found":
        return "not_found", "<span class='badge-idle'>Docker: Not Created</span>"
    elif status == "foreign":
        return "foreign", "<span class='badge-failed'>Docker: Foreign Container</span>"
    elif status == "exited":
        return "stopped", "<span class='badge-stopped'>Docker: Stopped</span>"
    elif status == "restarting":
        return "error", "<span class='badge-failed'>Inference Server: Crash Loop</span>"
    elif status == "running":
        if check_server_ready(port):
            return "ready", "<span class='badge-success'>Inference Server: Ready</span>"
        elif get_docker_restart_count(container_name) > 0:
            return "error", "<span class='badge-failed'>Inference Server: Startup Failed</span>"
        else:
            return "starting", "<span class='badge-running'>Server: Starting / Loading Model</span>"
    else:
        return "error", "<span class='badge-failed'>Docker: Error</span>"


def get_docker_logs(tail: int = 200, container_name: str = CONTAINER_NAME) -> str:
    """Fetch stdout/stderr logs from the vLLM docker container."""
    try:
        res = subprocess.run(
            ["docker", "logs", "--tail", str(int(tail)), container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            output = res.stdout or ""
            if res.stderr:
                output = (output + "\n" + res.stderr) if output else res.stderr
            return output.strip() or "No log output available from container."
        elif "no such object" in res.stderr.lower() or "no such container" in res.stderr.lower():
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
    """Return allowlisted models, plus compatible cached models under admin override."""
    from settings_manager import SUPPORTED_MODELS

    models = list(SUPPORTED_MODELS)

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
                            if (
                                _env_enabled("KIRAG_ADVANCED_MODEL_OVERRIDE")
                                and is_vllm_compatible_model(model_id)
                                and model_id not in models
                            ):
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
    if os.environ.get("TESTING") != "true":
        from settings_manager import load_settings
        from vllm_lifecycle import switch_vllm

        settings = load_settings()
        role = {"analysis_262k": "analysis", "ocr_only": "ocr"}.get(
            settings.get("startup_mode", "analysis"), settings.get("startup_mode", "analysis")
        )
        if role not in {"ocr", "analysis"}:
            return False, "No inference role is selected."
        try:
            switch_vllm(role, settings.get("analysis_model_name") if role == "analysis" else None)
            return True, f"{role.upper()} inference is ready."
        except Exception as exc:
            return False, str(exc)
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
            subprocess.run(["docker", "start", CONTAINER_NAME], check=True, capture_output=True)
            audit_event("container_start", "success", container=CONTAINER_NAME)
            msg_parts.append("Container 'olmocr' started successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            audit_event(
                "container_start",
                "failure",
                container=CONTAINER_NAME,
                error=_decode_stderr(e),
            )
            msg_parts.append(f"Failed to start container 'olmocr': {_decode_stderr(e)}")
    elif status in ("running", "restarting"):
        msg_parts.append("Container 'olmocr' is already running.")
    elif status == "not_found":
        try:
            from settings_manager import load_settings

            settings = load_settings()
            hf_token = settings.get("hf_token", os.environ.get("HF_TOKEN", ""))
            port = settings.get("docker_port", 8000)
            model = settings.get("model_name", DEFAULT_MODEL)
            gpu_mem = settings.get("docker_gpu_mem", 0.8)
            max_len = settings.get("docker_max_model_len", 15360)
            tp = settings.get("docker_tensor_parallel", 1)

            create_ok, create_msg = create_docker_container(
                hf_token, port, model, gpu_mem, max_len, tp
            )
            if not create_ok:
                success = False
            msg_parts.append(f"Provisioned 'olmocr': {create_msg}")
        except Exception as e:
            success = False
            msg_parts.append(f"Failed to provision 'olmocr' container: {e}")
    elif status == "foreign":
        success = False
        msg_parts.append(
            "Container name 'olmocr' is occupied by an unmanaged container; "
            "explicit operator action is required."
        )
    else:
        success = False
        msg_parts.append(f"Container status is '{status}', cannot start.")

    return success, " ".join(msg_parts)


def stop_docker_container():
    """Stop only the managed vLLM container.

    RAG databases have an independent lifecycle and must remain available when
    inference is deliberately restarted or reconfigured.
    """
    if os.environ.get("TESTING") != "true":
        try:
            from vllm_lifecycle import stop_vllm

            stop_vllm()
            return True, "Inference slot stopped."
        except Exception as exc:
            return False, str(exc)
    status = get_docker_status()
    msg_parts = []
    success = True
    if status in ["running", "restarting"]:
        try:
            subprocess.run(["docker", "stop", CONTAINER_NAME], check=True, capture_output=True)
            audit_event("container_stop", "success", container=CONTAINER_NAME)
            msg_parts.append("Container 'olmocr' stopped successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            audit_event(
                "container_stop",
                "failure",
                container=CONTAINER_NAME,
                error=_decode_stderr(e),
            )
            msg_parts.append(f"Failed to stop container 'olmocr': {_decode_stderr(e)}")
    elif status == "foreign":
        success = False
        msg_parts.append(
            "Refusing to stop unmanaged container named 'olmocr'; "
            "explicit operator action is required."
        )
    else:
        msg_parts.append("Container 'olmocr' is not running.")

    return success, " ".join(msg_parts)


def set_vllm_role_running(role: str, running: bool) -> tuple[bool, str]:
    """Compatibility wrapper around the exclusive single-slot lifecycle."""
    try:
        from settings_manager import load_settings
        from vllm_lifecycle import read_state, stop_vllm, switch_vllm

        if running:
            model = load_settings().get("analysis_model_name") if role == "analysis" else None
            switch_vllm(role, model)
            return True, f"{role.upper()} vLLM is ready in the exclusive inference slot."
        if read_state().get("active_role") != role:
            return True, f"{role.upper()} vLLM is already inactive."
        stop_vllm()
        return True, "Inference slot stopped."
    except Exception as exc:
        return False, f"Unable to change vLLM role: {exc}"


def set_extended_analysis_context(enabled: bool) -> tuple[bool, str]:
    """Deprecated compatibility shim; simultaneous dual-model mode was removed."""
    if not enabled:
        return False, "Dual-model mode is unavailable; select either OCR or analysis."
    return set_vllm_role_running("analysis", True)


def resolve_vllm_image() -> str:
    image = os.environ.get("OLMOCR_VLLM_IMAGE", "").strip() or DEFAULT_VLLM_IMAGE
    if not _DIGEST_IMAGE_RE.fullmatch(image):
        raise ValueError(
            "OLMOCR_VLLM_IMAGE must be pinned by an immutable sha256 digest "
            "(repository@sha256:<64 hex characters>)."
        )
    return image


def find_port_occupant(port: int) -> tuple[str, str] | None:
    """Return the ID/name of a container publishing ``port`` without mutating it."""
    if os.environ.get("TESTING") == "true":
        return None
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
                        return cid, parts[1].strip() if len(parts) > 1 else ""
    except Exception as e:
        logger.warning(f"Error checking containers on port {port}: {e}")
    return None


def free_host_port(port: int):
    """Compatibility wrapper retained as a read-only port occupancy check."""
    return find_port_occupant(port)


def _port_conflict_message(port: int, occupant: tuple[str, str] | None) -> str:
    if occupant:
        cid, name = occupant
        identity = f"'{name}' ({cid})" if name else cid
        return (
            f"Port {port} is already published by container {identity}. "
            "No container was removed; explicit operator action is required."
        )
    return (
        f"Port {port} is already in use by a non-Docker or unidentified process. "
        "No process was stopped; explicit operator action is required."
    )


def _validate_model(model: str) -> tuple[bool, str]:
    from settings_manager import SUPPORTED_MODELS

    if model in SUPPORTED_MODELS:
        return True, ""
    if _env_enabled("KIRAG_ADVANCED_MODEL_OVERRIDE"):
        return True, ""
    return (
        False,
        f"Model '{model}' is not in the configured allowlist. "
        "An administrator must explicitly enable KIRAG_ADVANCED_MODEL_OVERRIDE "
        "to run an unlisted model.",
    )


def _remote_code_config() -> tuple[bool, str, str]:
    enabled = _env_enabled("KIRAG_ENABLE_REMOTE_CODE")
    if not enabled:
        return False, "", ""
    network = os.environ.get("KIRAG_REMOTE_CODE_NETWORK", "").strip()
    if not _DOCKER_NETWORK_RE.fullmatch(network) or network in {
        "host",
        "bridge",
        "default",
        "none",
    }:
        raise ValueError(
            "Remote code requires KIRAG_REMOTE_CODE_NETWORK to name a dedicated, "
            "operator-controlled Docker network (built-in and namespace-sharing "
            "network modes are refused)."
        )
    token = os.environ.get("KIRAG_REMOTE_CODE_HF_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "Remote code requires a dedicated scoped, short-lived credential in "
            "KIRAG_REMOTE_CODE_HF_TOKEN."
        )
    return True, network, token


def _validate_remote_network(network: str) -> None:
    """Require an explicitly labeled internal network for remote-code workloads."""
    try:
        result = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "--format",
                '{{index .Labels "com.kirag.remote-code"}}\t{{.Internal}}',
                network,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        raise ValueError(f"Remote-code network '{network}' could not be inspected: {e}") from e
    if result.returncode != 0:
        raise ValueError(
            f"Remote-code network '{network}' was not found or could not be inspected."
        )
    parts = result.stdout.strip().lower().split("\t", 1)
    if parts != ["true", "true"]:
        raise ValueError(
            f"Remote-code network '{network}' must be internal and labeled "
            "com.kirag.remote-code=true."
        )


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
    if os.environ.get("TESTING") != "true":
        if model != DEFAULT_MODEL or int(port) != 8000:
            return (
                False,
                "Only the pinned OCR profile on port 8000 may use this compatibility operation.",
            )
        try:
            from vllm_lifecycle import switch_vllm

            switch_vllm("ocr")
            return True, "OCR inference is ready in the exclusive slot."
        except Exception as exc:
            return False, str(exc)
    status = get_docker_status()
    # Fall back to default model if model is invalid, empty, or literally "model"
    if not model or not str(model).strip() or str(model).strip() == "model":
        model = DEFAULT_MODEL
    else:
        model = str(model).strip()

    model_ok, model_error = _validate_model(model)
    if not model_ok:
        audit_event("container_create", "denied", model=model, reason="model_not_allowlisted")
        return False, model_error

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

    # Validate all administrator-controlled security settings before mutating
    # an existing managed container.
    try:
        remote_code, remote_network, remote_token = _remote_code_config()
        target_image = resolve_vllm_image()
        if remote_code:
            _validate_remote_network(remote_network)
    except ValueError as e:
        audit_event("container_create", "denied", model=model, port=port_int, reason=str(e))
        return False, str(e)

    hf_cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    try:
        os.makedirs(hf_cache_dir, exist_ok=True)
    except OSError as e:
        audit_event(
            "container_create",
            "failure",
            model=model,
            port=port_int,
            reason="cache_unavailable",
            error=str(e),
        )
        return False, f"Failed to prepare the Hugging Face cache: {e}"

    if status == "foreign":
        audit_event(
            "container_create",
            "denied",
            container=CONTAINER_NAME,
            model=model,
            reason="foreign_name_conflict",
        )
        return (
            False,
            "Container name 'olmocr' is occupied by an unmanaged container. "
            "No container was removed; explicit operator action is required.",
        )

    occupant = find_port_occupant(port_int)
    if occupant and occupant[1] != CONTAINER_NAME:
        audit_event(
            "container_create",
            "denied",
            model=model,
            port=port_int,
            occupying_container_id=occupant[0],
            occupying_container_name=occupant[1],
            reason="port_conflict",
        )
        return False, _port_conflict_message(port_int, occupant)

    if status not in {"not_found", "error"}:
        try:
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=True, capture_output=True)
            audit_event(
                "container_delete",
                "success",
                container=CONTAINER_NAME,
                reason="recreate",
            )
        except subprocess.CalledProcessError as e:
            err_msg = _decode_stderr(e)
            audit_event(
                "container_delete",
                "failure",
                container=CONTAINER_NAME,
                reason="recreate",
                error=err_msg,
            )
            return False, f"Failed to remove existing container: {err_msg}"

    if not wait_for_port_free(port_int, timeout=2.0):
        occupant = find_port_occupant(port_int)
        audit_event(
            "container_create",
            "denied",
            model=model,
            port=port_int,
            reason="port_conflict",
            occupying_container_id=occupant[0] if occupant else "",
            occupying_container_name=occupant[1] if occupant else "",
        )
        return False, _port_conflict_message(port_int, occupant)

    # Resolve HF_TOKEN: filter out dummy/masked tokens ("********", "tok", empty strings)
    token_str = remote_token if remote_code else (str(hf_token).strip() if hf_token else "")
    if not remote_code and (not token_str or token_str in ("********", "tok")):
        token_str = os.environ.get("HF_TOKEN", "").strip()
    if not remote_code and (not token_str or token_str in ("********", "tok")):
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
    credential_path = None
    try:
        import tempfile

        fd, env_path = tempfile.mkstemp(prefix="olmocr_env_", suffix=".env")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if not remote_code and token_str and token_str != "********":
                fh.write(f"HF_TOKEN={token_str}\n")
            fh.write("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")
            # Disable PyTorch AOT Autograd caching to prevent pickling launcher errors
            fh.write("TORCH_AOT_AUTOGRAD_CACHE=0\n")
            # Newer vLLM releases reject max_model_len values above the model's
            # derived maximum. Allow the app's configured value to override it.
            fh.write("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1\n")
        env_file = env_path
        if remote_code:
            credential_fd, credential_path = tempfile.mkstemp(
                prefix="olmocr_hf_credential_", suffix=".token"
            )
            os.fchmod(credential_fd, 0o600)
            with os.fdopen(credential_fd, "w", encoding="utf-8") as credential_file:
                credential_file.write(remote_token)

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "--label",
            f"{MANAGED_LABEL}={MANAGED_LABEL_VALUE}",
            "--label",
            "com.kirag.component=vllm",
            "--restart",
            "unless-stopped",
            "--init",
            "--stop-timeout",
            os.environ.get("KIRAG_VLLM_STOP_TIMEOUT", "120").removesuffix("s"),
            "--health-cmd",
            "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)\"",
            "--health-interval",
            "15s",
            "--health-timeout",
            "6s",
            "--health-start-period",
            os.environ.get("KIRAG_VLLM_HEALTH_START_PERIOD", "30m"),
            "--health-retries",
            "5",
            "--log-opt",
            "max-size=20m",
            "--log-opt",
            "max-file=5",
            "--gpus",
            "all",
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
        ]
        if "qwen3" in model.lower():
            # Parse Qwen3 reasoning separately even though KIRAG analysis
            # requests disable thinking. This keeps the OpenAI-compatible API
            # correct for direct/advanced callers that explicitly enable it.
            cmd.extend(["--reasoning-parser", "qwen3"])
        if remote_code:
            model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
            safe_model_name = (
                re.sub(r"[^A-Za-z0-9_.-]+", "--", model).strip("-")[:100] + "-" + model_hash
            )
            remote_cache_dir = os.path.join(hf_cache_dir, "kirag-remote-code", safe_model_name)
            os.makedirs(remote_cache_dir, exist_ok=True)
            cache_mount_index = cmd.index("-v")
            cmd[cache_mount_index + 1] = f"{remote_cache_dir}:/model-cache"
            cmd[cache_mount_index:cache_mount_index] = [
                "--mount",
                f"type=bind,src={credential_path},dst=/run/secrets/kirag_hf_token,readonly",
            ]
            env_file_index = cmd.index("--env-file")
            cmd[env_file_index:env_file_index] = [
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--network",
                remote_network,
                "--shm-size",
                os.environ.get("KIRAG_REMOTE_CODE_SHM_SIZE", "8g"),
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=2g",
                "-e",
                "HF_HOME=/model-cache",
                "-e",
                "HF_TOKEN_PATH=/run/secrets/kirag_hf_token",
            ]
            restart_index = cmd.index("--restart")
            cmd[restart_index + 1] = "no"
            cmd.append("--trust-remote-code")
        else:
            # Dedicated shared memory avoids sharing the host IPC namespace.
            env_file_index = cmd.index("--env-file")
            cmd[env_file_index:env_file_index] = ["--shm-size", "8g"]

        audit_event(
            "container_create",
            "attempt",
            container=CONTAINER_NAME,
            image=target_image,
            model=model,
            port=port_int,
            remote_code=remote_code,
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            audit_event(
                "container_create",
                "success",
                container=CONTAINER_NAME,
                image=target_image,
                model=model,
                port=port_int,
                remote_code=remote_code,
            )
            return True, "Container created and started successfully."
        except subprocess.CalledProcessError as e:
            err_msg = str(e.stderr or "").strip()
            if (
                "port is already allocated" in err_msg.lower()
                or "address already in use" in err_msg.lower()
            ):
                occupant = find_port_occupant(port_int)
                audit_event(
                    "container_create",
                    "denied",
                    container=CONTAINER_NAME,
                    model=model,
                    port=port_int,
                    reason="port_conflict",
                    occupying_container_id=occupant[0] if occupant else "",
                    occupying_container_name=occupant[1] if occupant else "",
                )
                return False, _port_conflict_message(port_int, occupant)
            audit_event(
                "container_create",
                "failure",
                container=CONTAINER_NAME,
                image=target_image,
                model=model,
                port=port_int,
                remote_code=remote_code,
                error=err_msg,
            )
            return False, f"Failed to create container: {err_msg}"
        finally:
            # Always remove the temp env-file (best-effort; ignore cleanup errors).
            try:
                os.remove(env_path)
            except OSError:
                pass
            if credential_path:
                try:
                    os.remove(credential_path)
                except OSError:
                    pass
    finally:
        # Defensive cleanup in case an exception escaped the inner try/finally.
        if env_file is not None and os.path.exists(env_file):
            try:
                os.remove(env_path)
            except OSError:
                pass
        if credential_path is not None and os.path.exists(credential_path):
            try:
                os.remove(credential_path)
            except OSError:
                pass


def shutdown_docker_container():
    if os.environ.get("TESTING") != "true":
        try:
            from vllm_lifecycle import stop_vllm

            stop_vllm()
            return True, "Inference slot stopped and removed."
        except Exception as exc:
            return False, str(exc)
    status = get_docker_status()
    audit_event("container_shutdown", "attempt", container=CONTAINER_NAME, status=status)
    msg_parts = []
    success = True
    if status in ["running", "exited"]:
        try:
            if status == "running":
                subprocess.run(["docker", "stop", CONTAINER_NAME], check=True, capture_output=True)
            subprocess.run(["docker", "rm", CONTAINER_NAME], check=True, capture_output=True)
            audit_event("container_delete", "success", container=CONTAINER_NAME, reason="shutdown")
            audit_event("container_shutdown", "success", container=CONTAINER_NAME)
            msg_parts.append("Container 'olmocr' shutdown successfully.")
        except subprocess.CalledProcessError as e:
            success = False
            audit_event(
                "container_shutdown",
                "failure",
                container=CONTAINER_NAME,
                error=_decode_stderr(e),
            )
            msg_parts.append(f"Failed to shutdown container 'olmocr': {_decode_stderr(e)}")
    elif status == "foreign":
        success = False
        audit_event(
            "container_shutdown",
            "denied",
            container=CONTAINER_NAME,
            reason="unmanaged_container",
        )
        msg_parts.append(
            "Refusing to stop or remove unmanaged container named 'olmocr'; "
            "explicit operator action is required."
        )
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
        audit_event("docker_cleanup", "skipped", reason="testing")
        return
    # Persistent services are never owned by an application process.  This
    # legacy cleanup entry point is fail-closed and requires an explicit opt-in
    # for old workstation integrations that call it directly.
    if os.environ.get("KIRAG_ALLOW_APP_INFRA_SHUTDOWN") != "true":
        audit_event("docker_cleanup", "skipped", reason="persistent_infrastructure")
        return
    status = get_docker_status()
    audit_event("docker_cleanup", "attempt", container=CONTAINER_NAME, status=status)
    logger.info(
        "Application shutting down. Stopping local OLMOCR Docker container & RAG infra to release resources..."
    )
    try:
        if status in {"running", "restarting"}:
            res = subprocess.run(
                ["docker", "stop", CONTAINER_NAME], capture_output=True, text=True, check=False
            )
            if res.returncode == 0:
                audit_event("docker_cleanup", "success", container=CONTAINER_NAME)
                logger.info("Docker container 'olmocr' stopped successfully.")
            else:
                audit_event(
                    "docker_cleanup",
                    "failure",
                    container=CONTAINER_NAME,
                    error=res.stderr.strip(),
                )
        elif status == "foreign":
            audit_event(
                "docker_cleanup",
                "denied",
                container=CONTAINER_NAME,
                reason="unmanaged_container",
            )
            logger.warning("Refusing to stop unmanaged container named 'olmocr' during cleanup.")
    except Exception as e:
        audit_event("docker_cleanup", "failure", container=CONTAINER_NAME, error=str(e))
        logger.error(f"Error stopping container on shutdown: {e}")

    try:
        from rag_infra_manager import stop_rag_infrastructure

        stop_rag_infrastructure()
        logger.info("RAG infrastructure stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping RAG infrastructure on shutdown: {e}")
