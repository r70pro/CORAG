import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


# Wrap stdout/stderr to prevent crash with [Errno 5] Input/output error
# when running detached from a terminal.
class SafeStream:
    def __init__(self, original):
        self.original = original

    def write(self, data):
        if self.original:
            try:
                self.original.write(data)
            except OSError as e:
                if e.errno == 5:  # EIO
                    pass
                else:
                    raise
            except Exception:
                pass

    def flush(self):
        if self.original:
            try:
                self.original.flush()
            except OSError as e:
                if e.errno == 5:
                    pass
                else:
                    raise
            except Exception:
                pass

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self.original, name)


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)


# Load .env file manually if it exists
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(dotenv_path):
    try:
        with open(dotenv_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and val and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")


# Redirect Hugging Face cache to writeable workspace directory.
# If the workspace path is not writable (e.g. root-owned by Docker),
# fall back to a writable default cache location.
def _resolve_hf_home():
    workspace_hf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "workspace", "huggingface"
    )
    parent = os.path.dirname(workspace_hf)
    if os.access(parent, os.W_OK) or (
        os.path.isdir(workspace_hf) and os.access(workspace_hf, os.W_OK)
    ):
        return workspace_hf
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _resolve_hf_home()

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


# Use the bundled workspace directory when it is writable; otherwise fall back
# to a user-owned workspace so the app can still write runs, exports, and the
# Hugging Face cache (e.g. when Docker has created ./workspace as root).
def _resolve_workspace_dir():
    default_ws = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
    if os.access(default_ws, os.W_OK) or (
        os.path.isdir(default_ws) and os.access(default_ws, os.W_OK)
    ):
        return default_ws
    fallback = os.path.join(os.path.expanduser("~"), ".local", "share", "kirag", "workspace")
    os.makedirs(fallback, exist_ok=True)
    return fallback


WORKSPACE_DIR = _resolve_workspace_dir()

# Single source of truth for the application version. The Gradio UI sidebar
# (app.py) and the rag package (__init__.py) both import this so the displayed
# version can never drift apart.
VERSION = "2.0.3"

SUPPORTED_MODELS = [
    "allenai/olmOCR-2-7B-1025-FP8",
    "Qwen/Qwen3.6-35B-A3B",
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "openai/gpt-oss-120b",
    "google/gemma-4-31B-it",
]

MODEL_MAX_CONTENT_LENGTHS = {
    "allenai/olmOCR-2-7B-1025-FP8": 131072,
    "Qwen/Qwen3.6-35B-A3B": 262144,
    "nvidia/Qwen3.6-35B-A3B-NVFP4": 262144,
    "nvidia/Phi-4-reasoning-plus-NVFP4": 32768,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 1048576,
    "nvidia/Llama-3.3-70B-Instruct-NVFP4": 131072,
    "openai/gpt-oss-120b": 131072,
    "google/gemma-4-31B-it": 262144,
}


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
        "hf_token": os.environ.get("HF_TOKEN", ""),
        # RAG Analysis settings
        "analysis_model_name": "nvidia/Phi-4-reasoning-plus-NVFP4",
        "analysis_server_url": "http://localhost:8000/v1",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_device": "auto",
        "embedding_batch_size": 64,
        "chunk_size": 800,
        "chunk_overlap": 100,
        "retrieval_top_k": 15,
        "rag_auto_start_infra": False,
        "use_reranker": True,
        "reranker_model": "BAAI/bge-reranker-large",
        "reranker_device": "cuda",
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                user_settings = json.load(f)
                # Avoid overwriting hf_token with an empty string if env has a token
                if (
                    "hf_token" in user_settings
                    and not user_settings["hf_token"]
                    and defaults.get("hf_token")
                ):
                    user_settings.pop("hf_token")
                defaults.update(user_settings)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    return defaults


def save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {e}"


def get_available_runs(workspace_dir: str | None = None):
    """Scan workspace for completed OCR runs that have markdown output.

    Args:
        workspace_dir: Optional override path for workspace directory. Defaults to WORKSPACE_DIR.

    Returns:
        List of (display_name, run_dir_path) tuples for the dropdown.
    """
    runs = []
    import sys

    sm_mod = sys.modules.get("settings_manager")
    primary_ws = workspace_dir or getattr(sm_mod, "WORKSPACE_DIR", WORKSPACE_DIR)

    candidate_dirs = []
    if primary_ws:
        candidate_dirs.append(primary_ws)

    seen_names = set()

    for ws in candidate_dirs:
        if not os.path.exists(ws):
            continue
        try:
            dir_names = sorted(os.listdir(ws), reverse=True)
        except Exception:
            continue
        for name in dir_names:
            if name in seen_names or not name.startswith("run_"):
                continue
            run_dir = os.path.join(ws, name)
            if not os.path.isdir(run_dir):
                continue
            md_dir = os.path.join(run_dir, "markdown", "inputs")
            if os.path.exists(md_dir):
                md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
                if md_files:
                    seen_names.add(name)
                    display = f"{name} ({len(md_files)} file{'s' if len(md_files) != 1 else ''})"
                    runs.append((display, run_dir))

    # Sort runs by name descending
    runs.sort(key=lambda r: os.path.basename(r[1]), reverse=True)
    return runs

