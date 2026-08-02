import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

from path_security import (
    PathSecurityError,
    resolve_file_under,
    resolve_run_under,
    validate_run_name,
)

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

_DEFAULT_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "settings.json"
)
SETTINGS_FILE = os.path.abspath(
    os.path.expanduser(os.environ.get("KIRAG_SETTINGS_FILE", _DEFAULT_SETTINGS_FILE))
)
_SETTINGS_LOCK = threading.RLock()


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
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "openai/gpt-oss-120b",
    "google/gemma-4-31B-it",
]

MODEL_MAX_CONTENT_LENGTHS = {
    "allenai/olmOCR-2-7B-1025-FP8": 131072,
    "Qwen/Qwen3.6-35B-A3B": 262144,
    "nvidia/Phi-4-reasoning-plus-NVFP4": 32768,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 1048576,
    "nvidia/Llama-3.3-70B-Instruct-NVFP4": 131072,
    "openai/gpt-oss-120b": 131072,
    "google/gemma-4-31B-it": 262144,
}


def load_settings(*, include_env_secrets: bool = True):
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
        "docker_tensor_parallel": 1,
        "hf_token": os.environ.get("HF_TOKEN", "") if include_env_secrets else "",
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
        # Model profile restored by the supervised infrastructure service.
        # Analysis-only avoids loading OCR unless ingestion is requested.
        "startup_mode": "analysis_262k",
        "use_reranker": True,
        "reranker_model": "BAAI/bge-reranker-large",
        "reranker_device": "cuda",
    }
    with _SETTINGS_LOCK:
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
    # Sync analysis_model_name if empty or missing
    if not defaults.get("analysis_model_name"):
        defaults["analysis_model_name"] = defaults.get("model_name", "allenai/olmOCR-2-7B-1025-FP8")
    # Production can pin OCR and analysis to separate, continuously available
    # inference services. Environment values intentionally override UI state.
    env_overrides = {
        "server_url": "KIRAG_OCR_SERVER_URL",
        "model_name": "KIRAG_OCR_MODEL",
        "analysis_server_url": "KIRAG_ANALYSIS_SERVER_URL",
        "analysis_model_name": "KIRAG_ANALYSIS_MODEL",
    }
    if os.environ.get("TESTING") != "true":
        for setting_key, environment_key in env_overrides.items():
            if os.environ.get(environment_key, "").strip():
                defaults[setting_key] = os.environ[environment_key].strip()
        # A successfully smoke-tested UI/CLI switch is newer than deployment
        # defaults and intentionally overrides only the analysis profile.
        try:
            from analysis_profiles import read_runtime_profile
            runtime_profile = read_runtime_profile()
            if runtime_profile:
                defaults["analysis_model_name"] = runtime_profile["model"]
                defaults["analysis_server_url"] = "http://127.0.0.1:8002/v1"
        except Exception as exc:
            logger.warning("Unable to read runtime analysis profile: %s", exc)
    return defaults


def save_settings(settings):
    temporary_path = None
    try:
        settings_dir = os.path.dirname(SETTINGS_FILE)
        os.makedirs(settings_dir, exist_ok=True)
        with _SETTINGS_LOCK:
            # Ensure analysis_model_name is set if missing
            if "model_name" in settings and "analysis_model_name" not in settings:
                settings["analysis_model_name"] = settings["model_name"]
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=settings_dir,
                prefix=".settings.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(settings, temporary_file, indent=2)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, SETTINGS_FILE)
            temporary_path = None
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {e}"
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def delete_run_directory(run_id_or_path: str) -> bool:
    """Delete a matching run directory from the configured workspace only."""
    if (
        not run_id_or_path
        or "\x00" in run_id_or_path
        or Path(run_id_or_path).is_absolute()
        or "/" in run_id_or_path
        or "\\" in run_id_or_path
    ):
        return False
    import hashlib
    import shutil

    deleted_any = False
    workspace = Path(WORKSPACE_DIR).resolve()
    if not workspace.is_dir():
        return False
    try:
        entries = list(workspace.iterdir())
    except OSError:
        return False

    requested_name = run_id_or_path
    for entry in entries:
        try:
            validate_run_name(entry.name)
            target = resolve_run_under(workspace, entry.name)
        except PathSecurityError:
            continue
        if not target.is_dir():
            continue
        item_id = hashlib.sha256(str(target).encode()).hexdigest()[:16]
        if entry.name == requested_name or item_id == run_id_or_path:
            shutil.rmtree(target)
            deleted_any = True
    return deleted_any


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

    # Safely import DB index checker
    try:
        from rag.db import is_run_indexed
    except Exception:
        is_run_indexed = None

    for ws in candidate_dirs:
        workspace = Path(ws).resolve()
        if not workspace.is_dir():
            continue
        try:
            dir_names = sorted((entry.name for entry in workspace.iterdir()), reverse=True)
        except OSError:
            continue
        for name in dir_names:
            if name in seen_names:
                continue
            try:
                run_dir = resolve_run_under(workspace, name)
            except PathSecurityError:
                continue
            if not run_dir.is_dir():
                continue
            md_dir = run_dir / "markdown" / "inputs"
            if md_dir.is_dir():
                md_files = []
                for entry in md_dir.iterdir():
                    try:
                        candidate = resolve_file_under(md_dir, entry.name, {".md"})
                    except PathSecurityError:
                        continue
                    if candidate.is_file():
                        md_files.append(entry.name)
                if md_files:
                    seen_names.add(name)
                    is_indexed = False
                    if is_run_indexed is not None:
                        try:
                            import hashlib

                            hashed_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
                            is_indexed = (
                                is_run_indexed(name, check_vector_store=False)
                                or is_run_indexed(hashed_id, check_vector_store=False)
                                or is_run_indexed(str(run_dir), check_vector_store=False)
                            )
                        except Exception:
                            is_indexed = False

                    badge = "✅ " if is_indexed else "📄 "
                    suffix = " [INDEXED]" if is_indexed else ""
                    display = f"{badge}{name} ({len(md_files)} file{'s' if len(md_files) != 1 else ''}){suffix}"
                    runs.append((display, str(run_dir)))

    # Sort runs by name descending
    runs.sort(key=lambda r: os.path.basename(r[1]), reverse=True)
    return runs
