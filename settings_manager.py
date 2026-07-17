import os
import json

# Load .env file manually if it exists
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(dotenv_path):
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
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
        print(f"Error loading .env file: {e}")

# Redirect Hugging Face cache to writeable workspace directory
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "huggingface")

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

SUPPORTED_MODELS = [
    "allenai/olmOCR-2-7B-1025-FP8",
    "Qwen/Qwen3.6-35B-A3B",
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "openai/gpt-oss-120b",
    "google/gemma-4-31B-it"
]

MODEL_MAX_CONTENT_LENGTHS = {
    "allenai/olmOCR-2-7B-1025-FP8": 131072,
    "Qwen/Qwen3.6-35B-A3B": 262144,
    "nvidia/Qwen3.6-35B-A3B-NVFP4": 262144,
    "nvidia/Phi-4-reasoning-plus-NVFP4": 32768,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 1048576,
    "nvidia/Llama-3.3-70B-Instruct-NVFP4": 131072,
    "openai/gpt-oss-120b": 131072,
    "google/gemma-4-31B-it": 262144
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
        "embedding_device": "cpu",
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


def get_available_runs():
    """Scan workspace for completed OCR runs that have markdown output.

    Returns:
        List of (display_name, run_dir_path) tuples for the dropdown.
    """
    runs = []
    workspace = WORKSPACE_DIR
    if not os.path.exists(workspace):
        return runs

    for name in sorted(os.listdir(workspace), reverse=True):
        run_dir = os.path.join(workspace, name)
        if not os.path.isdir(run_dir) or not name.startswith("run_"):
            continue
        md_dir = os.path.join(run_dir, "markdown", "inputs")
        if os.path.exists(md_dir):
            md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
            if md_files:
                # Format: "run_20260711_092213 (1 file, 9 pages)"
                display = f"{name} ({len(md_files)} file{'s' if len(md_files) != 1 else ''})"
                runs.append((display, run_dir))

    return runs

