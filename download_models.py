import logging
import os
import socket
import time

logger = logging.getLogger(__name__)

# Force IPv4 DNS resolution to prevent hangs on broken IPv6 routes to Hugging Face
if getattr(socket.getaddrinfo, "__name__", None) != "_ipv4_getaddrinfo":
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(*args, **kwargs):
        responses = _orig_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]

    socket.getaddrinfo = _ipv4_getaddrinfo

# Set timeout for Hugging Face Hub operations to prevent infinite hangs
if "HF_HUB_DOWNLOAD_TIMEOUT" not in os.environ:
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download  # noqa: E402

# Load Hugging Face token dynamically from environment or settings.json
try:
    from settings_manager import load_settings

    settings = load_settings()
except ImportError:
    settings = {}

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = settings.get("hf_token")

if not HF_TOKEN:
    logger.warning(
        "HF_TOKEN environment variable not set, and hf_token not configured in settings.json. "
        "Proceeding without token (some gated/NVFP4 models may fail to download)..."
    )

MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "openai/gpt-oss-120b",
    "google/gemma-4-31B-it",
]


def download_all_models(models=MODELS, token=HF_TOKEN):
    logger.info("Starting download of NVFP4 models...")
    actual_token = token if token else None
    for model in models:
        success = False
        for attempt in range(1, 6):
            logger.info(f"Downloading {model} (Attempt {attempt}/5)...")
            try:
                path = snapshot_download(repo_id=model, token=actual_token, max_workers=4)
                logger.info(f"Successfully downloaded {model} to {path}")
                success = True
                break
            except Exception as e:
                logger.error(f"Error downloading {model} on attempt {attempt}: {e}")
                if attempt < 5:
                    logger.info("Waiting 10 seconds before retrying...")
                    time.sleep(10)
                else:
                    logger.exception(f"Failed attempt {attempt} for {model}")
        if not success:
            logger.error(f"Failed to download {model} after 5 attempts.")

    logger.info("All downloads complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    download_all_models()
