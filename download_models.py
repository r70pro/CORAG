import os
import time
from huggingface_hub import snapshot_download

# Load Hugging Face token dynamically from environment or settings.json
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from settings_manager import load_settings
        settings = load_settings()
        HF_TOKEN = settings.get("hf_token")
    except ImportError:
        pass

if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not set, and hf_token not configured in settings.json.")
    print("Proceeding without token (some gated/NVFP4 models may fail to download)...")

MODELS = [
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4"
]

def download_all_models(models=MODELS, token=HF_TOKEN):
    print("Starting download of NVFP4 models...")
    for model in models:
        success = False
        for attempt in range(1, 6):
            print(f"Downloading {model} (Attempt {attempt}/5)...")
            try:
                path = snapshot_download(repo_id=model, token=token, max_workers=4)
                print(f"Successfully downloaded {model} to {path}")
                success = True
                break
            except Exception as e:
                print(f"Error downloading {model} on attempt {attempt}: {e}")
                if attempt < 5:
                    print("Waiting 10 seconds before retrying...")
                    time.sleep(10)
                else:
                    import traceback
                    traceback.print_exc()
        if not success:
            print(f"Failed to download {model} after 5 attempts.")

    print("All downloads complete.")


if __name__ == "__main__":
    download_all_models()

