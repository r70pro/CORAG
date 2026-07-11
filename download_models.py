import os
import sys
import time
from huggingface_hub import snapshot_download

HF_TOKEN = "hf_REDACTED"
MODELS = [
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4"
]

print("Starting download of NVFP4 models...")
for model in MODELS:
    success = False
    for attempt in range(1, 6):
        print(f"Downloading {model} (Attempt {attempt}/5)...")
        try:
            path = snapshot_download(repo_id=model, token=HF_TOKEN, max_workers=4)
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
