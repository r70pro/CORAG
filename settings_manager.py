import os
import json

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

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
        "analysis_model_name": "microsoft/Phi-4-reasoning-plus",
        "analysis_server_url": "http://localhost:8000/v1",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_device": "cpu",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "retrieval_top_k": 8,
        "rag_auto_start_infra": False,
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
