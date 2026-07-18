import os

# Redirect Hugging Face cache to writeable workspace directory before any test imports run
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "huggingface")

import settings_manager  # noqa: F401

