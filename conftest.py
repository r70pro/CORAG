import os

# Redirect Hugging Face cache to the workspace directory when it is writable.
# settings_manager already resolves to a writable fallback when this is not set,
# so only force the workspace path here if it can actually be written to.
_ws_hf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "huggingface")
_ws_parent = os.path.dirname(_ws_hf)
if "HF_HOME" not in os.environ and (
    os.access(_ws_parent, os.W_OK) or (os.path.isdir(_ws_hf) and os.access(_ws_hf, os.W_OK))
):
    os.environ["HF_HOME"] = _ws_hf

import settings_manager  # noqa: E402, F401
