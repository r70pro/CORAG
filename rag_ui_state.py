import os  # noqa: F401
import datetime
from settings_manager import WORKSPACE_DIR, get_available_runs  # noqa: F401

import threading

RAG_LOG_BUFFER = []
RAG_LOG_LOCK = threading.Lock()
LAST_CREATED_RUN_ID = None

def log_to_rag(message: str):
    """Log a message to the RAG system log buffer with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Clean message of markdown formatting for console
    clean_msg = message.replace("**", "").replace("`", "").strip()
    if clean_msg:
        with RAG_LOG_LOCK:
            RAG_LOG_BUFFER.append(f"[{timestamp}] {clean_msg}")
            if len(RAG_LOG_BUFFER) > 500:
                RAG_LOG_BUFFER.pop(0)

def get_rag_logs() -> str:
    """Get all accumulated RAG logs as a single string."""
    with RAG_LOG_LOCK:
        return "\n".join(RAG_LOG_BUFFER)

def extract_text_content(content) -> str:
    """Extract plain text from potential Gradio 6 chatbot content format."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Gradio 6 format: {'text': "...", 'type': 'text'}
                if "text" in item:
                    text_parts.append(item["text"])
        return "".join(text_parts)
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
    return str(content)
