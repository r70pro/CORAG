import os
import shutil

import process_state
from settings_manager import WORKSPACE_DIR


def get_dir_size(start_path):
    total_size = 0
    if not os.path.exists(start_path):
        return 0
    if os.path.isfile(start_path):
        return os.path.getsize(start_path)
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return total_size


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def perform_reset_cleanup(clean_runs, clean_gradio, clean_pycache, clean_hf, workspace_dir=None):
    freed_space = 0
    deleted_items = []
    errors = []

    if workspace_dir is None:
        workspace_dir = WORKSPACE_DIR

    # 1. Clean obsolete runs
    if clean_runs:
        if os.path.exists(workspace_dir):
            for name in os.listdir(workspace_dir):
                dir_path = os.path.join(workspace_dir, name)
                if os.path.isdir(dir_path) and name.startswith("run_"):
                    # Check if active
                    is_active = False
                    with process_state.active_runs_lock:
                        for run_info in process_state.active_runs.values():
                            proc = run_info.get("proc")
                            if run_info.get("run_dir") == dir_path:
                                if proc and proc.poll() is None:
                                    is_active = True
                                    break
                                if not run_info.get("completed", False):
                                    is_active = True
                                    break
                    if not is_active:
                        size = get_dir_size(dir_path)
                        try:
                            shutil.rmtree(dir_path)
                            freed_space += size
                            deleted_items.append(
                                f"Obsolete run directory: `{name}` ({format_size(size)})"
                            )
                        except Exception as e:
                            errors.append(f"Failed to delete run directory `{name}`: {e}")

    # 2. Clean gradio temp
    if clean_gradio:
        gradio_temp_dir = "/tmp/gradio"
        if os.path.exists(gradio_temp_dir):
            size = get_dir_size(gradio_temp_dir)
            try:
                for name in os.listdir(gradio_temp_dir):
                    item_path = os.path.join(gradio_temp_dir, name)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                freed_space += size
                deleted_items.append(f"Gradio upload temp files ({format_size(size)})")
            except Exception as e:
                errors.append(f"Failed to clean Gradio temp files: {e}")

    # 3. Clean python bytecode cache
    if clean_pycache:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, _ in list(os.walk(repo_dir)):
            for d in list(dirs):
                if d == "__pycache__":
                    pycache_path = os.path.join(root, d)
                    size = get_dir_size(pycache_path)
                    try:
                        shutil.rmtree(pycache_path)
                        dirs.remove(d)  # Don't recurse into deleted dir
                        freed_space += size
                        deleted_items.append(
                            f"Bytecode cache: `{os.path.relpath(pycache_path, repo_dir)}` ({format_size(size)})"
                        )
                    except Exception as e:
                        errors.append(f"Failed to delete `{pycache_path}`: {e}")

    # 4. Clean hugging face cache
    if clean_hf:
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache_dir):
            size = get_dir_size(hf_cache_dir)
            try:
                for name in os.listdir(hf_cache_dir):
                    item_path = os.path.join(hf_cache_dir, name)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                freed_space += size
                deleted_items.append(f"Hugging Face cache ({format_size(size)})")
            except Exception as e:
                errors.append(f"Failed to clean Hugging Face cache: {e}")

    # Prepare markdown summary
    if not deleted_items and not errors:
        return "### No files selected or found to clean up."

    summary = f"### 🧹 Cleanup Summary\n\n**Total space freed:** `{format_size(freed_space)}`\n\n"
    if deleted_items:
        summary += "**Successfully cleaned:**\n"
        for item in deleted_items:
            summary += f"- {item}\n"
    if errors:
        summary += "\n**Warnings / Errors:**\n"
        for err in errors:
            summary += f"- <span style='color: #fca5a5;'>{err}</span>\n"

    return summary
