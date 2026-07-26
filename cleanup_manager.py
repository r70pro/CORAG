import os
import shutil
from pathlib import Path

import process_state
from audit_log import audit_event
from path_security import PathSecurityError, resolve_run_under, resolve_under
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
    audit_event(
        "application_cleanup",
        "attempt",
        clean_runs=bool(clean_runs),
        clean_gradio=bool(clean_gradio),
        clean_pycache=bool(clean_pycache),
        clean_hf=bool(clean_hf),
    )
    freed_space = 0
    deleted_items = []
    errors = []

    if workspace_dir is None:
        workspace_dir = WORKSPACE_DIR

    # 1. Clean obsolete runs
    if clean_runs:
        workspace = Path(workspace_dir).resolve()
        if workspace.is_dir():
            for entry in workspace.iterdir():
                try:
                    dir_path = resolve_run_under(workspace, entry.name)
                except PathSecurityError:
                    continue
                if dir_path.is_dir():
                    # Check if active
                    is_active = False
                    with process_state.active_runs_lock:
                        for run_info in process_state.active_runs.values():
                            proc = run_info.get("proc")
                            if Path(run_info.get("run_dir", "")).resolve() == dir_path:
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
                                f"Obsolete run directory: `{entry.name}` ({format_size(size)})"
                            )
                        except Exception:
                            errors.append(f"Failed to delete run directory `{entry.name}`")

    # 2. Clean gradio temp
    if clean_gradio:
        gradio_temp_dir = Path("/tmp/gradio").resolve()
        if gradio_temp_dir.is_dir():
            size = get_dir_size(gradio_temp_dir)
            try:
                for entry in gradio_temp_dir.iterdir():
                    try:
                        item_path = resolve_under(gradio_temp_dir, entry.name)
                    except PathSecurityError:
                        continue
                    if item_path.is_dir():
                        shutil.rmtree(item_path)
                    else:
                        item_path.unlink()
                freed_space += size
                deleted_items.append(f"Gradio upload temp files ({format_size(size)})")
            except Exception:
                errors.append("Failed to clean Gradio temp files")

    # 3. Clean python bytecode cache
    if clean_pycache:
        repo_dir = Path(__file__).resolve().parent
        for root, dirs, _ in list(os.walk(repo_dir)):
            for d in list(dirs):
                if d == "__pycache__":
                    try:
                        pycache_path = resolve_under(root, d)
                    except PathSecurityError:
                        continue
                    if not pycache_path.is_relative_to(repo_dir):
                        continue
                    size = get_dir_size(pycache_path)
                    try:
                        shutil.rmtree(pycache_path)
                        dirs.remove(d)  # Don't recurse into deleted dir
                        freed_space += size
                        deleted_items.append(
                            f"Bytecode cache: `{pycache_path.relative_to(repo_dir)}` ({format_size(size)})"
                        )
                    except Exception:
                        errors.append("Failed to delete a bytecode cache")

    # 4. Clean hugging face cache
    if clean_hf:
        hf_cache_dir = (Path.home() / ".cache" / "huggingface").resolve()
        if hf_cache_dir.is_dir():
            size = get_dir_size(hf_cache_dir)
            try:
                for entry in hf_cache_dir.iterdir():
                    try:
                        item_path = resolve_under(hf_cache_dir, entry.name)
                    except PathSecurityError:
                        continue
                    if item_path.is_dir():
                        shutil.rmtree(item_path)
                    else:
                        item_path.unlink()
                freed_space += size
                deleted_items.append(f"Hugging Face cache ({format_size(size)})")
            except Exception:
                errors.append("Failed to clean Hugging Face cache")

    # Prepare markdown summary
    if not deleted_items and not errors:
        audit_event("application_cleanup", "no_change", freed_bytes=0)
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

    audit_event(
        "application_cleanup",
        "partial_failure" if errors else "success",
        freed_bytes=freed_space,
        deleted_item_count=len(deleted_items),
        error_count=len(errors),
    )
    return summary
