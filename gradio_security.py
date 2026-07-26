"""Gradio file-serving boundaries."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from path_security import PathSecurityError, resolve_run_under, resolve_under

_allowed_paths: list[str] = []
_allowed_paths_lock = Lock()


def register_gradio_input_dir(workspace_dir: str | Path, run_dir: str | Path) -> None:
    """Add one validated run input directory to the live Gradio allowlist."""

    workspace = Path(workspace_dir).resolve()
    candidate_run = Path(run_dir)
    safe_run = resolve_run_under(workspace, candidate_run.name)
    if candidate_run.resolve() != safe_run or not safe_run.is_dir():
        raise PathSecurityError("Invalid run path")
    inputs_dir = resolve_under(safe_run, "inputs")
    allowed_path = str(inputs_dir)
    with _allowed_paths_lock:
        if allowed_path not in _allowed_paths:
            _allowed_paths.append(allowed_path)


def get_gradio_path_config(
    workspace_dir: str | Path,
    project_root: str | Path,
    user_home: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return narrow allow paths and explicit sensitive-path blocks.

    Gradio's blocklist takes precedence over its allowlist, so blocking the
    entire home directory would also block a workspace located below it.
    The home directory is denied by omission from ``allowed_paths`` while its
    sensitive children are additionally placed on ``blocked_paths``.
    """

    workspace = Path(workspace_dir).resolve()
    project = Path(project_root).resolve()
    home = Path(user_home).resolve() if user_home else Path.home().resolve()

    upload_dir = (workspace / "uploads").resolve()
    export_dir = (workspace / "exports").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    with _allowed_paths_lock:
        _allowed_paths.clear()
        _allowed_paths.extend([str(upload_dir), str(export_dir)])
    if workspace.is_dir():
        for entry in workspace.iterdir():
            try:
                register_gradio_input_dir(workspace, entry)
            except PathSecurityError:
                continue
    blocked_candidates = [
        project / ".env",
        project / "settings.json",
        project / ".cache",
        workspace / "huggingface",
        workspace / "cache",
        workspace / "caches",
        workspace / "postgres",
        workspace / "redis",
        workspace / "minio",
        workspace / "qdrant",
        workspace / "volumes",
        home / ".env",
        home / ".ssh",
        home / ".config",
        home / ".local",
        home / ".cache",
    ]
    return _allowed_paths, [str(path.resolve()) for path in blocked_candidates]
