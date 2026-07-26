"""Shared filesystem boundary checks for KIRAG-managed files."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a requested path does not stay inside an approved directory."""


_RUN_NAME_RE = re.compile(r"^run_[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def resolve_under(base: str | Path, *parts: str | Path) -> Path:
    """Resolve path components and require the target to remain below ``base``.

    Each part is a single path component.  Requiring components rather than
    arbitrary relative paths prevents alternate separator and filename
    confusion before ``Path.resolve()`` performs the symlink-aware boundary
    check.
    """

    base_path = Path(base).resolve()
    clean_parts: list[str] = []
    for raw_part in parts:
        part = str(raw_part)
        if (
            not part
            or "\x00" in part
            or Path(part).is_absolute()
            or "/" in part
            or "\\" in part
            or part in {".", ".."}
        ):
            raise PathSecurityError("Invalid path")
        clean_parts.append(part)

    target = base_path.joinpath(*clean_parts).resolve()
    if target == base_path or not target.is_relative_to(base_path):
        raise PathSecurityError("Invalid path")
    return target


def validate_run_name(run_name: str) -> str:
    """Return a valid workspace run directory name."""

    if "\x00" in run_name or not _RUN_NAME_RE.fullmatch(run_name):
        raise PathSecurityError("Invalid run name")
    return run_name


def validate_filename(filename: str, allowed_extensions: Iterable[str]) -> str:
    """Return a safe single-component filename with an approved extension."""

    if (
        not filename
        or "\x00" in filename
        or Path(filename).is_absolute()
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise PathSecurityError("Invalid filename")

    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in allowed_extensions
    }
    if Path(filename).suffix.lower() not in allowed:
        raise PathSecurityError("Invalid file type")
    return filename


def resolve_file_under(base: str | Path, filename: str, allowed_extensions: Iterable[str]) -> Path:
    """Validate a filename and resolve it beneath an approved directory."""

    return resolve_under(base, validate_filename(filename, allowed_extensions))


def resolve_run_under(workspace: str | Path, run_name: str) -> Path:
    """Validate and resolve a run directory beneath the workspace."""

    return resolve_under(workspace, validate_run_name(run_name))


def require_approved_file(
    path: str | Path,
    approved_bases: Iterable[str | Path],
    allowed_extensions: Iterable[str],
) -> Path:
    """Resolve an internal file path against one of a fixed set of bases."""

    raw_path = Path(path)
    if "\x00" in str(raw_path):
        raise PathSecurityError("Invalid path")
    resolved = raw_path.resolve()
    validate_filename(resolved.name, allowed_extensions)

    for base in approved_bases:
        base_path = Path(base).resolve()
        if resolved.is_relative_to(base_path):
            relative_parts = resolved.relative_to(base_path).parts
            if not relative_parts:
                break
            return resolve_under(base_path, *relative_parts)
    raise PathSecurityError("Invalid path")
