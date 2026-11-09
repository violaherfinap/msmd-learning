"""Path helpers that are safe on Windows long paths."""

from __future__ import annotations

import os
from pathlib import Path


def long_path(path: str | Path) -> str:
    """Return a Windows long-path-safe path when needed."""
    value = os.path.abspath(path)
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def relative_posix(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
