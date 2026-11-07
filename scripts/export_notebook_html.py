#!/usr/bin/env python3
"""Export an executed notebook to HTML and remove local absolute paths.

This is a convenience script for reviewers who want the raw notebook output
without cloning the Windows-specific directory structure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks" / "02_note_level_preprocessing.ipynb"
OUT_DIR = PROJECT / "reports"
OUT_HTML = OUT_DIR / "02_note_level_preprocessing.html"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "html",
            str(NOTEBOOK),
            "--output-dir",
            str(OUT_DIR),
            "--output",
            OUT_HTML.stem,
        ],
        check=True,
    )

    text = OUT_HTML.read_text(encoding="utf-8")
    for old in {str(PROJECT), PROJECT.as_posix(), str(PROJECT).replace("\\", "/")}:
        text = text.replace(old, ".")
    OUT_HTML.write_text(text, encoding="utf-8")
    print("Saved:", OUT_HTML)
    print("Sanitized local project path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
