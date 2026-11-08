"""Configuration loading for preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}
SPLITS = ("train", "val", "test")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    config_path = Path(path) if path is not None else root / "configs" / "preprocess.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {config_path}")
    config["_root"] = str(root)
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["_root"])
    keys = [
        "raw_dir",
        "data_dir",
        "processed_dir",
        "reports_dir",
        "figures_dir",
        "logs_dir",
        "subset_pieces",
        "subset_performances",
        "splits",
        "norm_stats",
    ]
    resolved: dict[str, Path] = {"root": root}
    for key in keys:
        resolved[key] = root / config[key]
    return resolved
