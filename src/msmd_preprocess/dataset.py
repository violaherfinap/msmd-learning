"""Dataset discovery and subset/split helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import long_path


def list_dirs(path: str | Path) -> list[str]:
    if not os.path.isdir(long_path(path)):
        return []
    with os.scandir(long_path(path)) as entries:
        return sorted(entry.name for entry in entries if entry.is_dir())


def score_dirs(raw_dir: str | Path, piece: str) -> list[Path]:
    root = Path(raw_dir) / piece / "scores"
    return [root / name for name in list_dirs(root)]


def has_mung(raw_dir: str | Path, piece: str) -> bool:
    for score_dir in score_dirs(raw_dir, piece):
        mung_dir = score_dir / "mung"
        if mung_dir.is_dir() and list(mung_dir.glob("*.xml")):
            return True
    return False


def page_items(raw_dir: str | Path, piece: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_num = 0
    for score_dir in score_dirs(raw_dir, piece):
        img_dir = score_dir / "img"
        pngs = sorted(img_dir.glob("*.png")) if img_dir.is_dir() else []
        mung_dir = score_dir / "mung"
        mungs = sorted(mung_dir.glob("*.xml")) if mung_dir.is_dir() else []
        mung_by_stem = {path.stem: path for path in mungs}
        for index, png in enumerate(pngs):
            page_num += 1
            mung = mung_by_stem.get(png.stem)
            if mung is None and len(mungs) == len(pngs):
                mung = mungs[index]
            items.append(
                {
                    "page_num": page_num,
                    "page_stem": png.stem,
                    "png": png,
                    "mung": mung,
                    "score_dir": score_dir,
                }
            )
    return items


def load_metadata(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(config["_root"])
    pieces = pd.read_csv(root / config["subset_pieces"])
    perfs = pd.read_csv(root / config["subset_performances"])
    splits = pd.read_csv(root / config["splits"])

    split_map = dict(zip(splits["piece"], splits["split"]))
    pieces["split"] = pieces["piece"].map(split_map).fillna(pieces["split"])
    perfs = perfs[perfs["piece"].isin(set(pieces["piece"]))].copy()
    perfs["split"] = perfs["piece"].map(split_map).fillna(perfs["split"])
    return pieces, perfs, splits


def exclude_no_mung(
    pieces: pd.DataFrame,
    perfs: pd.DataFrame,
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    excluded = [piece for piece in pieces["piece"] if not has_mung(raw_dir, piece)]
    excluded_set = set(excluded)
    return (
        pieces[~pieces["piece"].isin(excluded_set)].copy(),
        perfs[~perfs["piece"].isin(excluded_set)].copy(),
        excluded,
    )


def choose_perfs(perfs_list: list[dict[str, Any]], split: str, k: int) -> list[dict[str, Any]]:
    if split != "train":
        return list(perfs_list)
    ranked = sorted(
        perfs_list,
        key=lambda row: (
            abs((float(row["tempo"]) if pd.notna(row["tempo"]) else 1000.0) - 1000.0),
            str(row["soundfont"]),
            str(row["perf_name"]),
        ),
    )
    chosen: list[dict[str, Any]] = []
    seen_soundfonts: set[str] = set()
    for row in ranked:
        soundfont = str(row["soundfont"])
        if soundfont not in seen_soundfonts:
            chosen.append(row)
            seen_soundfonts.add(soundfont)
            if len(chosen) >= k:
                break
    for row in ranked:
        if len(chosen) >= k:
            break
        if all(row["perf_name"] != item["perf_name"] for item in chosen):
            chosen.append(row)
    return chosen[:k]


def features_dir(raw_dir: str | Path, piece: str, perf_name: str) -> Path:
    return Path(raw_dir) / piece / "performances" / perf_name / "features"


def performance_table(perfs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["perf_name"]): row for row in perfs.to_dict("records")}
