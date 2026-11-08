"""Task 1 preprocessing: audio window -> score system localization."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from PIL import Image

from .config import SPLIT_OFFSET, SPLITS
from .io import NpyAppendWriter, write_json, write_parquet
from .mung import load_gray, load_systems
from .paths import long_path, relative_posix

LOGGER = logging.getLogger(__name__)


def resize_with_pad(image: Image.Image, target_w: int, target_h: int, fill: int = 255) -> np.ndarray:
    image = image.convert("L")
    width, height = image.size
    if width == 0 or height == 0:
        return np.full((target_h, target_w), fill, dtype=np.uint8)
    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (target_w, target_h), fill)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 1]
    y = polygon[:, 0]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def system_crop(
    page_gray: np.ndarray,
    bbox: tuple[float, float, float, float],
    margin: int,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    height, width = page_gray.shape
    x0, y0, x1, y1 = bbox
    left = max(0, int(np.floor(x0)) - margin)
    top = max(0, int(np.floor(y0)) - margin)
    right = min(width, int(np.ceil(x1)) + margin)
    bottom = min(height, int(np.ceil(y1)) + margin)
    if right <= left or bottom <= top:
        return np.full((target_h, target_w), 255, dtype=np.uint8)
    crop = page_gray[top:bottom, left:right]
    return resize_with_pad(Image.fromarray(crop), target_w, target_h)


def choose_candidates(
    rng: np.random.Generator,
    pool: list[dict[str, Any]],
    count: int,
    chosen: set[str],
) -> list[dict[str, Any]]:
    available = [item for item in pool if item["system_id"] not in chosen]
    if not available or count <= 0:
        return []
    order = rng.permutation(len(available))[:count]
    return [available[int(index)] for index in order]


def build_split(config: dict[str, Any], split: str) -> dict[str, Any]:
    root = Path(config["_root"])
    proc_dir = root / config["processed_dir"]
    page_size = int(config["task1"]["page_size"])
    system_w = int(config["task1"]["system_crop_w"])
    system_h = int(config["task1"]["system_crop_h"])
    margin = int(config["task1"]["system_margin"])
    negatives_per_positive = int(config["task1"]["negatives_per_positive"])

    noteheads = pd.read_parquet(proc_dir / f"noteheads_{split}.parquet")
    windows = pd.read_parquet(proc_dir / f"windows_{split}.parquet")

    page_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    page_systems: dict[str, list[dict[str, Any]]] = defaultdict(list)

    page_writer = NpyAppendWriter(
        proc_dir / f"score_pages_{split}.npy",
        dtype=np.uint8,
        item_shape=(page_size, page_size, 1),
    )
    system_writer = NpyAppendWriter(
        proc_dir / f"system_crops_{split}.npy",
        dtype=np.uint8,
        item_shape=(system_h, system_w, 1),
    )

    page_sum = page_sumsq = 0.0
    page_count = 0
    system_sum = system_sumsq = 0.0
    system_count = 0

    pages = (
        noteheads[["page_id", "piece", "work_group", "split", "page_num", "png_path"]]
        .drop_duplicates("page_id")
        .sort_values(["piece", "page_num"])
        .to_dict("records")
    )

    for page in pages:
        page_id = str(page["page_id"])
        png_path = root / str(page["png_path"])
        page_stem = png_path.stem
        score_dir = png_path.parent.parent
        systems = load_systems(score_dir, page_stem)
        page_gray = load_gray(png_path)
        height, width = page_gray.shape

        page_array = resize_with_pad(Image.fromarray(page_gray), page_size, page_size)
        page_row = page_writer.append(page_array[..., None])
        if split == "train":
            scaled = page_array.astype(np.float64) / 255.0
            page_sum += float(scaled.sum())
            page_sumsq += float((scaled * scaled).sum())
            page_count += int(scaled.size)

        y_centers = systems.mean(axis=1)[:, 0] if len(systems) else np.zeros((0,), dtype=float)
        for index, polygon in enumerate(systems):
            bbox = (
                float(polygon[:, 1].min()),
                float(polygon[:, 0].min()),
                float(polygon[:, 1].max()),
                float(polygon[:, 0].max()),
            )
            bbox = (
                max(0.0, min(float(width), bbox[0])),
                max(0.0, min(float(height), bbox[1])),
                max(0.0, min(float(width), bbox[2])),
                max(0.0, min(float(height), bbox[3])),
            )
            crop = system_crop(page_gray, bbox, margin, system_w, system_h)
            crop_row = system_writer.append(crop[..., None])
            if split == "train":
                scaled = crop.astype(np.float64) / 255.0
                system_sum += float(scaled.sum())
                system_sumsq += float((scaled * scaled).sum())
                system_count += int(scaled.size)

            system_id = f"{page_id}__s{index:02d}"
            system = {
                "system_id": system_id,
                "page_id": page_id,
                "piece": page["piece"],
                "work_group": page["work_group"],
                "split": split,
                "page_num": int(page["page_num"]),
                "system_idx": int(index),
                "bbox_x0": bbox[0],
                "bbox_y0": bbox[1],
                "bbox_x1": bbox[2],
                "bbox_y1": bbox[3],
                "y_center": float(polygon[:, 0].mean()),
                "x_center": float(polygon[:, 1].mean()),
                "polygon_area": polygon_area(polygon),
                "n_noteheads": 0,
                "crop_row": int(crop_row),
                "page_row": int(page_row),
            }
            system_rows.append(system)
            page_systems[page_id].append(system)

        page_rows.append(
            {
                "page_id": page_id,
                "piece": page["piece"],
                "work_group": page["work_group"],
                "split": split,
                "page_num": int(page["page_num"]),
                "image_path": str(page["png_path"]),
                "width": int(width),
                "height": int(height),
                "n_systems": int(len(systems)),
                "page_row": int(page_row),
            }
        )

        page_noteheads = noteheads[noteheads["page_id"] == page_id]
        paths = [
            MplPath(polygon[:, [1, 0]]) for polygon in systems
        ]
        for notehead in page_noteheads.to_dict("records"):
            x = float(notehead["cx"])
            y = float(notehead["cy"])
            hit = None
            for index, path in enumerate(paths):
                if path.contains_point((x, y)):
                    hit = index
                    break
            if hit is not None:
                method = "polygon"
                distance = 0.0
            elif len(systems):
                hit = int(np.argmin(np.abs(y_centers - y)))
                method = "nearest_y"
                distance = float(abs(y_centers[hit] - y))
            else:
                method = "unassigned"
                distance = np.nan
            if hit is None:
                assignment_rows.append(
                    {
                        "notehead_id": notehead["notehead_id"],
                        "page_id": page_id,
                        "piece": notehead["piece"],
                        "split": split,
                        "system_id": None,
                        "system_idx": -1,
                        "assignment_method": method,
                        "system_distance": distance,
                    }
                )
                continue
            system = page_systems[page_id][hit]
            system["n_noteheads"] += 1
            assignment_rows.append(
                {
                    "notehead_id": notehead["notehead_id"],
                    "page_id": page_id,
                    "piece": notehead["piece"],
                    "split": split,
                    "system_id": system["system_id"],
                    "system_idx": int(hit),
                    "assignment_method": method,
                    "system_distance": distance,
                }
            )

    n_pages = page_writer.finalize()
    n_systems = system_writer.finalize()
    if n_pages != len(page_rows) or n_systems != len(system_rows):
        raise RuntimeError(
            f"Task 1 tensor/table mismatch for {split}: "
            f"pages {n_pages}/{len(page_rows)}, systems {n_systems}/{len(system_rows)}"
        )

    pages_df = pd.DataFrame(page_rows)
    systems_df = pd.DataFrame(system_rows)
    assignments_df = pd.DataFrame(assignment_rows)
    write_parquet(pages_df, proc_dir / f"pages_{split}.parquet")
    write_parquet(systems_df, proc_dir / f"systems_{split}.parquet")
    write_parquet(assignments_df, proc_dir / f"notehead_systems_{split}.parquet")

    systems_by_id = {
        row["system_id"]: row for row in systems_df.to_dict("records")
    }
    systems_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    systems_by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_systems: list[dict[str, Any]] = []
    for system in systems_df.to_dict("records"):
        systems_by_page[system["page_id"]].append(system)
        systems_by_piece[system["piece"]].append(system)
        all_systems.append(system)

    assignment_map = {
        row["notehead_id"]: row["system_id"]
        for row in assignments_df.to_dict("records")
        if row["system_id"] is not None
    }

    rng = np.random.default_rng(int(config["seed"]) + SPLIT_OFFSET[split] + 100)
    pair_rows: list[dict[str, Any]] = []
    skipped_queries = 0
    for window in windows.to_dict("records"):
        positive_id = assignment_map.get(window["notehead_id"])
        if positive_id is None:
            skipped_queries += 1
            continue
        positive = systems_by_id[positive_id]
        chosen = {positive_id}
        candidates: list[tuple[dict[str, Any], str]] = []

        same_page = [
            item
            for item in systems_by_page.get(positive["page_id"], [])
            if item["system_id"] != positive_id
        ]
        for item in choose_candidates(rng, same_page, 2, chosen):
            chosen.add(item["system_id"])
            candidates.append((item, "same_page"))

        same_piece = [
            item
            for item in systems_by_piece.get(positive["piece"], [])
            if item["page_id"] != positive["page_id"] and item["system_id"] not in chosen
        ]
        for item in choose_candidates(rng, same_piece, 1, chosen):
            chosen.add(item["system_id"])
            candidates.append((item, "same_piece_other_page"))

        cross_piece = [
            item for item in all_systems if item["piece"] != positive["piece"]
        ]
        for item in choose_candidates(rng, cross_piece, 1, chosen):
            chosen.add(item["system_id"])
            candidates.append((item, "cross_piece"))

        if len(candidates) < negatives_per_positive:
            remaining = [
                item for item in all_systems if item["system_id"] not in chosen
            ]
            need = negatives_per_positive - len(candidates)
            for item in choose_candidates(rng, remaining, need, chosen):
                chosen.add(item["system_id"])
                candidates.append((item, "other"))

        pair_rows.append(
            {
                "query_id": window["window_id"],
                "window_id": window["window_id"],
                "window_row": int(window["window_row"]),
                "notehead_id": window["notehead_id"],
                "piece": window["piece"],
                "split": split,
                "positive_system_id": positive_id,
                "candidate_system_id": positive_id,
                "candidate_page_id": positive["page_id"],
                "candidate_crop_row": int(positive["crop_row"]),
                "candidate_page_row": int(positive["page_row"]),
                "label": 1,
                "neg_type": "positive",
            }
        )
        for candidate, neg_type in candidates:
            pair_rows.append(
                {
                    "query_id": window["window_id"],
                    "window_id": window["window_id"],
                    "window_row": int(window["window_row"]),
                    "notehead_id": window["notehead_id"],
                    "piece": window["piece"],
                    "split": split,
                    "positive_system_id": positive_id,
                    "candidate_system_id": candidate["system_id"],
                    "candidate_page_id": candidate["page_id"],
                    "candidate_crop_row": int(candidate["crop_row"]),
                    "candidate_page_row": int(candidate["page_row"]),
                    "label": 0,
                    "neg_type": neg_type,
                }
            )

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df["pair_id"] = [f"{split}_sys_{index:07d}" for index in range(len(pairs_df))]
    write_parquet(pairs_df, proc_dir / f"pairs_system_{split}.parquet")

    page_mean = page_sum / page_count if page_count else 0.0
    page_var = max(page_sumsq / page_count - page_mean * page_mean, 0.0) if page_count else 0.0
    system_mean = system_sum / system_count if system_count else 0.0
    system_var = (
        max(system_sumsq / system_count - system_mean * system_mean, 0.0)
        if system_count
        else 0.0
    )

    return {
        "split": split,
        "n_pages": int(n_pages),
        "n_systems": int(n_systems),
        "n_noteheads": int(len(assignments_df)),
        "n_assigned": int((assignments_df["system_id"].notna()).sum()),
        "n_polygon": int((assignments_df["assignment_method"] == "polygon").sum()),
        "n_nearest": int((assignments_df["assignment_method"] == "nearest_y").sum()),
        "n_unassigned": int((assignments_df["assignment_method"] == "unassigned").sum()),
        "n_pairs": int(len(pairs_df)),
        "n_queries": int(pairs_df["query_id"].nunique()),
        "n_skipped_queries": int(skipped_queries),
        "page_mean": float(page_mean),
        "page_std": float(np.sqrt(page_var)),
        "system_mean": float(system_mean),
        "system_std": float(np.sqrt(system_var)),
    }


def run_task1(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_root"])
    proc_dir = root / config["processed_dir"]
    summaries: dict[str, Any] = {}
    for split in SPLITS:
        summaries[split] = build_split(config, split)
        LOGGER.info("Task 1 %s: %s", split, summaries[split])

    train = summaries["train"]
    norm_stats = {
        "page": {
            "size": int(config["task1"]["page_size"]),
            "mean": train["page_mean"],
            "std": train["page_std"],
            "storage": "uint8_raw_grayscale",
        },
        "system": {
            "crop_w": int(config["task1"]["system_crop_w"]),
            "crop_h": int(config["task1"]["system_crop_h"]),
            "margin": int(config["task1"]["system_margin"]),
            "mean": train["system_mean"],
            "std": train["system_std"],
            "storage": "uint8_raw_grayscale",
        },
        "level": "system",
        "bars_available": False,
        "assignment": "point_in_polygon_then_nearest_y",
    }
    write_json(proc_dir / "norm_stats_task1.json", norm_stats)

    summary = {
        "splits": summaries,
        "total_pages": int(sum(item["n_pages"] for item in summaries.values())),
        "total_systems": int(sum(item["n_systems"] for item in summaries.values())),
        "total_pairs": int(sum(item["n_pairs"] for item in summaries.values())),
    }
    write_json(proc_dir / "preprocess_summary_task1.json", summary)
    return summary
