"""Artifact validation for the regenerated preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SPLITS
from .io import write_json
from .paths import long_path

EXPECTED_CORE = {
    "train": {"crops": 14365, "windows": 28730, "pairs": 57460},
    "val": {"crops": 2430, "windows": 9720, "pairs": 19440},
    "test": {"crops": 2018, "windows": 8072, "pairs": 16144},
}
EXPECTED_TASK1 = {"pages": 60, "systems": 314, "pairs": 232610}
EXPECTED_TASK2 = {"source_windows": 46522, "error_windows": 186088}


def _load_npy(path: Path) -> np.ndarray:
    return np.load(long_path(path), mmap_mode="r")


def _finite_chunks(array: np.ndarray, chunk: int = 5000) -> bool:
    for start in range(0, array.shape[0], chunk):
        block = np.asarray(array[start : start + chunk], dtype=np.float32)
        if not np.isfinite(block).all():
            return False
    return True


def _tensor_stats(array: np.ndarray, rows: np.ndarray, chunk: int = 500) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for start in range(0, len(rows), chunk):
        block = np.asarray(array[rows[start : start + chunk]], dtype=np.float64)
        total += float(block.sum())
        total_sq += float((block * block).sum())
        count += int(block.size)
    mean = total / count if count else 0.0
    var = max(total_sq / count - mean * mean, 0.0) if count else 0.0
    return float(mean), float(np.sqrt(var))


def _compare_stats(
    stats: dict[str, Any],
    key: str,
    mean: float,
    std: float,
    tolerance: float = 1e-4,
) -> bool:
    return (
        abs(float(stats[key]["mean"]) - mean) <= tolerance
        and abs(float(stats[key]["std"]) - std) <= tolerance
    )


def validate_core(config: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    proc = Path(config["_root"]) / config["processed_dir"]
    report: dict[str, Any] = {"splits": {}}
    stats = json.loads((proc / "norm_stats_note.json").read_text(encoding="utf-8"))

    for split in SPLITS:
        crops = _load_npy(proc / f"score_crops_{split}.npy")
        windows = _load_npy(proc / f"audio_windows_{split}.npy")
        noteheads = pd.read_parquet(proc / f"noteheads_{split}.parquet")
        win_df = pd.read_parquet(proc / f"windows_{split}.parquet")
        pairs = pd.read_parquet(proc / f"pairs_note_{split}.parquet")

        expected = EXPECTED_CORE[split]
        actual = {
            "crops": int(crops.shape[0]),
            "windows": int(windows.shape[0]),
            "pairs": int(len(pairs)),
        }
        if actual != expected:
            errors.append(f"core {split}: counts {actual} != {expected}")

        if crops.shape[1:] != (int(config["crop_h"]), int(config["crop_w"]), 1):
            errors.append(f"core {split}: crop shape {crops.shape[1:]}")
        if windows.shape[1:] != (int(config["win_frames"]), int(config["spec_bins"])):
            errors.append(f"core {split}: window shape {windows.shape[1:]}")
        if crops.dtype != np.uint8 or windows.dtype != np.float16:
            errors.append(f"core {split}: dtype {crops.dtype}/{windows.dtype}")
        if not _finite_chunks(windows):
            errors.append(f"core {split}: non-finite audio windows")
        if len(noteheads) != actual["crops"]:
            errors.append(f"core {split}: noteheads table {len(noteheads)} != {actual['crops']}")
        if len(win_df) != actual["windows"]:
            errors.append(f"core {split}: windows table {len(win_df)} != {actual['windows']}")
        if not (noteheads["crop_row"].to_numpy() == np.arange(len(noteheads))).all():
            errors.append(f"core {split}: crop_row is not contiguous")
        if not (win_df["window_row"].to_numpy() == np.arange(len(win_df))).all():
            errors.append(f"core {split}: window_row is not contiguous")
        if not (win_df["aligned_T"].to_numpy() == np.minimum(win_df["spec_T"], win_df["midi_T"])).all():
            errors.append(f"core {split}: aligned_T mismatch")
        if not ((win_df["pad_left"] >= 0).all() and (win_df["pad_right"] >= 0).all()):
            errors.append(f"core {split}: negative padding")
        if not (
            (win_df["start_frame"] <= win_df["onset_frame"])
            & (win_df["onset_frame"] < win_df["end_frame"])
        ).all():
            errors.append(f"core {split}: onset outside window")
        if noteheads["notehead_id"].duplicated().any():
            errors.append(f"core {split}: duplicate notehead_id")
        if win_df["window_id"].duplicated().any():
            errors.append(f"core {split}: duplicate window_id")

        if len(pairs) != 2 * len(win_df):
            errors.append(
                f"core {split}: {len(pairs)} pairs != 2 x {len(win_df)} windows"
            )
        if int((pairs["label"] == 1).sum()) != len(win_df):
            errors.append(f"core {split}: positive pair count mismatch")
        window_counts = win_df.groupby("notehead_id").size()
        positive_counts = (
            pairs[pairs["label"] == 1].groupby("notehead_id").size().reindex(
                window_counts.index, fill_value=0
            )
        )
        negative_counts = (
            pairs[pairs["label"] == 0].groupby("notehead_id").size().reindex(
                window_counts.index, fill_value=0
            )
        )
        if not positive_counts.equals(window_counts):
            errors.append(f"core {split}: positive pairs per notehead mismatch")
        if not negative_counts.equals(window_counts):
            errors.append(f"core {split}: negative pairs per notehead mismatch")
        positive = pairs[pairs["label"] == 1].merge(
            win_df[["window_id", "notehead_id"]],
            on=["window_id", "notehead_id"],
            how="left",
            indicator=True,
        )
        if not (positive["_merge"] == "both").all():
            errors.append(f"core {split}: positive pair window/notehead mismatch")
        if not set(pairs["window_id"]).issubset(set(win_df["window_id"])):
            errors.append(f"core {split}: pair references unknown window")
        negatives = pairs[pairs["label"] == 0]
        if (negatives["window_piece"] == negatives["piece"]).any():
            errors.append(f"core {split}: negative sampled from the same piece")

        report["splits"][split] = actual

    train_crops = _load_npy(proc / "score_crops_train.npy")
    train_rows = np.arange(train_crops.shape[0])
    image_mean, image_std = _tensor_stats(train_crops, train_rows)
    image_mean /= 255.0
    image_std /= 255.0
    if not _compare_stats(stats, "image", image_mean, image_std):
        errors.append(
            f"core: image stats {image_mean:.8f}/{image_std:.8f} "
            f"!= {stats['image']['mean']:.8f}/{stats['image']['std']:.8f}"
        )
    report["image_stats"] = {"mean": image_mean, "std": image_std}

    return report


def validate_task1(config: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    proc = Path(config["_root"]) / config["processed_dir"]
    report: dict[str, Any] = {"splits": {}}
    total_pages = total_systems = total_pairs = 0

    for split in SPLITS:
        pages = pd.read_parquet(proc / f"pages_{split}.parquet")
        systems = pd.read_parquet(proc / f"systems_{split}.parquet")
        assignments = pd.read_parquet(proc / f"notehead_systems_{split}.parquet")
        pairs = pd.read_parquet(proc / f"pairs_system_{split}.parquet")
        page_tensor = _load_npy(proc / f"score_pages_{split}.npy")
        system_tensor = _load_npy(proc / f"system_crops_{split}.npy")

        actual = {
            "pages": int(len(pages)),
            "systems": int(len(systems)),
            "pairs": int(len(pairs)),
        }
        total_pages += actual["pages"]
        total_systems += actual["systems"]
        total_pairs += actual["pairs"]

        if page_tensor.shape[0] != len(pages) or system_tensor.shape[0] != len(systems):
            errors.append(f"task1 {split}: tensor/table row mismatch")
        if page_tensor.shape[1:] != (
            int(config["task1"]["page_size"]),
            int(config["task1"]["page_size"]),
            1,
        ):
            errors.append(f"task1 {split}: page tensor shape {page_tensor.shape[1:]}")
        if system_tensor.shape[1:] != (
            int(config["task1"]["system_crop_h"]),
            int(config["task1"]["system_crop_w"]),
            1,
        ):
            errors.append(f"task1 {split}: system tensor shape {system_tensor.shape[1:]}")
        if page_tensor.dtype != np.uint8 or system_tensor.dtype != np.uint8:
            errors.append(f"task1 {split}: tensor dtype")
        if assignments["system_id"].isna().any():
            errors.append(f"task1 {split}: unassigned noteheads")
        if assignments["notehead_id"].duplicated().any():
            errors.append(f"task1 {split}: duplicate notehead assignments")
        if not set(assignments["system_id"].dropna()).issubset(set(systems["system_id"])):
            errors.append(f"task1 {split}: assignment references unknown system")
        if not (systems["crop_row"].to_numpy() == np.arange(len(systems))).all():
            errors.append(f"task1 {split}: system crop_row is not contiguous")
        if not (pages["page_row"].to_numpy() == np.arange(len(pages))).all():
            errors.append(f"task1 {split}: page_row is not contiguous")

        proposed = pd.read_parquet(proc / f"windows_{split}.parquet")
        grouped = pairs.groupby("query_id")["label"].agg(["size", "sum"])
        if len(grouped) != len(proposed):
            errors.append(
                f"task1 {split}: {len(grouped)} queries != {len(proposed)} core windows"
            )
        if not (grouped["size"] == 5).all() or not (grouped["sum"] == 1).all():
            errors.append(f"task1 {split}: pairs are not 1 positive + 4 negatives")
        if not set(pairs["candidate_system_id"]).issubset(set(systems["system_id"])):
            errors.append(f"task1 {split}: pair references unknown candidate system")
        if not set(pairs["positive_system_id"]).issubset(set(systems["system_id"])):
            errors.append(f"task1 {split}: pair references unknown positive system")
        negatives = pairs[pairs["label"] == 0]
        if (negatives["candidate_system_id"] == negatives["positive_system_id"]).any():
            errors.append(f"task1 {split}: negative candidate equals positive")

        report["splits"][split] = {
            **actual,
            "n_queries": int(pairs["query_id"].nunique()),
            "n_assigned": int(assignments["system_id"].notna().sum()),
            "n_polygon": int((assignments["assignment_method"] == "polygon").sum()),
            "n_nearest": int((assignments["assignment_method"] == "nearest_y").sum()),
            "n_unassigned": int((assignments["assignment_method"] == "unassigned").sum()),
        }

    if [total_pages, total_systems, total_pairs] != [
        EXPECTED_TASK1["pages"],
        EXPECTED_TASK1["systems"],
        EXPECTED_TASK1["pairs"],
    ]:
        errors.append(
            f"task1 totals pages={total_pages} systems={total_systems} pairs={total_pairs}"
        )
    report["totals"] = {
        "pages": total_pages,
        "systems": total_systems,
        "pairs": total_pairs,
    }
    return report


def validate_task2(config: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    proc = Path(config["_root"]) / config["processed_dir"]
    report: dict[str, Any] = {"splits": {}}
    total_source = total_error = 0

    for split in SPLITS:
        table = pd.read_parquet(proc / f"task2_windows_{split}.parquet")
        tensor = _load_npy(proc / f"error_windows_{split}.npy")

        actual = {
            "source_windows": int(table["source_window_id"].nunique()),
            "error_windows": int(len(table)),
        }
        total_source += actual["source_windows"]
        total_error += actual["error_windows"]
        core_windows = pd.read_parquet(proc / f"windows_{split}.parquet")
        if actual["source_windows"] != len(core_windows):
            errors.append(
                f"task2 {split}: {actual['source_windows']} sources "
                f"!= {len(core_windows)} core windows"
            )

        if tensor.shape[0] != len(table):
            errors.append(f"task2 {split}: tensor/table row mismatch")
        if tensor.shape[1:] != (int(config["win_frames"]), int(config["spec_bins"])):
            errors.append(f"task2 {split}: tensor shape {tensor.shape[1:]}")
        if tensor.dtype != np.float16:
            errors.append(f"task2 {split}: tensor dtype {tensor.dtype}")
        if not _finite_chunks(tensor):
            errors.append(f"task2 {split}: non-finite error windows")
        if not (table["error_row"].to_numpy() == np.arange(len(table))).all():
            errors.append(f"task2 {split}: error_row is not contiguous")
        if table["error_id"].duplicated().any():
            errors.append(f"task2 {split}: duplicate error_id")

        grouped = table.groupby("source_window_id")["class_id"].agg(
            ["size", lambda values: tuple(sorted(set(values)))]
        )
        grouped.columns = ["size", "classes"]
        if not (grouped["size"] == 4).all():
            errors.append(f"task2 {split}: not exactly 4 variants per source")
        if not (grouped["classes"] == (0, 1, 2, 3)).all():
            errors.append(f"task2 {split}: class set is not (0,1,2,3)")

        wrong = table[table["class_name"] == "wrong_pitch"]
        delete = table[table["class_name"] == "delete"]
        insert = table[table["class_name"] == "insert"]
        if (wrong["pitch_target"] == wrong["pitch_source"]).any():
            errors.append(f"task2 {split}: wrong_pitch target equals source")
        if (delete["pitch_target"] != delete["pitch_source"]).any():
            errors.append(f"task2 {split}: delete changed pitch metadata")
        if (insert["pitch_target"] == insert["pitch_source"]).any():
            errors.append(f"task2 {split}: insert target equals source")

        report["splits"][split] = {
            **actual,
            "class_counts": {
                int(class_id): int((table["class_id"] == class_id).sum())
                for class_id in range(4)
            },
            "n_fallbacks": int(table["fallback"].sum()),
            "n_skipped": int(actual["source_windows"] * 4 - actual["error_windows"]),
        }

    if [total_source, total_error] != [
        EXPECTED_TASK2["source_windows"],
        EXPECTED_TASK2["error_windows"],
    ]:
        errors.append(f"task2 totals source={total_source} error={total_error}")
    report["totals"] = {"source_windows": total_source, "error_windows": total_error}
    return report


def validate_preprocessing(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "warnings": warnings,
        "core": {},
        "task1": {},
        "task2": {},
    }

    report["core"] = validate_core(config, errors, warnings)
    report["task1"] = validate_task1(config, errors, warnings)
    report["task2"] = validate_task2(config, errors, warnings)

    report["ok"] = not errors
    reports_dir = Path(config["_root"]) / config["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "validation_preprocessing.json", report)
    return report
