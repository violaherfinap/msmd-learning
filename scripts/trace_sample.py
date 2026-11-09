#!/usr/bin/env python3
"""Trace selected samples from raw inputs to generated artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.msmd_preprocess.mung import crop_notehead, load_gray
from src.msmd_preprocess.paths import long_path, relative_posix

SPLITS = ("train", "val", "test")


def load_npy(path: Path) -> np.ndarray:
    return np.load(long_path(path), mmap_mode="r")


def trace_sample(
    split: str,
    notehead: pd.Series,
    windows: pd.DataFrame,
    assignments: pd.DataFrame,
    systems: pd.DataFrame,
    pairs_system: pd.DataFrame,
    task2: pd.DataFrame,
    crops: np.ndarray,
    audio: np.ndarray,
    errors: np.ndarray,
) -> dict:
    notehead_id = str(notehead["notehead_id"])
    raw_path = Path(str(notehead["png_path"]))
    gray = load_gray(raw_path)
    raw_crop, _ = crop_notehead(gray, float(notehead["cx"]), float(notehead["cy"]), 160, 200)
    saved_crop = np.asarray(crops[int(notehead["crop_row"]), :, :, 0], dtype=np.int16)
    crop_diff = np.abs(raw_crop.astype(np.int16) - saved_crop)

    window_rows = windows[windows["notehead_id"] == notehead_id]
    if window_rows.empty:
        raise RuntimeError(f"Window tidak ditemukan untuk {notehead_id}")
    window = window_rows.iloc[0]
    window_row = int(window["window_row"])
    audio_window = np.asarray(audio[window_row], dtype=np.float32)

    assignment_rows = assignments[assignments["notehead_id"] == notehead_id]
    assignment = assignment_rows.iloc[0] if not assignment_rows.empty else None
    system_id = str(assignment["system_id"]) if assignment is not None else None
    system_row = systems[systems["system_id"] == system_id].iloc[0] if system_id else None

    query_id = str(window["window_id"])
    query = pairs_system[pairs_system["window_id"] == query_id]
    positive = query[query["label"] == 1] if not query.empty else query

    task2_rows = task2[task2["source_window_id"] == query_id]
    task2_rows = task2_rows.sort_values("error_row")

    return {
        "notehead_id": notehead_id,
        "piece": str(notehead["piece"]),
        "page_id": str(notehead["page_id"]),
        "pitch": int(notehead["pitch"]),
        "raw_png": str(raw_path),
        "raw_crop_max_abs_diff": int(crop_diff.max()),
        "raw_crop_mean_abs_diff": float(crop_diff.mean()),
        "saved_crop_shape": list(saved_crop.shape),
        "window_id": query_id,
        "window_row": window_row,
        "window_shape": list(audio_window.shape),
        "window_finite": bool(np.isfinite(audio_window).all()),
        "window_nonzero": bool(np.abs(audio_window).max() > 0),
        "onset_frame": int(window["onset_frame"]),
        "start_frame": int(window["start_frame"]),
        "end_frame": int(window["end_frame"]),
        "onset_inside_window": bool(
            int(window["start_frame"]) <= int(window["onset_frame"]) < int(window["end_frame"])
        ),
        "spec_T": int(window["spec_T"]),
        "midi_T": int(window["midi_T"]),
        "aligned_T": int(window["aligned_T"]),
        "aligned_T_ok": int(window["aligned_T"]) == min(int(window["spec_T"]), int(window["midi_T"])),
        "assignment_method": str(assignment["assignment_method"]) if assignment is not None else None,
        "system_id": system_id,
        "system_bbox": (
            [
                float(system_row["bbox_x0"]),
                float(system_row["bbox_y0"]),
                float(system_row["bbox_x1"]),
                float(system_row["bbox_y1"]),
            ]
            if system_row is not None
            else None
        ),
        "task1_query_rows": int(len(query)),
        "task1_positive_rows": int(len(positive)),
        "task1_positive_system_id": (
            str(positive.iloc[0]["positive_system_id"]) if not positive.empty else None
        ),
        "task2_rows": int(len(task2_rows)),
        "task2_classes": [int(value) for value in task2_rows["class_id"].tolist()],
        "task2_error_rows": [int(value) for value in task2_rows["error_row"].tolist()],
        "task2_error_shapes": [
            list(np.asarray(errors[int(row)], dtype=np.float32).shape)
            for row in task2_rows["error_row"].tolist()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace sampel preprocessing MSMD")
    parser.add_argument("--split", default="val", choices=SPLITS)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/qa_v2/trace_samples.json")
    args = parser.parse_args()

    proc = ROOT / "data" / "processed"
    noteheads = pd.read_parquet(proc / f"noteheads_{args.split}.parquet")
    windows = pd.read_parquet(proc / f"windows_{args.split}.parquet")
    assignments = pd.read_parquet(proc / f"notehead_systems_{args.split}.parquet")
    systems = pd.read_parquet(proc / f"systems_{args.split}.parquet")
    pairs_system = pd.read_parquet(proc / f"pairs_system_{args.split}.parquet")
    task2 = pd.read_parquet(proc / f"task2_windows_{args.split}.parquet")
    crops = load_npy(proc / f"score_crops_{args.split}.npy")
    audio = load_npy(proc / f"audio_windows_{args.split}.npy")
    errors = load_npy(proc / f"error_windows_{args.split}.npy")

    samples = []
    for _, row in noteheads.sample(frac=1.0, random_state=args.seed).iterrows():
        if not Path(str(row["png_path"])).is_file():
            continue
        if windows[windows["notehead_id"] == row["notehead_id"]].empty:
            continue
        samples.append(
            trace_sample(
                args.split,
                row,
                windows,
                assignments,
                systems,
                pairs_system,
                task2,
                crops,
                audio,
                errors,
            )
        )
        if len(samples) >= args.count:
            break

    issues = []
    for sample in samples:
        if sample["raw_crop_max_abs_diff"] != 0:
            issues.append(f"{sample['notehead_id']}: crop diff {sample['raw_crop_max_abs_diff']}")
        if not sample["window_finite"] or not sample["window_nonzero"]:
            issues.append(f"{sample['notehead_id']}: window tidak valid")
        if not sample["onset_inside_window"]:
            issues.append(f"{sample['notehead_id']}: onset di luar window")
        if not sample["aligned_T_ok"]:
            issues.append(f"{sample['notehead_id']}: aligned_T tidak cocok")
        if sample["task1_query_rows"] != 5 or sample["task1_positive_rows"] != 1:
            issues.append(f"{sample['notehead_id']}: query Task 1 tidak valid")
        if sample["task2_rows"] != 4 or sorted(sample["task2_classes"]) != [0, 1, 2, 3]:
            issues.append(f"{sample['notehead_id']}: varian Task 2 tidak lengkap")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": not issues,
        "issues": issues,
        "samples": samples,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "split": args.split,
                "samples": len(samples),
                "output": relative_posix(out, ROOT),
                "issues": issues,
            },
            indent=2,
        )
    )
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
