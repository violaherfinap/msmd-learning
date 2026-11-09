#!/usr/bin/env python3
"""Generate visual QA figures from existing preprocessing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.msmd_preprocess.mung import crop_notehead, load_gray
from src.msmd_preprocess.paths import long_path, relative_posix

SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "clean", 1: "wrong_pitch", 2: "delete", 3: "insert"}


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def load_npy(path: Path) -> np.ndarray:
    return np.load(long_path(path), mmap_mode="r")


def page_image(
    page_row: pd.Series,
    page_tensor: np.ndarray,
) -> tuple[Image.Image, float, float]:
    raw_path = str(page_row.get("image_path", ""))
    if raw_path and raw_path.lower() != "nan" and Path(raw_path).is_file():
        gray = load_gray(raw_path)
        image = Image.fromarray(gray).convert("RGB")
        width = float(page_row.get("width", image.width))
        height = float(page_row.get("height", image.height))
        return image, image.width / max(width, 1.0), image.height / max(height, 1.0)

    row = int(page_row["page_row"])
    gray = np.asarray(page_tensor[row, :, :, 0], dtype=np.uint8)
    image = Image.fromarray(gray).convert("RGB")
    width = float(page_row.get("width", image.width))
    height = float(page_row.get("height", image.height))
    return image, image.width / max(width, 1.0), image.height / max(height, 1.0)


def fit_width(image: Image.Image, target: int = 1100) -> Image.Image:
    if image.width <= target:
        return image
    scale = target / image.width
    return image.resize((target, int(image.height * scale)), Image.Resampling.LANCZOS)


def save_figure(fig, path: Path) -> str:
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return relative_posix(path, ROOT)


def figure_notehead_page(
    split: str,
    proc: Path,
    out: Path,
    page_tensor: np.ndarray,
) -> str:
    noteheads = read_parquet(proc / f"noteheads_{split}.parquet")
    pages = read_parquet(proc / f"pages_{split}.parquet")
    page_counts = noteheads.groupby("page_id").size().sort_values(ascending=False)
    page_id = str(page_counts.index[0])
    page_row = pages[pages["page_id"] == page_id].iloc[0]
    image, scale_x, scale_y = page_image(page_row, page_tensor)
    image = fit_width(image).convert("RGB")
    if image.width != int(float(page_row.get("width", image.width))):
        resize_scale = image.width / max(float(page_row.get("width", image.width)), 1.0)
        scale_x *= resize_scale
        scale_y *= resize_scale

    rows = noteheads[noteheads["page_id"] == page_id]
    chosen = rows.sample(min(18, len(rows)), random_state=42)
    chosen_ids = set(chosen["notehead_id"].astype(str))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for row in rows.itertuples(index=False):
        x0 = float(row.box_left) * scale_x
        y0 = float(row.box_top) * scale_y
        x1 = x0 + float(row.box_w) * scale_x
        y1 = y0 + float(row.box_h) * scale_y
        draw.rectangle([x0, y0, x1, y1], outline=(210, 40, 40), width=1)
        if str(row.notehead_id) in chosen_ids:
            cx = float(row.cx) * scale_x
            cy = float(row.cy) * scale_y
            draw.rectangle(
                [cx - 80 * scale_x, cy - 100 * scale_y, cx + 80 * scale_x, cy + 100 * scale_y],
                outline=(20, 140, 70),
                width=2,
            )
            draw.text((x1 + 2, y0 - 10), f"p{int(row.pitch)}", fill=(20, 80, 170), font=font)

    path = out / "01_notehead_page_overlay.png"
    image.save(path)
    return relative_posix(path, ROOT)


def figure_notehead_crops(split: str, proc: Path, out: Path, seed: int, count: int) -> str:
    noteheads = read_parquet(proc / f"noteheads_{split}.parquet")
    crops = load_npy(proc / f"score_crops_{split}.npy")
    rows = noteheads.sample(min(count, len(noteheads)), random_state=seed)
    cols = 4
    nrows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (_, row) in zip(axes, rows.iterrows()):
        crop = crops[int(row["crop_row"]), :, :, 0]
        axis.imshow(crop, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"pitch={int(row['pitch'])} | ink={float(row['ink_ratio']):.3f}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes[len(rows):]:
        axis.axis("off")
    fig.suptitle(f"Crop notehead {split} (200x160)", fontsize=12)
    fig.tight_layout()
    path = out / "02_notehead_crops.png"
    return save_figure(fig, path)


def figure_audio_windows(split: str, proc: Path, out: Path, seed: int, count: int) -> str:
    windows = read_parquet(proc / f"windows_{split}.parquet")
    audio = load_npy(proc / f"audio_windows_{split}.npy")
    rows = windows.sample(min(count, len(windows)), random_state=seed)
    cols = 2
    nrows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (_, row) in zip(axes, rows.iterrows()):
        window = np.asarray(audio[int(row["window_row"])], dtype=np.float32)
        onset = int(row["onset_frame"] - row["start_frame"])
        axis.imshow(window.T, aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3)
        axis.axvline(onset - 0.5, color="cyan", linewidth=1.2)
        axis.set_title(
            f"pitch={int(row['pitch'])} | onset={onset} | {row['soundfont']}",
            fontsize=8,
        )
        axis.set_xlabel("frame")
        axis.set_ylabel("bin")
    for axis in axes[len(rows):]:
        axis.axis("off")
    fig.suptitle(f"Window audio {split} (42 frame, garis cyan = onset)", fontsize=12)
    fig.tight_layout()
    path = out / "03_audio_windows.png"
    return save_figure(fig, path)


def pick_task1_query(split: str, proc: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pairs = read_parquet(proc / f"pairs_system_{split}.parquet")
    systems = read_parquet(proc / f"systems_{split}.parquet")
    grouped = pairs.groupby("query_id")
    for query_id, group in grouped:
        if len(group) == 5 and int((group["label"] == 1).sum()) == 1:
            positive = group[group["label"] == 1].iloc[0]
            system_row = systems[systems["system_id"] == positive["positive_system_id"]]
            if len(system_row) == 1:
                return group.reset_index(drop=True), systems, system_row.iloc[0]
    raise RuntimeError(f"Tidak menemukan query Task 1 yang valid untuk split {split}")


def figure_task1_overlay(
    split: str,
    proc: Path,
    out: Path,
    page_tensor: np.ndarray,
) -> str:
    group, systems, positive_system = pick_task1_query(split, proc)
    pages = read_parquet(proc / f"pages_{split}.parquet")
    page_id = str(positive_system["page_id"])
    page_row = pages[pages["page_id"] == page_id].iloc[0]
    image, scale_x, scale_y = page_image(page_row, page_tensor)
    image = fit_width(image).convert("RGB")
    if image.width != int(float(page_row.get("width", image.width))):
        resize_scale = image.width / max(float(page_row.get("width", image.width)), 1.0)
        scale_x *= resize_scale
        scale_y *= resize_scale

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    page_systems = systems[systems["page_id"] == page_id]
    for row in page_systems.itertuples(index=False):
        x0 = float(row.bbox_x0) * scale_x
        y0 = float(row.bbox_y0) * scale_y
        x1 = float(row.bbox_x1) * scale_x
        y1 = float(row.bbox_y1) * scale_y
        draw.rectangle([x0, y0, x1, y1], outline=(20, 140, 70), width=2)

    off_page = []
    for row in group.itertuples(index=False):
        if str(row.candidate_page_id) != page_id:
            off_page.append(str(row.neg_type))
            continue
        candidate = systems[systems["system_id"] == row.candidate_system_id].iloc[0]
        x0 = float(candidate["bbox_x0"]) * scale_x
        y0 = float(candidate["bbox_y0"]) * scale_y
        x1 = float(candidate["bbox_x1"]) * scale_x
        y1 = float(candidate["bbox_y1"]) * scale_y
        color = (220, 40, 40) if int(row.label) == 1 else (240, 150, 20)
        width = 3 if int(row.label) == 1 else 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        draw.text((x0 + 3, y0 + 3), f"{row.neg_type}", fill=color, font=font)

    if off_page:
        note = "Kandidat off-page: " + ", ".join(off_page)
        draw.rectangle([0, image.height - 20, image.width, image.height], fill=(255, 255, 255))
        draw.text((6, image.height - 16), note, fill=(30, 30, 30), font=font)

    path = out / "04_task1_system_overlay.png"
    image.save(path)
    return relative_posix(path, ROOT)


def figure_task1_crops(split: str, proc: Path, out: Path) -> str:
    group, systems, _ = pick_task1_query(split, proc)
    crops = load_npy(proc / f"system_crops_{split}.npy")
    fig, axes = plt.subplots(1, len(group), figsize=(4.0 * len(group), 3.3))
    axes = np.atleast_1d(axes)
    for axis, (_, row) in zip(axes, group.iterrows()):
        system = systems[systems["system_id"] == row["candidate_system_id"]].iloc[0]
        crop = crops[int(system["crop_row"]), :, :, 0]
        axis.imshow(crop, cmap="gray", vmin=0, vmax=255)
        page_short = str(row["candidate_page_id"]).split("__")[-1]
        system_short = str(row["candidate_system_id"]).split("__")[-1]
        axis.set_title(
            f"label={int(row['label'])} | {row['neg_type']}\n{page_short}/{system_short}",
            fontsize=7,
        )
        axis.axis("off")
    fig.suptitle(f"Task 1 candidates {split}: 1 positif + 4 negatif", fontsize=12)
    fig.tight_layout()
    path = out / "05_task1_system_crops.png"
    return save_figure(fig, path)


def select_task2_source(split: str, proc: Path) -> pd.DataFrame:
    table = read_parquet(proc / f"task2_windows_{split}.parquet")
    for _, group in table.groupby("source_window_id", sort=False):
        if len(group) == 4 and set(group["class_id"].astype(int)) == {0, 1, 2, 3}:
            return group.sort_values("error_row").reset_index(drop=True)
    raise RuntimeError(f"Tidak menemukan source Task 2 dengan 4 kelas untuk split {split}")


def figure_task2_grid(split: str, proc: Path, out: Path) -> str:
    group = select_task2_source(split, proc)
    errors = load_npy(proc / f"error_windows_{split}.npy")
    clean = np.asarray(errors[int(group.iloc[0]["error_row"])], dtype=np.float32)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for column, (_, row) in enumerate(group.iterrows()):
        variant = np.asarray(errors[int(row["error_row"])], dtype=np.float32)
        axes[0, column].imshow(
            variant.T, aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3
        )
        axes[0, column].set_title(f"{int(row['class_id'])}: {row['class_name']}", fontsize=9)
        axes[0, column].set_xlabel("frame")
        axes[0, column].set_ylabel("bin")
        diff = variant - clean
        axes[1, column].imshow(
            diff.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1.5, vmax=1.5
        )
        axes[1, column].set_title(f"diff: {row['class_name']}", fontsize=9)
        axes[1, column].set_xlabel("frame")
        axes[1, column].set_ylabel("bin")
    source = group.iloc[0]
    fig.suptitle(
        f"Task 2 {split}: {source['soundfont']} | pitch={int(source['pitch_source'])} | {source['source_window_id']}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / "06_task2_class_grid.png"
    return save_figure(fig, path)


def figure_task2_diffs(split: str, proc: Path, out: Path) -> str:
    table = read_parquet(proc / f"task2_windows_{split}.parquet")
    errors = load_npy(proc / f"error_windows_{split}.npy")
    groups = []
    seen_pitch = set()
    fallback = []
    for _, group in table.groupby("source_window_id", sort=False):
        if len(group) == 4 and set(group["class_id"].astype(int)) == {0, 1, 2, 3}:
            ordered = group.sort_values("error_row").reset_index(drop=True)
            fallback.append(ordered)
            pitch = int(ordered.iloc[0]["pitch_source"])
            if pitch not in seen_pitch:
                seen_pitch.add(pitch)
                groups.append(ordered)
            if len(groups) == 3:
                break
    if len(groups) < 3:
        for group in fallback:
            if len(groups) == 3:
                break
            if all(group is not item for item in groups):
                groups.append(group)
    if not groups:
        raise RuntimeError(f"Tidak menemukan source Task 2 untuk split {split}")

    fig, axes = plt.subplots(4, len(groups), figsize=(4.0 * len(groups), 11))
    axes = np.atleast_2d(axes)
    for column, group in enumerate(groups):
        clean = np.asarray(errors[int(group.iloc[0]["error_row"])], dtype=np.float32)
        for row, (_, item) in enumerate(group.iterrows()):
            variant = np.asarray(errors[int(item["error_row"])], dtype=np.float32)
            diff = variant - clean
            axes[row, column].imshow(
                diff.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1.5, vmax=1.5
            )
            axes[row, column].set_title(
                f"{item['class_name']} | p{int(item['pitch_source'])}",
                fontsize=8,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle(f"Task 2 difference examples {split} (variant - clean)", fontsize=12)
    fig.tight_layout()
    path = out / "07_task2_diff_examples.png"
    return save_figure(fig, path)


def build_summary(
    split: str,
    proc: Path,
    out: Path,
    seed: int,
    files: list[str],
    issues: list[str],
) -> dict:
    noteheads = read_parquet(proc / f"noteheads_{split}.parquet")
    windows = read_parquet(proc / f"windows_{split}.parquet")
    pairs_note = read_parquet(proc / f"pairs_note_{split}.parquet")
    pages = read_parquet(proc / f"pages_{split}.parquet")
    systems = read_parquet(proc / f"systems_{split}.parquet")
    assignments = read_parquet(proc / f"notehead_systems_{split}.parquet")
    pairs_system = read_parquet(proc / f"pairs_system_{split}.parquet")
    task2 = read_parquet(proc / f"task2_windows_{split}.parquet")
    crops = load_npy(proc / f"score_crops_{split}.npy")
    audio = load_npy(proc / f"audio_windows_{split}.npy")

    crop_checks = []
    sample = noteheads.sample(min(3, len(noteheads)), random_state=seed)
    for _, row in sample.iterrows():
        raw_path = Path(str(row["png_path"]))
        if not raw_path.is_file():
            issues.append(f"raw png tidak ditemukan: {raw_path}")
            continue
        gray = load_gray(raw_path)
        raw_crop, _ = crop_notehead(gray, float(row["cx"]), float(row["cy"]), 160, 200)
        saved = np.asarray(crops[int(row["crop_row"]), :, :, 0], dtype=np.int16)
        diff = np.abs(raw_crop.astype(np.int16) - saved)
        crop_checks.append(
            {
                "notehead_id": str(row["notehead_id"]),
                "max_abs_diff": int(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        )

    sample_audio = audio[np.arange(min(5, len(audio)))]
    task1_sizes = pairs_system.groupby("query_id").size()
    task1_positives = pairs_system.groupby("query_id")["label"].sum()
    task2_groups = task2.groupby("source_window_id")["class_id"].nunique()

    warnings = []
    norm_stats = json.loads((proc / "norm_stats_note.json").read_text(encoding="utf-8"))
    reported_frames = int(norm_stats.get("audio", {}).get("segment_frames", -1))
    actual_frames = int(windows.iloc[0]["end_frame"] - windows.iloc[0]["start_frame"])
    if reported_frames != actual_frames:
        warnings.append(
            f"norm_stats_note.json audio.segment_frames={reported_frames}, "
            f"tetapi artifact window={actual_frames} frame; field ini legacy dan tidak dipakai loader"
        )

    return {
        "split": split,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "noteheads": int(len(noteheads)),
            "windows": int(len(windows)),
            "pairs_note": int(len(pairs_note)),
            "pages": int(len(pages)),
            "systems": int(len(systems)),
            "assignments": int(len(assignments)),
            "pairs_system": int(len(pairs_system)),
            "task2_windows": int(len(task2)),
        },
        "checks": {
            "crop_matches_raw": crop_checks,
            "audio_selected_finite": bool(np.isfinite(sample_audio.astype(np.float32)).all()),
            "task1_query_size_all_5": bool((task1_sizes == 5).all()),
            "task1_one_positive_per_query": bool((task1_positives == 1).all()),
            "task2_four_classes_per_source": bool((task2_groups == 4).all()),
            "task2_class_counts": {
                str(name): int(count)
                for name, count in task2["class_name"].value_counts().sort_index().items()
            },
        },
        "figures": files,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual QA untuk artefak preprocessing MSMD")
    parser.add_argument("--split", default="val", choices=SPLITS)
    parser.add_argument("--out-dir", default="reports/qa_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    proc = ROOT / "data" / "processed"
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    figures: list[str] = []

    page_tensor = load_npy(proc / f"score_pages_{args.split}.npy")
    figures.append(figure_notehead_page(args.split, proc, out, page_tensor))
    figures.append(figure_notehead_crops(args.split, proc, out, args.seed, args.samples))
    figures.append(figure_audio_windows(args.split, proc, out, args.seed, args.samples))
    figures.append(figure_task1_overlay(args.split, proc, out, page_tensor))
    figures.append(figure_task1_crops(args.split, proc, out))
    figures.append(figure_task2_grid(args.split, proc, out))
    figures.append(figure_task2_diffs(args.split, proc, out))

    summary = build_summary(args.split, proc, out, args.seed, figures, issues)
    summary_path = out / "qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "split": args.split, "out_dir": str(out), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
