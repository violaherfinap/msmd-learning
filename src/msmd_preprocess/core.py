"""Core note-level preprocessing: crops, audio windows, and retrieval pairs."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SPLIT_OFFSET, SPLITS
from .dataset import choose_perfs, features_dir, page_items
from .io import NpyAppendWriter, write_json, write_parquet
from .mung import crop_notehead, load_gray, parse_mung_noteheads
from .paths import long_path, relative_posix

LOGGER = logging.getLogger(__name__)


def load_norm_stats(config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    root = Path(config["_root"])
    with (root / config["norm_stats"]).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    spec_mean = np.asarray(stats["audio"]["spec_mean"], dtype=np.float32)
    spec_std = np.asarray(stats["audio"]["spec_std"], dtype=np.float32)
    if spec_mean.shape != (config["spec_bins"],) or spec_std.shape != (config["spec_bins"],):
        raise ValueError("norm_stats audio arrays do not match spec_bins")
    return stats, spec_mean, spec_std


def load_performance(
    raw_dir: str | Path,
    piece: str,
    perf_name: str,
    spec_mean: np.ndarray,
    spec_std: np.ndarray,
) -> dict[str, Any]:
    fdir = features_dir(raw_dir, piece, perf_name)
    notes = np.load(long_path(fdir / f"{perf_name}.flac_notes.npy"))
    spec = np.load(long_path(fdir / f"{perf_name}.flac_spec.npy")).T.astype(np.float32)
    midi = np.load(long_path(fdir / f"{perf_name}.flac_midi.npy")).astype(np.float32)
    if spec.ndim != 2 or midi.ndim != 2:
        raise ValueError(f"Unexpected feature shapes for {perf_name}")
    spec_t = int(spec.shape[0])
    midi_t = int(midi.shape[1])
    aligned_t = min(spec_t, midi_t)
    spec = spec[:aligned_t]
    midi = midi[:, :aligned_t]
    spec_norm = np.clip((spec - spec_mean) / spec_std, -5, 5)
    return {
        "notes": notes,
        "spec_raw": spec,
        "spec_norm": spec_norm,
        "midi": midi,
        "spec_T": spec_t,
        "midi_T": midi_t,
        "aligned_T": aligned_t,
    }


def extract_window(
    spec_norm: np.ndarray,
    onset_frame: int,
    win_frames: int,
    spec_bins: int,
) -> tuple[np.ndarray, dict[str, int]]:
    n_frames = int(spec_norm.shape[0])
    start = int(onset_frame) - win_frames // 2
    end = start + win_frames
    out = np.zeros((win_frames, spec_bins), dtype=np.float16)
    x0, x1 = max(0, start), min(n_frames, end)
    if x1 > x0:
        offset = x0 - start
        out[offset : offset + (x1 - x0)] = spec_norm[x0:x1].astype(np.float16)
    return out, {
        "start_frame": int(start),
        "end_frame": int(end),
        "pad_left": int(max(0, -start)),
        "pad_right": int(max(0, end - n_frames)),
        "n_frames": n_frames,
    }


def build_pairs(
    split: str,
    noteheads: pd.DataFrame,
    windows: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + SPLIT_OFFSET[split])
    meta = noteheads[["piece", "composer", "work_group"]].drop_duplicates().set_index("piece")
    all_pieces = list(meta.index)
    wins_by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows.to_dict("records"):
        wins_by_piece[row["piece"]].append(row)

    rows: list[dict[str, Any]] = []
    for window in windows.to_dict("records"):
        pos_piece = window["piece"]
        pos_work_group = window["work_group"]
        pos_composer = window["composer"]

        rows.append(
            {
                "notehead_id": window["notehead_id"],
                "window_id": window["window_id"],
                "label": 1,
                "piece": pos_piece,
                "work_group": pos_work_group,
                "split": split,
                "neg_type": "positive",
                "window_piece": pos_piece,
                "window_work_group": pos_work_group,
            }
        )

        neg_pieces = [
            piece
            for piece in all_pieces
            if piece != pos_piece and meta.loc[piece, "work_group"] != pos_work_group
        ]
        same_composer = [
            piece for piece in neg_pieces if meta.loc[piece, "composer"] == pos_composer
        ]
        if same_composer and rng.random() < 0.5:
            pool = same_composer
            neg_type = "same_composer"
        else:
            pool = neg_pieces
            neg_type = "random"
        if not pool:
            pool = neg_pieces
            neg_type = "random"
        if not pool:
            continue

        neg_piece = pool[int(rng.integers(len(pool)))]
        neg_list = wins_by_piece.get(neg_piece, [])
        if not neg_list:
            continue
        neg = neg_list[int(rng.integers(len(neg_list)))]
        rows.append(
            {
                "notehead_id": window["notehead_id"],
                "window_id": neg["window_id"],
                "label": 0,
                "piece": pos_piece,
                "work_group": pos_work_group,
                "split": split,
                "neg_type": neg_type,
                "window_piece": neg_piece,
                "window_work_group": meta.loc[neg_piece, "work_group"],
            }
        )

    frame = pd.DataFrame(rows)
    frame["pair_id"] = [f"{split}_{index:06d}" for index in range(len(frame))]
    return frame[
        [
            "pair_id",
            "notehead_id",
            "window_id",
            "label",
            "piece",
            "work_group",
            "split",
            "neg_type",
            "window_piece",
            "window_work_group",
        ]
    ]


def build_split(
    config: dict[str, Any],
    split: str,
    pieces: pd.DataFrame,
    perfs: pd.DataFrame,
    spec_mean: np.ndarray,
    spec_std: np.ndarray,
) -> dict[str, Any]:
    root = Path(config["_root"])
    raw_dir = root / config["raw_dir"]
    proc_dir = root / config["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    fps = int(config["fps"])
    crop_w = int(config["crop_w"])
    crop_h = int(config["crop_h"])
    win_frames = int(config["win_frames"])
    spec_bins = int(config["spec_bins"])

    perfs_by_piece = {
        piece: group.to_dict("records") for piece, group in perfs.groupby("piece")
    }
    piece_meta = pieces.set_index("piece")

    crop_writer = NpyAppendWriter(
        proc_dir / f"score_crops_{split}.npy",
        dtype=np.uint8,
        item_shape=(crop_h, crop_w, 1),
    )
    window_writer = NpyAppendWriter(
        proc_dir / f"audio_windows_{split}.npy",
        dtype=np.float16,
        item_shape=(win_frames, spec_bins),
    )

    crop_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    image_sum = 0.0
    image_sumsq = 0.0
    image_count = 0

    split_pieces = pieces.loc[pieces["split"] == split, "piece"].tolist()
    for piece_index, piece in enumerate(split_pieces, start=1):
        pperfs = choose_perfs(
            perfs_by_piece.get(piece, []),
            split,
            int(config["train_perfs_per_note"]),
        )
        if not pperfs:
            LOGGER.warning("No performances for %s (%s)", piece, split)
            continue

        perf_data: dict[str, dict[str, Any]] = {}
        for perf in pperfs:
            perf_name = str(perf["perf_name"])
            perf_data[perf_name] = load_performance(
                raw_dir, piece, perf_name, spec_mean, spec_std
            )

        pages = page_items(raw_dir, piece)
        n_noteheads = 0
        n_mapped = 0
        n_tied_skipped = 0
        n_missing_event = 0
        n_valid_windows = 0
        n_edge_crops = 0

        for page in pages:
            if page["mung"] is None:
                continue
            page_gray = load_gray(page["png"])
            noteheads = parse_mung_noteheads(page["mung"])
            page_id = f"{piece}__p{page['page_num']:02d}"
            n_noteheads += len(noteheads)

            for notehead in noteheads:
                valid_windows: list[dict[str, Any]] = []
                for perf in pperfs:
                    perf_name = str(perf["perf_name"])
                    data = perf_data[perf_name]
                    notes = data["notes"]
                    idx_raw = notehead["data"].get(f"{perf_name}_note_event_idx")
                    if idx_raw in (None, ""):
                        n_missing_event += 1
                        continue
                    try:
                        note_event_idx = int(float(idx_raw))
                    except (TypeError, ValueError):
                        n_missing_event += 1
                        continue
                    if note_event_idx < 0 or note_event_idx >= len(notes):
                        n_missing_event += 1
                        continue
                    if int(notes[note_event_idx, 1]) != int(notehead["pitch"]):
                        n_missing_event += 1
                        continue

                    onset_raw = notehead["data"].get(f"{perf_name}_onset_seconds")
                    try:
                        onset_s = (
                            float(onset_raw)
                            if onset_raw not in (None, "")
                            else float(notes[note_event_idx, 0])
                        )
                    except (TypeError, ValueError):
                        onset_s = float(notes[note_event_idx, 0])
                    if not np.isfinite(onset_s):
                        n_missing_event += 1
                        continue

                    aligned_t = int(data["aligned_T"])
                    onset_frame = int(round(onset_s * fps))
                    if onset_frame < 0 or onset_frame >= aligned_t:
                        n_missing_event += 1
                        continue

                    window, window_meta = extract_window(
                        data["spec_norm"], onset_frame, win_frames, spec_bins
                    )
                    valid_windows.append(
                        {
                            "perf_name": perf_name,
                            "note_event_idx": note_event_idx,
                            "onset_s": onset_s,
                            "onset_frame": onset_frame,
                            "window": window,
                            "spec_T": int(data["spec_T"]),
                            "midi_T": int(data["midi_T"]),
                            "aligned_T": aligned_t,
                            "soundfont": str(perf.get("soundfont", "")),
                            "tempo": perf.get("tempo"),
                            **window_meta,
                        }
                    )

                if not valid_windows:
                    if int(notehead["tied"]) == 1:
                        n_tied_skipped += 1
                    continue

                crop, pad_info = crop_notehead(
                    page_gray, notehead["cx"], notehead["cy"], crop_w, crop_h
                )
                edge = int(pad_info["pad_frac"] > 0.30)
                n_edge_crops += edge
                n_mapped += 1

                mung_token = re.sub(r"[^A-Za-z0-9]+", "", str(notehead["mung_id"])) or "0"
                notehead_id = f"{page_id}__nh{mung_token}"
                crop_row = crop_writer.append(crop[..., None])

                if split == "train":
                    scaled = crop.astype(np.float64) / 255.0
                    image_sum += float(scaled.sum())
                    image_sumsq += float((scaled * scaled).sum())
                    image_count += int(scaled.size)

                crop_rows.append(
                    {
                        "notehead_id": notehead_id,
                        "page_id": page_id,
                        "piece": piece,
                        "composer": piece_meta.loc[piece, "composer"],
                        "work_group": piece_meta.loc[piece, "work_group"],
                        "split": split,
                        "page_num": int(page["page_num"]),
                        "mung_id": notehead["mung_id"],
                        "cx": notehead["cx"],
                        "cy": notehead["cy"],
                        "box_top": notehead["top"],
                        "box_left": notehead["left"],
                        "box_w": notehead["width"],
                        "box_h": notehead["height"],
                        "pitch": int(notehead["pitch"]),
                        "tied": int(notehead["tied"]),
                        "edge": edge,
                        "pad_frac": float(pad_info["pad_frac"]),
                        "ink_ratio": float((crop < 250).mean()),
                        "png_path": relative_posix(page["png"], root),
                        "mung_path": relative_posix(page["mung"], root),
                        "crop_row": crop_row,
                    }
                )

                for valid in valid_windows:
                    window_id = f"{notehead_id}__{valid['perf_name']}__w"
                    window_row = window_writer.append(valid["window"])
                    window_rows.append(
                        {
                            "window_id": window_id,
                            "notehead_id": notehead_id,
                            "page_id": page_id,
                            "piece": piece,
                            "composer": piece_meta.loc[piece, "composer"],
                            "work_group": piece_meta.loc[piece, "work_group"],
                            "split": split,
                            "perf_name": valid["perf_name"],
                            "soundfont": valid["soundfont"],
                            "tempo": valid["tempo"],
                            "note_event_idx": valid["note_event_idx"],
                            "onset_s": valid["onset_s"],
                            "onset_frame": valid["onset_frame"],
                            "start_frame": valid["start_frame"],
                            "end_frame": valid["end_frame"],
                            "pad_left": valid["pad_left"],
                            "pad_right": valid["pad_right"],
                            "n_frames": valid["n_frames"],
                            "spec_T": valid["spec_T"],
                            "midi_T": valid["midi_T"],
                            "aligned_T": valid["aligned_T"],
                            "pitch": int(notehead["pitch"]),
                            "window_row": window_row,
                        }
                    )
                    n_valid_windows += 1

        coverage_rows.append(
            {
                "piece": piece,
                "split": split,
                "n_pages": len(pages),
                "n_noteheads": n_noteheads,
                "n_mapped_noteheads": n_mapped,
                "n_tied_skipped": n_tied_skipped,
                "n_missing_event": n_missing_event,
                "n_valid_windows": n_valid_windows,
                "n_edge_crops": n_edge_crops,
                "pct_mapped": (100.0 * n_mapped / n_noteheads) if n_noteheads else 0.0,
            }
        )
        LOGGER.info(
            "core %s [%d/%d] %s: crops=%d windows=%d",
            split,
            piece_index,
            len(split_pieces),
            piece,
            n_mapped,
            n_valid_windows,
        )

    n_crops = crop_writer.finalize()
    n_windows = window_writer.finalize()
    if n_crops != len(crop_rows) or n_windows != len(window_rows):
        raise RuntimeError(
            f"Tensor/table row mismatch for {split}: "
            f"crops {n_crops}/{len(crop_rows)}, windows {n_windows}/{len(window_rows)}"
        )

    noteheads = pd.DataFrame(crop_rows)
    windows = pd.DataFrame(window_rows)
    coverage = pd.DataFrame(coverage_rows)
    pairs = build_pairs(split, noteheads, windows, int(config["seed"]))

    write_parquet(noteheads, proc_dir / f"noteheads_{split}.parquet")
    write_parquet(windows, proc_dir / f"windows_{split}.parquet")
    write_parquet(pairs, proc_dir / f"pairs_note_{split}.parquet")
    write_parquet(coverage, proc_dir / f"coverage_{split}.parquet")

    image_mean = image_sum / image_count if image_count else 0.0
    image_var = max(image_sumsq / image_count - image_mean * image_mean, 0.0) if image_count else 0.0

    return {
        "split": split,
        "n_crops": int(n_crops),
        "n_windows": int(n_windows),
        "n_pairs": int(len(pairs)),
        "n_pairs_positive": int((pairs["label"] == 1).sum()),
        "n_pairs_negative": int((pairs["label"] == 0).sum()),
        "n_pieces": int(len(coverage)),
        "image_sum": image_sum,
        "image_sumsq": image_sumsq,
        "image_count": image_count,
        "image_mean": float(image_mean),
        "image_std": float(np.sqrt(image_var)),
    }


def run_core(
    config: dict[str, Any],
    pieces: pd.DataFrame,
    perfs: pd.DataFrame,
    excluded: list[str],
) -> dict[str, Any]:
    root = Path(config["_root"])
    proc_dir = root / config["processed_dir"]
    stats, spec_mean, spec_std = load_norm_stats(config)

    summaries: dict[str, Any] = {}
    for split in SPLITS:
        summaries[split] = build_split(
            config, split, pieces, perfs, spec_mean, spec_std
        )

    train = summaries["train"]
    norm_stats = {
        "image": {
            "crop_w": int(config["crop_w"]),
            "crop_h": int(config["crop_h"]),
            "mean": train["image_mean"],
            "std": train["image_std"],
            "storage": "uint8_raw_grayscale",
        },
        "audio": {
            **stats["audio"],
            "win_frames": int(config["win_frames"]),
            "fps": int(config["fps"]),
            "spec_bins": int(config["spec_bins"]),
            "storage": "float16_normalized_clipped",
        },
        "split": "group_by_work_group",
        "seed": int(config["seed"]),
        "train_perfs_per_note": int(config["train_perfs_per_note"]),
        "alignment": "min(spec_T, midi_T)",
    }
    write_json(proc_dir / "norm_stats_note.json", norm_stats)

    summary = {
        "excluded_pieces": excluded,
        "alignment": "min(spec_T, midi_T)",
        "splits": summaries,
        "total_crops": int(sum(item["n_crops"] for item in summaries.values())),
        "total_windows": int(sum(item["n_windows"] for item in summaries.values())),
        "total_pairs": int(sum(item["n_pairs"] for item in summaries.values())),
    }
    write_json(proc_dir / "preprocess_summary_note.json", summary)
    return summary
