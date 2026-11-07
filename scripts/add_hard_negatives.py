#!/usr/bin/env python3
"""Create additional hard-negative pairs without touching existing artifacts.

The original ``pairs_note_{split}.csv`` files only contain cross-piece
negatives.  That is useful for a first retrieval baseline, but it makes the
task easier than real score-audio matching.  This script adds *separate*
files:

    data/processed/pairs_note_hard_{split}.csv

Each notehead gets at most one hard negative, chosen with this priority:

1. same piece, same pitch, different notehead
2. same piece, different notehead
3. cross piece, same pitch
4. cross piece, different notehead

The original ``pairs_note_*`` files are not modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PROC = PROJECT / "data" / "processed"
SPLITS = ("train", "val", "test")
SEED = 42


def pick(rng: np.random.Generator, items: list[dict]) -> dict:
    return items[int(rng.integers(0, len(items)))]


def main() -> int:
    stats: dict[str, dict] = {}
    for offset, split in enumerate(SPLITS):
        rng = np.random.default_rng(SEED + offset)
        nh = pd.read_csv(PROC / f"noteheads_{split}.csv")
        wd = pd.read_csv(PROC / f"windows_{split}.csv")

        # Keep only columns needed for candidate selection.
        wd_small = wd[["window_id", "notehead_id", "piece", "work_group", "pitch", "perf_name"]].copy()
        by_piece: dict[str, list[dict]] = {
            piece: group.to_dict("records") for piece, group in wd_small.groupby("piece", sort=False)
        }
        by_piece_pitch: dict[tuple[str, int], list[dict]] = {
            (piece, int(pitch)): group.to_dict("records")
            for (piece, pitch), group in wd_small.groupby(["piece", "pitch"], sort=False)
        }
        all_by_pitch: dict[int, list[dict]] = {
            int(pitch): group.to_dict("records") for pitch, group in wd_small.groupby("pitch", sort=False)
        }

        rows: list[dict] = []
        for nh_row in nh.itertuples(index=False):
            piece = str(nh_row.piece)
            notehead_id = str(nh_row.notehead_id)
            pitch = int(nh_row.pitch)

            same_piece = [w for w in by_piece.get(piece, []) if w["notehead_id"] != notehead_id]
            same_piece_same_pitch = [
                w for w in by_piece_pitch.get((piece, pitch), []) if w["notehead_id"] != notehead_id
            ]

            if same_piece_same_pitch:
                chosen = pick(rng, same_piece_same_pitch)
                neg_type = "same_piece_same_pitch"
                priority = 1
            elif same_piece:
                chosen = pick(rng, same_piece)
                neg_type = "same_piece_random"
                priority = 2
            else:
                cross_same_pitch = [w for w in all_by_pitch.get(pitch, []) if w["piece"] != piece]
                if cross_same_pitch:
                    chosen = pick(rng, cross_same_pitch)
                    neg_type = "cross_piece_same_pitch"
                    priority = 3
                else:
                    # Extremely unlikely, but keep the script total.
                    pool = [w for w in wd_small.to_dict("records") if w["piece"] != piece]
                    if not pool:
                        continue
                    chosen = pick(rng, pool)
                    neg_type = "cross_piece_random"
                    priority = 4

            rows.append(
                {
                    "pair_id": f"hard_{split}_{len(rows):06d}",
                    "notehead_id": notehead_id,
                    "window_id": chosen["window_id"],
                    "label": 0,
                    "piece": piece,
                    "work_group": str(nh_row.work_group),
                    "split": split,
                    "neg_type": neg_type,
                    "window_piece": chosen["piece"],
                    "window_work_group": chosen["work_group"],
                    "notehead_pitch": pitch,
                    "window_pitch": int(chosen["pitch"]),
                    "priority": priority,
                }
            )

        out = pd.DataFrame(rows)
        out_path = PROC / f"pairs_note_hard_{split}.csv"
        out.to_csv(out_path, index=False)
        stats[split] = {
            "rows": int(len(out)),
            "neg_type_counts": {str(k): int(v) for k, v in out["neg_type"].value_counts().items()},
            "same_piece": int((out["piece"] == out["window_piece"]).sum()),
            "same_pitch": int((out["notehead_pitch"] == out["window_pitch"]).sum()),
        }
        print(f"{split}: {len(out):,} hard negatives -> {out_path}")

    with open(PROC / "pairs_note_hard_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print("Saved:", PROC / "pairs_note_hard_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
