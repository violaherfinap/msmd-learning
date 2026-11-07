#!/usr/bin/env python3
"""Validate note-level preprocessing artifacts and write a manifest.

This script is deliberately independent from the notebooks. It reads the
already-generated artifacts in ``data/processed`` and produces:

- data/processed/checks_note_level_extended.csv
- data/processed/artifact_manifest.json
- data/processed/environment_note.json

It exits with status 1 when any hard check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import __version__ as PIL_VERSION

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PROC = DATA / "processed"
REPORTS = PROJECT / "reports"

SPLITS = ("train", "val", "test")
CROP_H = 200
CROP_W = 160
WIN_FRAMES = 42
SPEC_BINS = 92
FPS = 20
SEED = 42


def add_check(checks: list[dict], category: str, name: str, ok: bool, detail="") -> None:
    checks.append(
        {
            "category": category,
            "check": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "detail": str(detail) if detail != "" else "",
        }
    )


def add_skip(checks: list[dict], category: str, name: str, detail="") -> None:
    checks.append(
        {
            "category": category,
            "check": name,
            "status": "SKIP",
            "detail": str(detail),
        }
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_tensor_stats(split: str, crops, wins, checks: list[dict]) -> None:
    for start in range(0, len(crops), 2000):
        chunk = crops[start : start + 2000]
        ink = (chunk < 250).mean(axis=(1, 2, 3))
        if len(ink) and float(ink.min()) <= 0.001:
            add_check(checks, "image", f"{split}: no all-white crop", False, f"min ink={ink.min():.4f}")
            break
    else:
        add_check(checks, "image", f"{split}: no all-white crop", True)

    for start in range(0, len(wins), 2000):
        chunk = wins[start : start + 2000].astype(np.float32)
        finite = bool(np.isfinite(chunk).all())
        nonzero = bool((np.abs(chunk).max(axis=(1, 2)) > 0).all())
        if not finite or not nonzero:
            add_check(
                checks,
                "audio",
                f"{split}: audio finite and non-zero",
                False,
                f"finite={finite}, nonzero={nonzero}",
            )
            break
    else:
        add_check(checks, "audio", f"{split}: audio finite and non-zero", True)


def check_split(split: str, bundle: dict, checks: list[dict], check_raw_paths: bool) -> dict:
    crops = np.load(PROC / f"score_crops_{split}.npy", mmap_mode="r")
    wins = np.load(PROC / f"audio_windows_{split}.npy", mmap_mode="r")
    nh = pd.read_csv(PROC / f"noteheads_{split}.csv")
    wd = pd.read_csv(PROC / f"windows_{split}.csv")
    pr = pd.read_csv(PROC / f"pairs_note_{split}.csv")

    add_check(checks, "shape", f"{split}: crop rows = tensor", len(crops) == len(nh), f"{len(crops)} vs {len(nh)}")
    add_check(checks, "shape", f"{split}: window rows = tensor", len(wins) == len(wd), f"{len(wins)} vs {len(wd)}")
    add_check(checks, "shape", f"{split}: crop shape", tuple(crops.shape[1:]) == (CROP_H, CROP_W, 1), crops.shape[1:])
    add_check(checks, "shape", f"{split}: window shape", tuple(wins.shape[1:]) == (WIN_FRAMES, SPEC_BINS), wins.shape[1:])
    add_check(checks, "shape", f"{split}: crop dtype", crops.dtype == np.uint8, crops.dtype)
    add_check(checks, "shape", f"{split}: window dtype", wins.dtype == np.float16, wins.dtype)

    add_check(checks, "index", f"{split}: notehead_id unik", nh["notehead_id"].is_unique)
    add_check(checks, "index", f"{split}: window_id unik", wd["window_id"].is_unique)
    add_check(checks, "index", f"{split}: pair_id unik", pr["pair_id"].is_unique)
    add_check(
        checks,
        "index",
        f"{split}: crop_row kontigu",
        bool((nh["crop_row"].to_numpy() == np.arange(len(nh))).all()),
    )
    add_check(
        checks,
        "index",
        f"{split}: window_row kontigu",
        bool((wd["window_row"].to_numpy() == np.arange(len(wd))).all()),
    )
    add_check(
        checks,
        "index",
        f"{split}: page_id pattern",
        bool(nh["page_id"].astype(str).str.match(r"^.+__p\d{2}$").all()),
    )
    page_num_consistent = (
        nh.groupby("piece")["page_id"].nunique() == nh.groupby("piece")["page_num"].nunique()
    ).all()
    add_check(checks, "index", f"{split}: page_id/page_num konsisten", bool(page_num_consistent))
    add_check(
        checks,
        "index",
        f"{split}: mung_id unik per page",
        bool((nh.groupby("page_id")["mung_id"].nunique() == nh.groupby("page_id").size()).all()),
    )
    add_check(
        checks,
        "index",
        f"{split}: png_path konsisten per page",
        bool((nh.groupby("page_id")["png_path"].nunique() == 1).all()),
    )
    add_check(
        checks,
        "index",
        f"{split}: mung_path konsisten per page",
        bool((nh.groupby("page_id")["mung_path"].nunique() == 1).all()),
    )

    add_check(checks, "metadata", f"{split}: label hanya 0/1", set(pr["label"].unique()) <= {0, 1}, sorted(pr["label"].unique().tolist()))
    add_check(checks, "metadata", f"{split}: split konsisten", set(nh["split"].unique()) == {split} and set(wd["split"].unique()) == {split} and set(pr["split"].unique()) == {split})
    add_check(checks, "metadata", f"{split}: pitch 0..127", bool(nh["pitch"].between(0, 127).all()), f"{nh['pitch'].min()}..{nh['pitch'].max()}")
    add_check(checks, "metadata", f"{split}: tied 0/1", set(nh["tied"].unique()) <= {0, 1}, sorted(nh["tied"].unique().tolist()))
    add_check(checks, "metadata", f"{split}: edge 0/1", set(nh["edge"].unique()) <= {0, 1}, sorted(nh["edge"].unique().tolist()))
    add_check(checks, "metadata", f"{split}: pad_frac 0..1", bool(nh["pad_frac"].between(0, 1).all()))
    add_check(checks, "metadata", f"{split}: ink_ratio 0..1", bool(nh["ink_ratio"].between(0, 1).all()))
    add_check(checks, "metadata", f"{split}: page_num > 0", bool((nh["page_num"] > 0).all()))

    add_check(
        checks,
        "window",
        f"{split}: end_frame - start_frame == {WIN_FRAMES}",
        bool((wd["end_frame"] - wd["start_frame"] == WIN_FRAMES).all()),
    )
    expected_onset = np.rint(wd["onset_s"].to_numpy(dtype=float) * FPS).astype(np.int64)
    add_check(
        checks,
        "window",
        f"{split}: onset_frame == round(onset_s * {FPS})",
        bool((wd["onset_frame"].to_numpy() == expected_onset).all()),
    )
    add_check(
        checks,
        "window",
        f"{split}: onset di dalam window",
        bool(((wd["start_frame"] <= wd["onset_frame"]) & (wd["onset_frame"] < wd["end_frame"])).all()),
    )
    add_check(checks, "window", f"{split}: padding nonnegatif", bool(((wd["pad_left"] >= 0) & (wd["pad_right"] >= 0)).all()))
    add_check(checks, "window", f"{split}: n_frames > 0", bool((wd["n_frames"] > 0).all()))
    add_check(
        checks,
        "window",
        f"{split}: window_id pattern",
        bool((wd["window_id"] == wd["notehead_id"] + "__" + wd["perf_name"] + "__w").all()),
    )

    perf_pairs = bundle["perfs"][["piece", "perf_name"]].drop_duplicates()
    wd_perf = wd[["piece", "perf_name"]].merge(perf_pairs, on=["piece", "perf_name"], how="left", indicator=True)
    add_check(
        checks,
        "window",
        f"{split}: perf_name milik piece",
        bool((wd_perf["_merge"] == "both").all()),
        f"missing={int((wd_perf['_merge'] != 'both').sum())}",
    )

    add_check(checks, "pairs", f"{split}: pair notehead valid", bool(pr["notehead_id"].isin(set(nh["notehead_id"])).all()))
    add_check(checks, "pairs", f"{split}: pair window valid", bool(pr["window_id"].isin(set(wd["window_id"])).all()))
    pos = pr[pr["label"] == 1].merge(wd[["window_id", "notehead_id"]], on="window_id", suffixes=("", "_w"))
    add_check(checks, "pairs", f"{split}: positif notehead sama", bool((pos["notehead_id"] == pos["notehead_id_w"]).all()))
    pos_meta = pr[pr["label"] == 1]
    add_check(
        checks,
        "pairs",
        f"{split}: positif piece/work_group sama",
        bool((pos_meta["piece"] == pos_meta["window_piece"]).all() and (pos_meta["work_group"] == pos_meta["window_work_group"]).all()),
    )
    neg_meta = pr[pr["label"] == 0]
    add_check(
        checks,
        "pairs",
        f"{split}: negatif beda piece",
        bool((neg_meta["piece"] != neg_meta["window_piece"]).all()),
    )
    add_check(
        checks,
        "pairs",
        f"{split}: negatif beda work_group",
        bool((neg_meta["work_group"] != neg_meta["window_work_group"]).all()),
    )
    add_check(
        checks,
        "pairs",
        f"{split}: label seimbang",
        abs(int((pr["label"] == 1).sum()) - int((pr["label"] == 0).sum())) <= max(10, int(0.1 * len(pr))),
        f"pos={int((pr['label'] == 1).sum())}, neg={int((pr['label'] == 0).sum())}",
    )

    check_tensor_stats(split, crops, wins, checks)

    if check_raw_paths:
        png_ok = nh["png_path"].map(lambda p: Path(p).exists()).all()
        mung_ok = nh["mung_path"].map(lambda p: Path(p).exists()).all()
        add_check(checks, "raw", f"{split}: png_path exists", bool(png_ok))
        add_check(checks, "raw", f"{split}: mung_path exists", bool(mung_ok))
    else:
        add_skip(checks, "raw", f"{split}: png_path exists", "tidak diminta / dataset mentah tidak tersedia")
        add_skip(checks, "raw", f"{split}: mung_path exists", "tidak diminta / dataset mentah tidak tersedia")

    return {
        "crops": crops,
        "wins": wins,
        "nh": nh,
        "wd": wd,
        "pr": pr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-raw-paths",
        action="store_true",
        help="Also verify that png_path/mung_path in noteheads CSVs exist.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip SHA256 manifest generation (faster).",
    )
    args = parser.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    subset = pd.read_csv(DATA / "subset_pieces.csv")
    perf = pd.read_csv(DATA / "subset_performances.csv")
    summary = read_json(PROC / "preprocess_summary_note.json")
    norm = read_json(PROC / "norm_stats_note.json")
    coverage = pd.read_csv(PROC / "coverage_note_level.csv")
    old_checks = pd.read_csv(PROC / "checks_note_level.csv")

    checks: list[dict] = []
    bundles = {}

    add_check(checks, "summary", "summary: total checks lama 63", int(len(old_checks)) == 63, len(old_checks))
    add_check(checks, "summary", "summary: checks lama semua PASS", bool((old_checks["status"] == "PASS").all()))
    add_check(checks, "summary", "summary: 24 piece note-level", int(len(coverage)) == 24, len(coverage))
    add_check(checks, "summary", "summary: GobbaertsL dikecualikan", "GobbaertsL__Les_Etoiles_dOr_No1__Les_Etoiles_dOr_No1" not in set(coverage["piece"]))
    add_check(checks, "summary", "summary: image mean finite", bool(np.isfinite(norm["image"]["mean"])))
    add_check(checks, "summary", "summary: image std > 0", float(norm["image"]["std"]) > 0, norm["image"]["std"])
    add_check(checks, "summary", "summary: audio mean length 92", len(norm["audio"]["spec_mean"]) == SPEC_BINS, len(norm["audio"]["spec_mean"]))
    add_check(checks, "summary", "summary: audio std length 92", len(norm["audio"]["spec_std"]) == SPEC_BINS, len(norm["audio"]["spec_std"]))
    add_check(checks, "summary", "summary: audio mean finite", bool(np.isfinite(np.asarray(norm["audio"]["spec_mean"], dtype=float)).all()))
    add_check(checks, "summary", "summary: audio std positive", bool((np.asarray(norm["audio"]["spec_std"], dtype=float) > 0).all()))
    add_check(checks, "summary", "summary: crop size konsisten", (summary.get("crop_w"), summary.get("crop_h")) == (CROP_W, CROP_H), (summary.get("crop_w"), summary.get("crop_h")))
    add_check(checks, "summary", "summary: window length konsisten", summary.get("win_frames") == WIN_FRAMES, summary.get("win_frames"))
    add_check(checks, "summary", "summary: fps konsisten", summary.get("fps") == FPS, summary.get("fps"))

    for split in SPLITS:
        bundles[split] = check_split(
            split,
            {"perfs": perf},
            checks,
            check_raw_paths=args.check_raw_paths,
        )

    # Cross-split leakage.
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            wa = set(bundles[a]["nh"]["work_group"])
            wb = set(bundles[b]["nh"]["work_group"])
            add_check(checks, "split", f"split {a} vs {b} disjoint", not (wa & wb), f"overlap={sorted(wa & wb)}")

    # Optional hard-negative files. They are additive and must not be confused
    # with the original balanced pair files.
    allowed_hard_types = {
        "same_piece_same_pitch",
        "same_piece_random",
        "cross_piece_same_pitch",
        "cross_piece_random",
    }
    for split in SPLITS:
        hard_path = PROC / f"pairs_note_hard_{split}.csv"
        if not hard_path.exists():
            continue
        hard = pd.read_csv(hard_path)
        b = bundles[split]
        add_check(checks, "hard_negatives", f"{split}: hard pair_id unik", hard["pair_id"].is_unique)
        add_check(checks, "hard_negatives", f"{split}: hard label 0", bool((hard["label"] == 0).all()))
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard notehead/window valid",
            bool(hard["notehead_id"].isin(set(b["nh"]["notehead_id"])).all() and hard["window_id"].isin(set(b["wd"]["window_id"])).all()),
        )
        hard_with_nh = hard.merge(
            b["wd"][["window_id", "notehead_id"]],
            on="window_id",
            how="left",
            suffixes=("", "_window"),
        )
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard beda notehead",
            bool((hard_with_nh["notehead_id"] != hard_with_nh["notehead_id_window"]).all()),
        )
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard neg_type valid",
            set(hard["neg_type"].unique()) <= allowed_hard_types,
            sorted(hard["neg_type"].unique().tolist()),
        )
        same_piece_types = hard["neg_type"].str.startswith("same_piece")
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard same-piece konsisten",
            bool((hard.loc[same_piece_types, "piece"] == hard.loc[same_piece_types, "window_piece"]).all())
            and bool((hard.loc[same_piece_types, "work_group"] == hard.loc[same_piece_types, "window_work_group"]).all()),
        )
        cross_types = hard["neg_type"].str.startswith("cross_piece")
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard cross-piece konsisten",
            bool((hard.loc[cross_types, "piece"] != hard.loc[cross_types, "window_piece"]).all())
            and bool((hard.loc[cross_types, "work_group"] != hard.loc[cross_types, "window_work_group"]).all()),
        )
        same_pitch_types = hard["neg_type"].str.endswith("same_pitch")
        add_check(
            checks,
            "hard_negatives",
            f"{split}: hard same-pitch konsisten",
            bool((hard.loc[same_pitch_types, "notehead_pitch"] == hard.loc[same_pitch_types, "window_pitch"]).all()),
        )
        positive_pairs = set(zip(b["pr"].loc[b["pr"]["label"] == 1, "notehead_id"], b["pr"].loc[b["pr"]["label"] == 1, "window_id"]))
        hard_pairs = set(zip(hard["notehead_id"], hard["window_id"]))
        add_check(checks, "hard_negatives", f"{split}: hard tidak menyalin positif", not (hard_pairs & positive_pairs))
        add_check(checks, "hard_negatives", f"{split}: hard count = notehead", len(hard) == len(b["nh"]), f"{len(hard)} vs {len(b['nh'])}")

    # Counts against summary.
    for split in SPLITS:
        b = bundles[split]
        add_check(checks, "summary", f"{split}: crop count = summary", len(b["crops"]) == int(summary["split_crops"][split]), f"{len(b['crops'])} vs {summary['split_crops'][split]}")
        add_check(checks, "summary", f"{split}: window count = summary", len(b["wins"]) == int(summary["split_windows"][split]), f"{len(b['wins'])} vs {summary['split_windows'][split]}")
        add_check(checks, "summary", f"{split}: pair count = summary", len(b["pr"]) == int(summary["split_pairs"][split]), f"{len(b['pr'])} vs {summary['split_pairs'][split]}")
        cov = coverage[coverage["split"] == split]
        add_check(checks, "summary", f"{split}: coverage pieces = noteheads pieces", len(cov) == b["nh"]["piece"].nunique(), f"{len(cov)} vs {b['nh']['piece'].nunique()}")
        add_check(checks, "summary", f"{split}: coverage mapped = noteheads CSV", int(cov["n_mapped_noteheads"].sum()) == len(b["nh"]), f"{int(cov['n_mapped_noteheads'].sum())} vs {len(b['nh'])}")
        add_check(checks, "summary", f"{split}: coverage windows = windows CSV", int(cov["n_valid_windows"].sum()) == len(b["wd"]), f"{int(cov['n_valid_windows'].sum())} vs {len(b['wd'])}")
        add_check(checks, "summary", f"{split}: coverage total >= mapped", bool((cov["n_noteheads"] >= cov["n_mapped_noteheads"]).all()))

    # Coverage formula sanity.
    expected_pct = coverage["n_mapped_noteheads"] / coverage["n_noteheads"] * 100.0
    add_check(
        checks,
        "summary",
        "coverage: pct_mapped formula",
        bool(np.allclose(coverage["pct_mapped"].to_numpy(dtype=float), expected_pct.to_numpy(dtype=float), atol=1e-6)),
    )
    add_check(checks, "summary", "coverage: all 0..100", bool(coverage["pct_mapped"].between(0, 100).all()))

    checks_df = pd.DataFrame(checks)
    checks_path = PROC / "checks_note_level_extended.csv"
    checks_df.to_csv(checks_path, index=False)

    # Environment metadata.
    env = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "seed": SEED,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "Pillow": PIL_VERSION,
            "matplotlib": package_version("matplotlib"),
            "nbformat": package_version("nbformat"),
            "nbconvert": package_version("nbconvert"),
        },
        "artifacts": {
            "crop_h": CROP_H,
            "crop_w": CROP_W,
            "win_frames": WIN_FRAMES,
            "spec_bins": SPEC_BINS,
            "fps": FPS,
        },
    }
    with open(PROC / "environment_note.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)

    # Artifact manifest.
    if not args.skip_manifest:
        manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
        for path in sorted(PROC.iterdir()):
            if path.name == "artifact_manifest.json" or not path.is_file():
                continue
            if path.suffix.lower() not in {".npy", ".csv", ".json"}:
                continue
            item = {
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            if path.suffix.lower() == ".npy":
                arr = np.load(path, mmap_mode="r")
                item["shape"] = list(arr.shape)
                item["dtype"] = str(arr.dtype)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                item["rows"] = int(len(df))
                item["columns"] = int(len(df.columns))
            manifest["files"].append(item)
        with open(PROC / "artifact_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    n_fail = int((checks_df["status"] == "FAIL").sum())
    n_pass = int((checks_df["status"] == "PASS").sum())
    n_skip = int((checks_df["status"] == "SKIP").sum())
    print(f"Extended checks: {len(checks_df)} total | PASS={n_pass} | SKIP={n_skip} | FAIL={n_fail}")
    print("Saved:", checks_path)
    print("Saved:", PROC / "environment_note.json")
    if not args.skip_manifest:
        print("Saved:", PROC / "artifact_manifest.json")

    if n_fail:
        print("\nFAILED CHECKS:")
        print(checks_df[checks_df["status"] == "FAIL"].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
