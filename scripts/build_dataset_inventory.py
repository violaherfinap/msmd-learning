"""Build a complete inventory of the raw MSMD dataset.

The inventory distinguishes pieces with no performances from pieces whose
performances exist but do not have the five expected feature files.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FEATURE_SUFFIXES = [
    "spec",
    "midi",
    "notes",
    "onsets",
    "durations",
]

FEATURE_FILES = {name: f".flac_{name}.npy" for name in FEATURE_SUFFIXES}


def long_path(path: str | Path) -> str:
    """Return a Windows long-path-safe path when needed."""
    value = os.path.abspath(path)
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def list_dirs(path: str | Path) -> list[str]:
    if not os.path.isdir(long_path(path)):
        return []
    with os.scandir(long_path(path)) as entries:
        return sorted(entry.name for entry in entries if entry.is_dir())


def has_mung(piece_dir: Path) -> bool:
    scores_dir = piece_dir / "scores"
    if not scores_dir.is_dir():
        return False
    for score_dir in list_dirs(scores_dir):
        mung_dir = scores_dir / score_dir / "mung"
        if mung_dir.is_dir() and list(mung_dir.glob("*.xml")):
            return True
    return False


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "msmd_aug_v1-1_no-audio"
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw dataset not found: {raw_dir}")

    performance_rows: list[dict] = []
    piece_rows: list[dict] = []

    for piece in list_dirs(raw_dir):
        piece_dir = raw_dir / piece
        performances_dir = piece_dir / "performances"
        performances = list_dirs(performances_dir)

        complete = 0
        incomplete = 0
        for perf_name in performances:
            features_dir = performances_dir / perf_name / "features"
            present = {}
            for name, suffix in FEATURE_FILES.items():
                feature_path = features_dir / f"{perf_name}{suffix}"
                exists = os.path.isfile(long_path(feature_path))
                nonempty = exists and os.path.getsize(long_path(feature_path)) > 0
                present[name] = bool(nonempty)

            n_present = sum(present.values())
            if n_present == len(FEATURE_FILES):
                complete += 1
                status = "complete"
            elif n_present == 0:
                incomplete += 1
                status = "missing_all_features"
            else:
                incomplete += 1
                status = "partial_features"

            performance_rows.append(
                {
                    "piece": piece,
                    "perf_name": perf_name,
                    "features_dir_exists": features_dir.is_dir(),
                    **{f"{name}_present": value for name, value in present.items()},
                    "n_features_present": int(n_present),
                    "status": status,
                }
            )

        n_performances = len(performances)
        if n_performances == 0:
            piece_status = "empty_performances"
        elif complete == n_performances:
            piece_status = "complete"
        elif complete == 0:
            piece_status = "missing_all_features"
        else:
            piece_status = "partial_features"

        piece_rows.append(
            {
                "piece": piece,
                "n_performances": n_performances,
                "n_complete_performances": complete,
                "n_incomplete_performances": incomplete,
                "has_scores": (piece_dir / "scores").is_dir(),
                "has_mung": has_mung(piece_dir),
                "status": piece_status,
            }
        )

    performances_df = pd.DataFrame(performance_rows)
    pieces_df = pd.DataFrame(piece_rows)

    performances_path = out_dir / "dataset_performances.parquet"
    pieces_path = out_dir / "dataset_inventory.parquet"
    performances_df.to_parquet(performances_path, index=False)
    pieces_df.to_parquet(pieces_path, index=False)

    status_counts = pieces_df["status"].value_counts().to_dict()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "feature_files": FEATURE_FILES,
        "total_pieces": int(len(pieces_df)),
        "total_performances": int(len(performances_df)),
        "piece_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "complete_pieces": int((pieces_df["status"] == "complete").sum()),
        "empty_performance_pieces": int(
            (pieces_df["status"] == "empty_performances").sum()
        ),
        "feature_incomplete_pieces": int(
            (pieces_df["status"].isin(["missing_all_features", "partial_features"])).sum()
        ),
        "complete_performances": int(
            (performances_df["status"] == "complete").sum()
        ),
        "incomplete_performances": int(
            (performances_df["status"] != "complete").sum()
        ),
        "outputs": {
            "pieces": str(pieces_path.relative_to(repo_root)),
            "performances": str(performances_path.relative_to(repo_root)),
        },
    }

    summary_path = out_dir / "dataset_inventory.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
