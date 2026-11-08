"""File groups used by the manifest and release bundle builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import SPLITS


def artifact_groups(config: dict[str, Any]) -> dict[str, list[Path]]:
    root = Path(config["_root"])
    proc = root / config["processed_dir"]
    reports = root / config["reports_dir"]

    core = [
        proc / "dataset_inventory.parquet",
        proc / "dataset_performances.parquet",
        proc / "dataset_inventory.json",
    ]
    for split in SPLITS:
        core.extend(
            [
                proc / f"noteheads_{split}.parquet",
                proc / f"windows_{split}.parquet",
                proc / f"pairs_note_{split}.parquet",
                proc / f"coverage_{split}.parquet",
                proc / f"score_crops_{split}.npy",
                proc / f"audio_windows_{split}.npy",
            ]
        )
    core.extend([proc / "norm_stats_note.json", proc / "preprocess_summary_note.json"])

    task1: list[Path] = []
    for split in SPLITS:
        task1.extend(
            [
                proc / f"pages_{split}.parquet",
                proc / f"systems_{split}.parquet",
                proc / f"notehead_systems_{split}.parquet",
                proc / f"pairs_system_{split}.parquet",
                proc / f"score_pages_{split}.npy",
                proc / f"system_crops_{split}.npy",
            ]
        )
    task1.extend([proc / "norm_stats_task1.json", proc / "preprocess_summary_task1.json"])

    task2: list[Path] = []
    for split in SPLITS:
        task2.extend(
            [
                proc / f"task2_windows_{split}.parquet",
                proc / f"error_windows_{split}.npy",
            ]
        )
    task2.extend(
        [
            proc / "templates_task2.npz",
            proc / "templates_task2.json",
            proc / "task2_class_map.json",
            proc / "preprocess_summary_task2.json",
        ]
    )

    meta = [
        root / "data" / "subset_pieces.csv",
        root / "data" / "subset_performances.csv",
        root / "data" / "splits.csv",
        root / "configs" / "preprocess.yaml",
        root / "requirements.txt",
        root / "README.md",
        reports / "validation_preprocessing.json",
        proc / "manifest_preprocessing.json",
    ]

    groups = {"core": core, "task1": task1, "task2": task2, "meta": meta}
    return {
        name: [path for path in paths if path.exists()]
        for name, paths in groups.items()
    }
