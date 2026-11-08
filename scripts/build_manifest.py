"""Build manifest and SHA256SUMS for preprocessing artifacts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.msmd_preprocess.bundles import artifact_groups
from src.msmd_preprocess.config import SPLITS, load_config
from src.msmd_preprocess.io import artifact_record, write_manifest
from src.msmd_preprocess.paths import long_path

LOGGER = logging.getLogger(__name__)


def npy_info(path: Path) -> tuple[int, tuple[int, ...], str]:
    array = np.load(long_path(path), mmap_mode="r")
    return int(array.shape[0]), tuple(int(x) for x in array.shape), str(array.dtype)


def parquet_rows(path: Path) -> int:
    return int(pd.read_parquet(path).shape[0])


def split_from_name(name: str) -> str | None:
    for split in SPLITS:
        if f"_{split}." in name or name.endswith(f"_{split}"):
            return split
    return None


def collect(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups = artifact_groups(config)
    for stage, paths in groups.items():
        for path in paths:
            if path.name == "manifest_preprocessing.json":
                continue
            split = split_from_name(path.name)
            if path.suffix == ".npy":
                rows, shape, dtype = npy_info(path)
                records.append(
                    artifact_record(
                        path,
                        ROOT,
                        stage,
                        split=split,
                        rows=rows,
                        shape=shape,
                        dtype=dtype,
                    )
                )
            elif path.suffix == ".parquet":
                records.append(
                    artifact_record(
                        path, ROOT, stage, split=split, rows=parquet_rows(path)
                    )
                )
            else:
                records.append(
                    artifact_record(path, ROOT, stage, split=split)
                )
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()
    records = collect(config)
    manifest_path = ROOT / config["processed_dir"] / "manifest_preprocessing.json"
    write_manifest(records, manifest_path, ROOT, config)

    lines = []
    for record in records:
        digest = record.get("sha256")
        if digest:
            lines.append(f"{digest}  {record['path']}")
    sums_path = ROOT / config["processed_dir"] / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(int(record["bytes"]) for record in records)
    print(json.dumps({"artifacts": len(records), "bytes": total}, indent=2))
    for stage in ("core", "task1", "task2", "meta"):
        subset = [record for record in records if record["stage"] == stage]
        print(
            f"{stage}: {len(subset)} artifacts, "
            f"{sum(int(record['bytes']) for record in subset) / 1e9:.3f} GB"
        )


if __name__ == "__main__":
    main()
