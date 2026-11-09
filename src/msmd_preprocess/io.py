"""I/O helpers: atomic writes, Parquet, memory-friendly .npy writing, manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .paths import long_path, relative_posix


def write_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, target)


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    os.replace(tmp, target)


class NpyAppendWriter:
    """Write a large .npy file without keeping all rows in memory."""

    def __init__(
        self,
        final_path: str | Path,
        dtype: str | np.dtype,
        item_shape: tuple[int, ...],
    ) -> None:
        self.final_path = Path(final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.final_path.with_suffix(self.final_path.suffix + ".part")
        self.dtype = np.dtype(dtype)
        self.item_shape = tuple(int(x) for x in item_shape)
        self.handle = self.tmp_path.open("wb")
        self.count = 0

    def append(self, array: np.ndarray) -> int:
        array = np.ascontiguousarray(array, dtype=self.dtype)
        expected = self.item_shape
        if array.shape != expected:
            raise ValueError(f"Expected {expected}, got {array.shape}")
        array.tofile(self.handle)
        row = self.count
        self.count += 1
        return row

    def finalize(self) -> int:
        self.handle.close()
        header = {
            "descr": self.dtype.str,
            "fortran_order": False,
            "shape": (self.count,) + self.item_shape,
        }
        with self.final_path.open("wb") as out:
            np.lib.format.write_array_header_1_0(out, header)
            with self.tmp_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
        self.tmp_path.unlink(missing_ok=True)
        return self.count

    def close(self) -> None:
        self.handle.close()
        self.tmp_path.unlink(missing_ok=True)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: str | Path,
    root: str | Path,
    stage: str,
    split: str | None = None,
    rows: int | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    hash_file: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    record: dict[str, Any] = {
        "path": relative_posix(target, root),
        "stage": stage,
        "split": split,
        "rows": int(rows) if rows is not None else None,
        "shape": [int(x) for x in shape] if shape is not None else None,
        "dtype": dtype,
        "bytes": int(target.stat().st_size),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if hash_file:
        record["sha256"] = sha256_file(target)
    if extra:
        record.update(extra)
    return record


def write_manifest(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    root: str | Path,
    config: dict[str, Any],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config.get("_config_path"),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "artifacts": list(records),
    }
    write_json(path, payload)
