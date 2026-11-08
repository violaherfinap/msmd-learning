"""Pitch/soundfont template bank for synthetic performance errors."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SPLIT_OFFSET
from .io import write_json
from .paths import long_path

LOGGER = logging.getLogger(__name__)

METHOD_MISSING = 0
METHOD_DIRECT = 1
METHOD_RIDGE = 2
METHOD_NEAREST = 3
METHOD_GLOBAL = 4
METHOD_NAMES = {
    METHOD_MISSING: "missing",
    METHOD_DIRECT: "direct",
    METHOD_RIDGE: "ridge",
    METHOD_NEAREST: "nearest",
    METHOD_GLOBAL: "global",
}


def _poly_features(pitches: np.ndarray) -> np.ndarray:
    values = pitches.astype(np.float64)
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 1.0
    z = (values - mean) / (std if std > 1e-8 else 1.0)
    return np.stack([np.ones_like(z), z, z * z, z * z * z], axis=1)


def _ridge_fit(
    pitches: np.ndarray,
    templates: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, float, float]:
    x = _poly_features(pitches)
    reg = ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    y = templates.reshape(len(templates), -1).astype(np.float64)
    beta = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return beta, float(pitches.mean()), float(pitches.std() if pitches.std() > 1e-8 else 1.0)


def _ridge_predict(
    pitches: np.ndarray,
    beta: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    values = (np.asarray(pitches, dtype=np.float64) - mean) / (std if std > 1e-8 else 1.0)
    x = np.stack([np.ones_like(values), values, values * values, values * values * values], axis=1)
    return x @ beta


def _nearest_template(
    pitch: int,
    available: np.ndarray,
    templates: np.ndarray,
    exclude: int | None = None,
) -> np.ndarray:
    indices = {int(item): index for index, item in enumerate(available)}
    candidates = [int(item) for item in available if item != exclude]
    if not candidates:
        return np.zeros_like(templates[0], dtype=np.float32)
    lower = [item for item in candidates if item < pitch]
    upper = [item for item in candidates if item > pitch]
    if lower and upper:
        lo = max(lower)
        hi = min(upper)
        weight = (pitch - lo) / float(hi - lo)
        return (
            (1.0 - weight) * templates[indices[lo]] + weight * templates[indices[hi]]
        ).astype(np.float32)
    nearest = min(candidates, key=lambda item: (abs(item - pitch), item))
    return templates[indices[nearest]].astype(np.float32)


def _ridge_loocv_mse(
    pitches: np.ndarray,
    templates: np.ndarray,
    ridge_lambda: float,
) -> float:
    if len(pitches) < 5:
        return float("inf")
    errors = []
    for index in range(len(pitches)):
        mask = np.arange(len(pitches)) != index
        beta, mean, std = _ridge_fit(pitches[mask], templates[mask], ridge_lambda)
        predicted = _ridge_predict(pitches[index : index + 1], beta, mean, std)[0]
        errors.append(float(np.mean((predicted - templates[index].reshape(-1)) ** 2)))
    return float(np.mean(errors))


def _nearest_loocv_mse(pitches: np.ndarray, templates: np.ndarray) -> float:
    if len(pitches) < 3:
        return float("inf")
    errors = []
    for index, pitch in enumerate(pitches):
        predicted = _nearest_template(
            int(pitch), pitches, templates, exclude=int(pitch)
        )
        errors.append(float(np.mean((predicted - templates[index]) ** 2)))
    return float(np.mean(errors))


def _fill_missing(
    direct_pitches: np.ndarray,
    direct_templates: np.ndarray,
    target_pitches: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(direct_pitches) < 3:
        return direct_templates[:0], np.zeros((0,), dtype=int), "none"
    ridge_mse = _ridge_loocv_mse(direct_pitches, direct_templates, ridge_lambda)
    nearest_mse = _nearest_loocv_mse(direct_pitches, direct_templates)
    use_ridge = np.isfinite(ridge_mse) and ridge_mse <= nearest_mse
    method = METHOD_RIDGE if use_ridge else METHOD_NEAREST
    if use_ridge:
        beta, mean, std = _ridge_fit(direct_pitches, direct_templates, ridge_lambda)
        filled = _ridge_predict(target_pitches, beta, mean, std)
        filled = np.clip(filled, -5.0, 5.0).reshape(
            len(target_pitches), *direct_templates.shape[1:]
        ).astype(np.float16)
        LOGGER.info(
            "fallback=ridge pitches=%d loo_ridge=%.6g loo_nearest=%.6g",
            len(target_pitches),
            ridge_mse,
            nearest_mse,
        )
    else:
        filled = np.stack(
            [
                _nearest_template(int(pitch), direct_pitches, direct_templates)
                for pitch in target_pitches
            ],
            axis=0,
        ).astype(np.float16)
        LOGGER.info(
            "fallback=nearest pitches=%d loo_ridge=%.6g loo_nearest=%.6g",
            len(target_pitches),
            ridge_mse,
            nearest_mse,
        )
    return filled, np.full(len(target_pitches), method, dtype=np.int8), METHOD_NAMES[method]


def build_templates(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_root"])
    proc_dir = root / config["processed_dir"]
    min_frames = int(config["task2"]["template_min_frames"])
    ridge_lambda = float(config["task2"]["ridge_lambda"])

    windows = pd.read_parquet(proc_dir / "windows_train.parquet")
    audio = np.load(
        long_path(proc_dir / "audio_windows_train.npy"),
        mmap_mode="r",
    )
    usable = windows[(windows["pad_left"] == 0) & (windows["pad_right"] == 0)]
    soundfonts = sorted(windows["soundfont"].dropna().astype(str).unique().tolist())

    global_sums: dict[int, np.ndarray] = {}
    global_counts: dict[int, int] = {}
    for row in usable.to_dict("records"):
        pitch = int(row["pitch"])
        index = int(row["window_row"])
        block = np.asarray(audio[index], dtype=np.float64)
        if pitch not in global_sums:
            global_sums[pitch] = np.zeros_like(block)
            global_counts[pitch] = 0
        global_sums[pitch] += block
        global_counts[pitch] += 1
    global_pitches = np.array(sorted(global_sums), dtype=int)
    global_direct = np.stack(
        [global_sums[pitch] / global_counts[pitch] for pitch in global_pitches],
        axis=0,
    ).astype(np.float16)
    global_direct_mask = np.array(
        [global_counts[pitch] >= min_frames for pitch in global_pitches], dtype=bool
    )
    global_pitches = global_pitches[global_direct_mask]
    global_direct = global_direct[global_direct_mask]
    global_min = int(global_pitches.min())
    global_max = int(global_pitches.max())
    global_pitches_all = np.arange(global_min, global_max + 1, dtype=int)
    global_filled, global_methods, global_method_name = _fill_missing(
        global_pitches, global_direct, global_pitches_all, ridge_lambda
    )
    global_templates = np.zeros((128, *global_direct.shape[1:]), dtype=np.float16)
    global_method = np.zeros((128,), dtype=np.int8)
    global_counts_array = np.zeros((128,), dtype=np.int32)
    global_templates[global_pitches_all] = global_filled
    global_method[global_pitches_all] = global_methods
    global_counts_array[global_pitches] = np.array(
        [global_counts[int(pitch)] for pitch in global_pitches], dtype=np.int32
    )

    templates = np.zeros((len(soundfonts), 128, *global_direct.shape[1:]), dtype=np.float16)
    counts = np.zeros((len(soundfonts), 128), dtype=np.int32)
    methods = np.zeros((len(soundfonts), 128), dtype=np.int8)
    source_pitch = np.full((len(soundfonts), 128), -1, dtype=np.int16)
    coverage: dict[str, Any] = {}

    for s_index, soundfont in enumerate(soundfonts):
        subset = usable[usable["soundfont"].astype(str) == soundfont]
        grouped: dict[int, list[int]] = {}
        for row in subset.to_dict("records"):
            grouped.setdefault(int(row["pitch"]), []).append(int(row["window_row"]))
        direct_pitches = np.array(
            sorted(pitch for pitch, rows in grouped.items() if len(rows) >= min_frames),
            dtype=int,
        )
        if len(direct_pitches):
            direct_templates = np.stack(
                [
                    np.asarray(audio[grouped[int(pitch)]], dtype=np.float64).mean(axis=0)
                    for pitch in direct_pitches
                ],
                axis=0,
            ).astype(np.float16)
        else:
            direct_templates = np.zeros((0, *global_direct.shape[1:]), dtype=np.float16)

        if len(direct_pitches):
            for pitch in direct_pitches:
                templates[s_index, int(pitch)] = direct_templates[
                    int(np.where(direct_pitches == pitch)[0][0])
                ]
                methods[s_index, int(pitch)] = METHOD_DIRECT
                source_pitch[s_index, int(pitch)] = int(pitch)
                counts[s_index, int(pitch)] = len(grouped[int(pitch)])

            pitch_lo = int(direct_pitches.min())
            pitch_hi = int(direct_pitches.max())
            target_pitches = np.array(
                [
                    pitch
                    for pitch in range(pitch_lo, pitch_hi + 1)
                    if pitch not in set(int(item) for item in direct_pitches)
                ],
                dtype=int,
            )
            if len(target_pitches):
                filled, filled_methods, fallback_name = _fill_missing(
                    direct_pitches, direct_templates, target_pitches, ridge_lambda
                )
                templates[s_index, target_pitches] = filled
                methods[s_index, target_pitches] = filled_methods
                source_pitch[s_index, target_pitches] = direct_pitches[
                    np.argmin(np.abs(target_pitches[:, None] - direct_pitches[None, :]), axis=1)
                ]
            else:
                fallback_name = "none"
        else:
            pitch_lo = global_min
            pitch_hi = global_max
            fallback_name = "global"

        for pitch in range(128):
            if methods[s_index, pitch] != METHOD_MISSING:
                continue
            if global_method[pitch] == METHOD_MISSING:
                continue
            templates[s_index, pitch] = global_templates[pitch]
            methods[s_index, pitch] = METHOD_GLOBAL
            source_pitch[s_index, pitch] = -1
            counts[s_index, pitch] = global_counts_array[pitch]

        coverage[soundfont] = {
            "n_direct_pitches": int(len(direct_pitches)),
            "pitch_lo": int(pitch_lo),
            "pitch_hi": int(pitch_hi),
            "fallback": fallback_name,
            "n_usable_windows": int(len(subset)),
            "n_direct": int((methods[s_index] == METHOD_DIRECT).sum()),
            "n_ridge": int((methods[s_index] == METHOD_RIDGE).sum()),
            "n_nearest": int((methods[s_index] == METHOD_NEAREST).sum()),
            "n_global": int((methods[s_index] == METHOD_GLOBAL).sum()),
        }
        LOGGER.info("templates %s: %s", soundfont, coverage[soundfont])

    output = proc_dir / "templates_task2.npz"
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            soundfonts=np.asarray(soundfonts),
            templates=templates,
            counts=counts,
            methods=methods,
            source_pitch=source_pitch,
            global_templates=global_templates,
            global_methods=global_method,
            global_counts=global_counts_array,
        )
    os.replace(tmp, output)

    summary = {
        "min_frames": min_frames,
        "ridge_lambda": ridge_lambda,
        "global_fallback": global_method_name,
        "global_pitch_range": [global_min, global_max],
        "soundfonts": soundfonts,
        "coverage": coverage,
    }
    write_json(proc_dir / "templates_task2.json", summary)
    return summary
