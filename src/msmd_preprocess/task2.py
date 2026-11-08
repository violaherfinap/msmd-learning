"""Task 2 preprocessing: synthesize 4-class performance errors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SPLIT_OFFSET, SPLITS
from .dataset import features_dir
from .io import NpyAppendWriter, write_json, write_parquet
from .paths import long_path
from .templates import (
    METHOD_MISSING,
    METHOD_NAMES,
    build_templates,
)

LOGGER = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "clean",
    1: "wrong_pitch",
    2: "delete",
    3: "insert",
}
CLASS_IDS = {name: index for index, name in CLASS_NAMES.items()}


def load_notes(
    raw_dir: Path,
    piece: str,
    perf_name: str,
    cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (piece, perf_name)
    if key not in cache:
        path = features_dir(raw_dir, piece, perf_name) / f"{perf_name}.flac_notes.npy"
        cache[key] = np.load(long_path(path))
    return cache[key]


def attenuate(
    patch: np.ndarray,
    onset_pos: int,
    duration_frames: int,
    factor: float,
) -> None:
    start = max(0, min(patch.shape[0], int(onset_pos)))
    if start >= patch.shape[0]:
        return
    end = max(start + 1, min(patch.shape[0], start + max(1, int(duration_frames))))
    patch[start:end] *= factor


def add_centered(
    patch: np.ndarray,
    template: np.ndarray,
    center: int,
    scale: float = 1.0,
) -> None:
    half = template.shape[0] // 2
    start = int(center) - half
    src_start = max(0, -start)
    dst_start = max(0, start)
    length = min(template.shape[0] - src_start, patch.shape[0] - dst_start)
    if length > 0:
        patch[dst_start : dst_start + length] += (
            scale * template[src_start : src_start + length]
        )


def add_attack(
    patch: np.ndarray,
    template: np.ndarray,
    start: int,
    scale: float,
    decay_frames: int,
) -> int:
    half = template.shape[0] // 2
    attack = template[half : half + 12]
    if len(attack) == 0:
        return 0
    envelope = np.exp(-np.arange(len(attack), dtype=np.float32) / max(1.0, decay_frames))
    attack = attack * envelope[:, None]
    dst_start = max(0, int(start))
    length = min(len(attack), patch.shape[0] - dst_start)
    if length <= 0:
        return 0
    patch[dst_start : dst_start + length] += scale * attack[:length]
    return length


def energy(patch: np.ndarray) -> float:
    values = patch.astype(np.float64)
    return float((values * values).sum())


def choose_target(
    rng: np.random.Generator,
    available: np.ndarray,
    source_pitch: int,
    template_bank: np.ndarray,
    nearby_radius: int = 12,
) -> int | None:
    candidates = [
        int(pitch)
        for pitch in available
        if int(pitch) != int(source_pitch)
        and float(np.abs(template_bank[int(pitch)]).sum()) > 1e-4
    ]
    if not candidates:
        return None
    nearby = [pitch for pitch in candidates if abs(pitch - source_pitch) <= nearby_radius]
    pool = nearby if nearby else candidates
    return int(pool[int(rng.integers(len(pool)))])


def make_variant(
    rng: np.random.Generator,
    class_name: str,
    clean: np.ndarray,
    template_bank: np.ndarray,
    available: np.ndarray,
    source_pitch: int,
    onset_pos: int,
    duration_frames: int,
    decay_frames: int,
    max_retries: int,
) -> tuple[np.ndarray, int, str, str, int]:
    if class_name == "clean":
        return clean.copy(), source_pitch, "n/a", "", 0

    if class_name == "delete":
        patch = clean.copy()
        attenuate(patch, onset_pos, duration_frames, 0.05)
        if energy(patch) > 0.85 * energy(clean):
            attenuate(patch, onset_pos, duration_frames, 0.0)
        return patch, source_pitch, "n/a", "", 0

    for _ in range(max(1, max_retries)):
        target = choose_target(rng, available, source_pitch, template_bank)
        if target is None:
            continue
        method = "bank"
        patch = clean.copy()
        if class_name == "wrong_pitch":
            attenuate(patch, onset_pos, duration_frames, 0.1)
            add_centered(patch, template_bank[target], onset_pos, scale=1.0)
            if float(np.mean(np.abs(patch - clean))) > 1e-3:
                return patch, target, method, "", 0
        elif class_name == "insert":
            offset = int(rng.integers(4, 13))
            placed = add_attack(
                patch,
                template_bank[target],
                onset_pos + offset,
                scale=0.9,
                decay_frames=decay_frames,
            )
            if placed and energy(patch) > 1.02 * energy(clean):
                return patch, target, method, "", offset

    fallback = clean.copy()
    if class_name == "wrong_pitch":
        target = choose_target(rng, available, source_pitch, template_bank)
        if target is not None:
            fallback = template_bank[target].copy()
            return fallback, target, "fallback_template", "fallback", 0
    elif class_name == "insert":
        target = choose_target(rng, available, source_pitch, template_bank)
        if target is not None:
            fallback = clean.copy()
            add_attack(
                fallback,
                template_bank[target],
                onset_pos,
                scale=1.0,
                decay_frames=decay_frames,
            )
            return fallback, target, "fallback_overlap", "fallback", 0
    target = choose_target(rng, available, source_pitch, template_bank)
    return fallback, int(target) if target is not None else -1, "fallback_none", "skip", 0


def build_split(
    config: dict[str, Any],
    split: str,
    templates: dict[str, np.ndarray],
    raw_dir: Path,
    proc_dir: Path,
) -> dict[str, Any]:
    fps = int(config["fps"])
    win_frames = int(config["win_frames"])
    spec_bins = int(config["spec_bins"])
    max_retries = int(config["task2"]["max_retries"])
    max_per_class = int(config["task2"]["max_per_class"])
    decay_frames = int(config["task2"]["window_decay_frames"])

    windows = pd.read_parquet(proc_dir / f"windows_{split}.parquet")
    if max_per_class > 0:
        windows = windows.iloc[:max_per_class].copy()
    audio = np.load(
        long_path(proc_dir / f"audio_windows_{split}.npy"),
        mmap_mode="r",
    )

    soundfonts = [str(item) for item in templates["soundfonts"].tolist()]
    soundfont_index = {name: index for index, name in enumerate(soundfonts)}
    template_stack = templates["templates"]
    method_stack = templates["methods"]
    available_by_soundfont: dict[int, np.ndarray] = {}
    for index in range(len(soundfonts)):
        methods = method_stack[index]
        available = np.flatnonzero(methods != METHOD_MISSING)
        available = np.array(
            [
                pitch
                for pitch in available
                if float(np.abs(template_stack[index, int(pitch)]).sum()) > 1e-4
            ],
            dtype=int,
        )
        available_by_soundfont[index] = available

    rng = np.random.default_rng(int(config["seed"]) + SPLIT_OFFSET[split] + 200)
    writer = NpyAppendWriter(
        proc_dir / f"error_windows_{split}.npy",
        dtype=np.float16,
        item_shape=(win_frames, spec_bins),
    )
    notes_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    skipped = 0
    fallbacks = 0

    for source_index, window in enumerate(windows.to_dict("records")):
        source_pitch = int(window["pitch"])
        soundfont = str(window["soundfont"])
        s_index = soundfont_index.get(soundfont, 0)
        available = available_by_soundfont[s_index]
        if len(available) == 0:
            skipped += 4
            continue

        clean = np.asarray(audio[int(window["window_row"])], dtype=np.float32)
        onset_pos = int(window["onset_frame"] - window["start_frame"])
        notes = load_notes(
            raw_dir,
            str(window["piece"]),
            str(window["perf_name"]),
            notes_cache,
        )
        note_event_idx = int(window["note_event_idx"])
        duration_s = float(notes[note_event_idx, 2]) if note_event_idx < len(notes) else 0.1
        duration_frames = max(6, int(round(duration_s * fps)))

        for class_id in [0, 1, 2, 3]:
            class_name = CLASS_NAMES[class_id]
            variant, target_pitch, method, fallback, insert_offset = make_variant(
                rng,
                class_name,
                clean,
                template_stack[s_index],
                available,
                source_pitch,
                onset_pos,
                duration_frames,
                decay_frames,
                max_retries,
            )
            if fallback == "skip":
                skipped += 1
                continue
            if fallback:
                fallbacks += 1
            variant = np.clip(variant, -5.0, 5.0).astype(np.float16)
            error_row = writer.append(variant)
            if class_name == "clean":
                error_id = f"{window['window_id']}__clean"
            else:
                error_id = f"{window['window_id']}__{class_name}__p{target_pitch}"
            rows.append(
                {
                    "error_id": error_id,
                    "source_window_id": window["window_id"],
                    "source_window_row": int(window["window_row"]),
                    "source_index": int(source_index),
                    "notehead_id": window["notehead_id"],
                    "page_id": window["page_id"],
                    "piece": window["piece"],
                    "composer": window["composer"],
                    "work_group": window["work_group"],
                    "split": split,
                    "perf_name": window["perf_name"],
                    "soundfont": soundfont,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "pitch_source": source_pitch,
                    "pitch_target": int(target_pitch),
                    "pitch_delta": int(target_pitch - source_pitch)
                    if target_pitch >= 0
                    else 0,
                    "template_method": method,
                    "fallback": bool(fallback),
                    "onset_pos": int(onset_pos),
                    "duration_frames": int(duration_frames),
                    "insert_offset": int(insert_offset),
                    "error_row": int(error_row),
                }
            )

    n_error_windows = writer.finalize()
    if n_error_windows != len(rows):
        raise RuntimeError(
            f"Task 2 tensor/table mismatch for {split}: "
            f"{n_error_windows}/{len(rows)}"
        )
    frame = pd.DataFrame(rows)
    write_parquet(frame, proc_dir / f"task2_windows_{split}.parquet")

    class_counts = {
        CLASS_NAMES[class_id]: int((frame["class_id"] == class_id).sum())
        for class_id in CLASS_NAMES
    }
    method_counts = (
        frame["template_method"].value_counts().to_dict() if len(frame) else {}
    )
    return {
        "split": split,
        "n_source_windows": int(len(windows)),
        "n_error_windows": int(n_error_windows),
        "n_skipped": int(skipped),
        "n_fallbacks": int(fallbacks),
        "class_counts": class_counts,
        "method_counts": {str(k): int(v) for k, v in method_counts.items()},
        "n_soundfonts": int(len(soundfonts)),
    }


def run_task2(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_root"])
    raw_dir = root / config["raw_dir"]
    proc_dir = root / config["processed_dir"]

    template_summary = build_templates(config)
    templates_path = proc_dir / "templates_task2.npz"
    with np.load(templates_path, allow_pickle=False) as loaded:
        templates = {key: loaded[key] for key in loaded.files}

    summaries: dict[str, Any] = {}
    for split in SPLITS:
        summaries[split] = build_split(
            config, split, templates, raw_dir, proc_dir
        )
        LOGGER.info("Task 2 %s: %s", split, summaries[split])

    summary = {
        "templates": template_summary,
        "class_names": CLASS_NAMES,
        "splits": summaries,
        "total_error_windows": int(
            sum(item["n_error_windows"] for item in summaries.values())
        ),
        "total_skipped": int(sum(item["n_skipped"] for item in summaries.values())),
        "total_fallbacks": int(sum(item["n_fallbacks"] for item in summaries.values())),
    }
    write_json(proc_dir / "preprocess_summary_task2.json", summary)
    write_json(
        proc_dir / "task2_class_map.json",
        {
            "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
            "source": "synthetic_errors_from_clean_windows",
        },
    )
    return summary
