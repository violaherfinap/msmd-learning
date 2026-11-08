"""MuNG notehead parsing and score crop helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import long_path


def parse_mung_noteheads(mung_path: str | Path) -> list[dict[str, Any]]:
    tree = ET.parse(long_path(mung_path))
    noteheads: list[dict[str, Any]] = []
    for obj in tree.getroot().iter("CropObject"):
        if obj.findtext("ClassName") != "notehead-full":
            continue
        data = {item.get("key"): item.text for item in obj.iter("DataItem")}
        pitch_raw = data.get("midi_pitch_code")
        if pitch_raw in (None, ""):
            continue
        top = int(obj.findtext("Top"))
        left = int(obj.findtext("Left"))
        width = int(obj.findtext("Width"))
        height = int(obj.findtext("Height"))
        noteheads.append(
            {
                "mung_id": obj.findtext("Id"),
                "top": top,
                "left": left,
                "width": width,
                "height": height,
                "cx": left + width / 2.0,
                "cy": top + height / 2.0,
                "pitch": int(float(pitch_raw)),
                "tied": int(float(data.get("tied", 0) or 0)),
                "data": data,
            }
        )
    return noteheads


def load_gray(path: str | Path) -> np.ndarray:
    image = Image.open(long_path(path)).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(background, image).convert("L"), dtype=np.uint8)


def crop_notehead(
    page_gray: np.ndarray,
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = page_gray.shape
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(width, x1), min(height, y1)
    crop = page_gray[ys:ye, xs:xe]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=255,
        )

    if crop.shape != (crop_h, crop_w):
        output = np.full((crop_h, crop_w), 255, dtype=np.uint8)
        out_h, out_w = min(crop_h, crop.shape[0]), min(crop_w, crop.shape[1])
        output[:out_h, :out_w] = crop[:out_h, :out_w]
        crop = output

    pad_frac = (pad_left + pad_top + pad_right + pad_bottom) / float(crop_w + crop_h)
    return crop, {
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "pad_frac": pad_frac,
    }


def load_systems(score_dir: str | Path, page_stem: str) -> np.ndarray:
    path = Path(score_dir) / "coords" / f"systems_{page_stem}.npy"
    if not path.is_file():
        return np.zeros((0, 4, 2), dtype=np.float64)
    systems = np.load(long_path(path))
    if systems.ndim != 3 or systems.shape[1:] != (4, 2):
        raise ValueError(f"Unexpected systems shape: {path} -> {systems.shape}")
    return systems.astype(np.float64)
