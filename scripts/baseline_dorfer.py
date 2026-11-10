"""Shared baseline utilities for Dorfer et al. (2018) audio-sheet retrieval.

Paper
-----
Dorfer, M., Hajic jr., J., Arzt, A., Frostel, H., & Widmer, G. (2018).
Learning Audio-Sheet Music Correspondences for Cross-Modal Retrieval and
Piece Identification. TISMIR, 1(1), 22-33. https://doi.org/10.5334/tismir.12

The implementation follows the official CPJKU code
(https://github.com/CPJKU/audio_sheet_retrieval), in particular
``models/mutopia_ccal_cont_rsz.py``:

- sheet input 160x200 -> 80x100 (resized by factor 2)
- audio input 92x42 (log-frequency bins x frames)
- four VGG-style blocks with 24/48/96/96 filters
- 1x1 conv to 32 channels + batch norm + global average pooling
- running CCA embedding layer + L2 normalization
- pairwise ranking loss (margin 0.7)

This module is intentionally dependency-light: only numpy, pandas, and torch.
Notebooks using it:

- notebooks/03_baseline_task1_dual_encoder.ipynb
- notebooks/04_baseline_task2_multimodal_classifier.ipynb

The notebooks run a **Dorfer-lite** profile by default (3 blocks, 16/32/64
channels, no CCA, 8 epochs) so that training fits in roughly one hour on a
CPU. Set ``channels=(24, 48, 96, 96)`` and ``use_cca=True`` to recover the
paper-sized backbone.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants from the preprocessing artifacts
# ---------------------------------------------------------------------------

IMAGE_MEAN = 0.8936307257819602
IMAGE_STD = 0.26513359410817494
IMAGE_H = 200
IMAGE_W = 160
MODEL_H = 100
MODEL_W = 80
SPEC_BINS = 92
SPEC_FRAMES = 42

CLASS_NAMES = ["correct", "wrong_pitch", "missing_note", "extra_note"]
CLASS_NAMES_SHORT = ["clean", "wrong_pitch", "delete", "insert"]

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Make a run as reproducible as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto") -> torch.device:
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(threads: int | None = None) -> None:
    if threads is not None and int(threads) > 0:
        torch.set_num_threads(int(threads))
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/torch/path objects to plain JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(obj), handle, indent=2, ensure_ascii=False)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """torch.load wrapper that also works with older torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_split_frames(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load notehead, window, and pair metadata for one split."""
    data_dir = Path(data_dir)
    noteheads = pd.read_parquet(data_dir / f"noteheads_{split}.parquet")
    windows = pd.read_parquet(data_dir / f"windows_{split}.parquet")
    pairs = pd.read_parquet(data_dir / f"pairs_note_{split}.parquet")
    return noteheads, windows, pairs


def build_positive_pairs(
    pairs: pd.DataFrame,
    noteheads: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Return positive notehead-window pairs with crop_row/window_row_eval."""
    pos = pairs.loc[pairs["label"] == 1].copy()
    crop_map = noteheads.set_index("notehead_id")["crop_row"]
    window_map = windows.set_index("window_id")["window_row"]
    pos["crop_row"] = pos["notehead_id"].map(crop_map)
    pos["window_row_eval"] = pos["window_id"].map(window_map)
    if pos["crop_row"].isna().any() or pos["window_row_eval"].isna().any():
        raise ValueError("Positive pair refers to a notehead/window missing from metadata.")
    pos["crop_row"] = pos["crop_row"].astype(int)
    pos["window_row_eval"] = pos["window_row_eval"].astype(int)
    return pos


def one_window_per_notehead(pos: pd.DataFrame) -> pd.DataFrame:
    """Deterministic paired view used for paper-style snippet retrieval."""
    return (
        pos.sort_values("window_row_eval")
        .drop_duplicates("notehead_id")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Preprocessing and augmentation
# ---------------------------------------------------------------------------


def preprocess_image_array(
    arr: np.ndarray,
    mean: float = IMAGE_MEAN,
    std: float = IMAGE_STD,
    out_h: int = MODEL_H,
    out_w: int = MODEL_W,
) -> torch.Tensor:
    """Normalize a raw crop and resize to the paper input (100x80)."""
    x = np.asarray(arr)
    if x.ndim == 3:
        x = x[..., 0]
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
        if float(x.max(initial=0.0)) > 1.5:
            x = x / 255.0
    x = (x - float(mean)) / float(std)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def preprocess_image_batch(
    arr: np.ndarray,
    mean: float = IMAGE_MEAN,
    std: float = IMAGE_STD,
    out_h: int = MODEL_H,
    out_w: int = MODEL_W,
) -> torch.Tensor:
    """Normalize a batch [B, H, W, 1] and resize to [B, 1, out_h, out_w]."""
    x = np.asarray(arr)
    if x.ndim == 4:
        x = x[..., 0]
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
        if float(x.max(initial=0.0)) > 1.5:
            x = x / 255.0
    x = (x - float(mean)) / float(std)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1)
    return F.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False)


def preprocess_audio_array(arr: np.ndarray) -> torch.Tensor:
    """Return a [bins, frames] float tensor from the stored [frames, bins] array."""
    x = np.asarray(arr).astype(np.float32)
    if x.shape == (SPEC_FRAMES, SPEC_BINS):
        x = x.T
    elif x.shape == (SPEC_BINS, SPEC_FRAMES):
        pass
    else:
        raise ValueError(f"Unexpected audio window shape: {x.shape}")
    return torch.from_numpy(np.ascontiguousarray(x))


def augment_image_batch(
    x: torch.Tensor,
    scale_range: Sequence[float] = (0.95, 1.05),
    translate: float = 5.0,
) -> torch.Tensor:
    """Paper-style sheet augmentation: scaling + x/y translation.

    ``x`` is a normalized batch [B, 1, H, W]. Translation is specified in
    pixels of the original crop (H=200, W=160), matching the paper.
    """
    batch, _, height, width = x.shape
    scales = torch.empty(batch, device=x.device).uniform_(float(scale_range[0]), float(scale_range[1]))
    tx = torch.empty(batch, device=x.device).uniform_(-float(translate), float(translate))
    ty = torch.empty(batch, device=x.device).uniform_(-float(translate), float(translate))

    theta = torch.zeros(batch, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = scales
    theta[:, 1, 1] = scales
    theta[:, 0, 2] = 2.0 * tx / max(width - 1, 1)
    theta[:, 1, 2] = 2.0 * ty / max(height - 1, 1)

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def augment_audio(
    spec: torch.Tensor,
    time_shift: int = 1,
    tempo_jitter: float = 0.05,
) -> torch.Tensor:
    """Audio augmentation: small onset shift and optional tempo jitter.

    ``spec`` has shape [bins, frames]. Tempo jitter is applied by resampling the
    time axis and center-cropping/padding back to the original length. It is a
    spectrogram-domain approximation of the paper's tempo variation.
    """
    frames = spec.shape[1]
    if tempo_jitter and tempo_jitter > 0 and random.random() < 0.5:
        factor = random.uniform(1.0 - float(tempo_jitter), 1.0 + float(tempo_jitter))
        new_len = max(1, int(round(frames * factor)))
        resized = F.interpolate(
            spec.unsqueeze(0).unsqueeze(0),
            size=(spec.shape[0], new_len),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        if new_len >= frames:
            start = (new_len - frames) // 2
            resized = resized[:, start:start + frames]
        else:
            pad = frames - new_len
            left = pad // 2
            resized = F.pad(resized, (left, pad - left))
        spec = resized

    if time_shift and int(time_shift) > 0:
        shift = random.randint(-int(time_shift), int(time_shift))
        if shift:
            spec = torch.roll(spec, shifts=shift, dims=1)
            # Jangan biarkan frame yang wrap-around; isi dengan 0 (mean setelah
            # normalisasi per-bin) agar tidak menambah artefak.
            if shift > 0:
                spec[:, :shift] = 0.0
            else:
                spec[:, shift:] = 0.0
    return spec


# ---------------------------------------------------------------------------
# Torch datasets
# ---------------------------------------------------------------------------


class PositivePairDataset(Dataset):
    """Positive (sheet crop, audio window) pairs for retrieval training."""

    def __init__(
        self,
        crops_path: str | Path,
        windows_path: str | Path,
        crop_rows: Sequence[int],
        window_rows: Sequence[int],
        augment: bool = False,
        image_scale: Sequence[float] = (0.95, 1.05),
        image_translate: float = 5.0,
        audio_time_shift: int = 1,
        audio_tempo_jitter: float = 0.05,
    ) -> None:
        self.crops = np.load(crops_path, mmap_mode="r")
        self.windows = np.load(windows_path, mmap_mode="r")
        self.crop_rows = np.asarray(crop_rows, dtype=np.int64)
        self.window_rows = np.asarray(window_rows, dtype=np.int64)
        if len(self.crop_rows) != len(self.window_rows):
            raise ValueError("crop_rows and window_rows must have the same length.")
        self.augment = bool(augment)
        self.image_scale = tuple(image_scale)
        self.image_translate = float(image_translate)
        self.audio_time_shift = int(audio_time_shift)
        self.audio_tempo_jitter = float(audio_tempo_jitter)

    def __len__(self) -> int:
        return len(self.crop_rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop = self.crops[int(self.crop_rows[index])]
        spec = self.windows[int(self.window_rows[index])]
        # Augment at the original 200x160 resolution so the paper's +/-5 px
        # translation is in the same units as the paper, then resize to 100x80.
        image = preprocess_image_array(crop, out_h=IMAGE_H, out_w=IMAGE_W)
        if self.augment:
            image = augment_image_batch(
                image.unsqueeze(0),
                scale_range=self.image_scale,
                translate=self.image_translate,
            ).squeeze(0)
        image = F.interpolate(
            image.unsqueeze(0), size=(MODEL_H, MODEL_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        audio = preprocess_audio_array(spec)
        if self.augment:
            audio = augment_audio(
                audio,
                time_shift=self.audio_time_shift,
                tempo_jitter=self.audio_tempo_jitter,
            )
        # Tambahkan dimensi channel agar cocok dengan Conv2d: [1, 92, 42].
        return image, audio.unsqueeze(0)


class Task2Dataset(Dataset):
    """Score reference + performed audio + error class.

    If ``score_bank_indices`` is provided, items return
    ``(score_bank_index, audio, label)`` and the score image is not loaded.
    This is used when the sheet encoder is frozen and score embeddings were
    precomputed once per split.

    If ``one_variant_per_source`` is True, the dataset length is the number of
    unique source windows and each access samples one random error variant of
    that source. This keeps Task 2 epochs cheap while still covering all four
    classes over several epochs.
    """

    def __init__(
        self,
        score_crops_path: str | Path,
        error_windows_path: str | Path,
        meta: pd.DataFrame,
        noteheads: pd.DataFrame,
        score_bank_indices: np.ndarray | Sequence[int] | None = None,
        augment_image: bool = False,
        image_scale: Sequence[float] = (0.95, 1.05),
        image_translate: float = 5.0,
        one_variant_per_source: bool = False,
    ) -> None:
        self.meta = meta.reset_index(drop=True)
        self.crops = np.load(score_crops_path, mmap_mode="r")
        self.errors = np.load(error_windows_path, mmap_mode="r")

        notehead_map = noteheads.set_index("notehead_id")["crop_row"].to_dict()
        crop_rows = self.meta["notehead_id"].map(notehead_map)
        if crop_rows.isna().any():
            raise ValueError("Task 2 metadata contains noteheads missing from the split.")
        self.crop_rows = crop_rows.to_numpy(dtype=np.int64)
        self.error_rows = self.meta["error_row"].to_numpy(dtype=np.int64)
        self.labels = self.meta["class_id"].to_numpy(dtype=np.int64)

        self.score_bank_indices = None
        if score_bank_indices is not None:
            self.score_bank_indices = np.asarray(score_bank_indices, dtype=np.int64)
            if len(self.score_bank_indices) != len(self.labels):
                raise ValueError("score_bank_indices must align with metadata rows.")

        self.augment_image = bool(augment_image)
        self.image_scale = tuple(image_scale)
        self.image_translate = float(image_translate)

        self.one_variant_per_source = bool(one_variant_per_source)
        self._source_rows: list[np.ndarray] | None = None
        if self.one_variant_per_source:
            grouped = self.meta.groupby("source_window_id").indices
            self._source_rows = [
                np.asarray(rows, dtype=np.int64) for rows in grouped.values()
            ]
            self._length = len(self._source_rows)
        else:
            self._length = len(self.labels)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if self._source_rows is not None:
            row = int(random.choice(self._source_rows[index]))
        else:
            row = int(index)
        audio = preprocess_audio_array(
            self.errors[int(self.error_rows[row])]
        ).unsqueeze(0)
        label = int(self.labels[row])
        if self.score_bank_indices is not None:
            return int(self.score_bank_indices[row]), audio, label
        image = preprocess_image_array(
            self.crops[int(self.crop_rows[row])],
            out_h=IMAGE_H,
            out_w=IMAGE_W,
        )
        if self.augment_image:
            image = augment_image_batch(
                image.unsqueeze(0),
                scale_range=self.image_scale,
                translate=self.image_translate,
            ).squeeze(0)
        image = F.interpolate(
            image.unsqueeze(0), size=(MODEL_H, MODEL_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return image, audio, label


# ---------------------------------------------------------------------------
# Model: Dorfer et al. dual encoder
# ---------------------------------------------------------------------------


class ConvStack(nn.Module):
    """Two 3x3 convolutions + ELU + BN, followed by 2x2 max pooling.

    The official code uses Conv -> ELU -> BN (``conv_bn`` in
    ``models/mutopia_ccal_cont_rsz.py``). The paper table writes BN-ELU; we
    follow the reference implementation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.act(self.conv1(x)))
        x = self.bn2(self.act(self.conv2(x)))
        return self.pool(x)


class _EncoderBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        in_channels: int = 1,
        channels: Sequence[int] = (24, 48, 96, 96),
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        if not channels:
            raise ValueError("channels must not be empty")
        blocks = []
        previous = int(in_channels)
        for channel in channels:
            blocks.append(ConvStack(previous, channel))
            previous = channel
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(previous, embed_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.proj_bn(self.proj(x))
        return x.mean(dim=(2, 3))


class SheetEncoder(_EncoderBackbone):
    """Sheet image encoder. Input [B, 1, 100, 80], output [B, 32]."""

    def __init__(
        self,
        embed_dim: int = 32,
        channels: Sequence[int] = (24, 48, 96, 96),
    ) -> None:
        super().__init__(embed_dim=embed_dim, channels=channels)


class AudioEncoder(_EncoderBackbone):
    """Audio spectrogram encoder. Input [B, 1, 92, 42], output [B, 32]."""

    def __init__(
        self,
        embed_dim: int = 32,
        channels: Sequence[int] = (24, 48, 96, 96),
    ) -> None:
        super().__init__(embed_dim=embed_dim, channels=channels)


class CCALayer(nn.Module):
    """Running canonical-correlation embedding layer.

    Mirrors ``CCALayer`` from the official Lasagne implementation:
    projections U and V are recomputed from the current batch statistics and
    kept as non-trainable buffers. Gradients flow through the centred features
    but not through the projections.
    """

    def __init__(
        self,
        dim: int = 32,
        r1: float = 1e-3,
        r2: float = 1e-3,
        rT: float = 1e-3,
        alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.rT = float(rT)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.register_buffer("mean1", torch.zeros(self.dim))
        self.register_buffer("mean2", torch.zeros(self.dim))
        self.register_buffer("U", torch.eye(self.dim))
        self.register_buffer("V", torch.eye(self.dim))

    def _eye(self, device: torch.device) -> torch.Tensor:
        return torch.eye(self.dim, device=device, dtype=self.mean1.dtype)

    @torch.no_grad()
    def update(self, h1: torch.Tensor, h2: torch.Tensor) -> None:
        batch = h1.shape[0]
        if batch < 2:
            return

        m1 = h1.mean(dim=0)
        m2 = h2.mean(dim=0)
        new_mean1 = (1.0 - self.alpha) * self.mean1 + self.alpha * m1
        new_mean2 = (1.0 - self.alpha) * self.mean2 + self.alpha * m2

        c1 = h1 - new_mean1
        c2 = h2 - new_mean2
        denom = float(batch - 1)
        s12 = (c1.T @ c2) / denom
        s11 = (c1.T @ c1) / denom
        s22 = (c2.T @ c2) / denom
        s11 = 0.5 * (s11 + s11.T) + self.r1 * self._eye(h1.device)
        s22 = 0.5 * (s22 + s22.T) + self.r2 * self._eye(h1.device)

        if not (torch.isfinite(s11).all() and torch.isfinite(s22).all() and torch.isfinite(s12).all()):
            return

        try:
            d1, a1 = torch.linalg.eigh(s11)
            d2, a2 = torch.linalg.eigh(s22)
        except Exception:
            return

        d1 = d1.clamp_min(self.eps)
        d2 = d2.clamp_min(self.eps)
        s11_inv_sqrt = (a1 * d1.rsqrt().unsqueeze(0)) @ a1.T
        s22_inv_sqrt = (a2 * d2.rsqrt().unsqueeze(0)) @ a2.T

        t_mat = s11_inv_sqrt @ s12 @ s22_inv_sqrt
        m1_mat = t_mat @ t_mat.T + self.rT * self._eye(h1.device)
        m2_mat = t_mat.T @ t_mat + self.rT * self._eye(h1.device)

        try:
            _, e_vec = torch.linalg.eigh(0.5 * (m1_mat + m1_mat.T))
            _, f_vec = torch.linalg.eigh(0.5 * (m2_mat + m2_mat.T))
        except Exception:
            return

        u_new = s11_inv_sqrt @ e_vec
        v_new = s22_inv_sqrt @ f_vec
        signs = torch.sign(torch.diagonal(u_new.T @ s12 @ v_new))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        u_new = u_new * signs.unsqueeze(0)

        if not (torch.isfinite(u_new).all() and torch.isfinite(v_new).all()):
            return

        self.mean1.copy_(new_mean1)
        self.mean2.copy_(new_mean2)
        self.U.copy_(u_new)
        self.V.copy_(v_new)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        update: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if update and self.training:
            self.update(h1.detach(), h2.detach())
        z1 = (h1 - self.mean1) @ self.U
        z2 = (h2 - self.mean2) @ self.V
        return F.normalize(z1, dim=1), F.normalize(z2, dim=1)


class DualEncoder(nn.Module):
    """Dorfer dual encoder: two CNN branches + CCA embedding + L2."""

    def __init__(
        self,
        embed_dim: int = 32,
        use_cca: bool = True,
        cca_kwargs: dict | None = None,
        channels: Sequence[int] = (24, 48, 96, 96),
    ) -> None:
        super().__init__()
        self.sheet = SheetEncoder(embed_dim=embed_dim, channels=channels)
        self.audio = AudioEncoder(embed_dim=embed_dim, channels=channels)
        self.cca = CCALayer(embed_dim, **(cca_kwargs or {})) if use_cca else None

    def forward(
        self,
        sheet_image: torch.Tensor,
        audio_spec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_sheet = self.sheet(sheet_image)
        h_audio = self.audio(audio_spec)
        if self.cca is not None:
            return self.cca(h_sheet, h_audio, update=self.training)
        return F.normalize(h_sheet, dim=1), F.normalize(h_audio, dim=1)


class RankingLoss(nn.Module):
    """Pairwise ranking hinge loss used in the paper (Kiros et al., 2014)."""

    def __init__(self, margin: float = 0.7) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, z_sheet: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        if z_sheet.shape[0] < 2:
            return z_sheet.sum() * 0.0
        scores = z_sheet @ z_audio.T
        diagonal = scores.diagonal().unsqueeze(1)
        mask = ~torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        losses = F.relu(self.margin - diagonal + scores)
        return losses[mask].mean()


class FusionClassifier(nn.Module):
    """Multimodal classifier for Task 2.

    Features: [x, y, x*y, |x-y|], where x and y are L2-normalized 32-d
    embeddings of the score reference and the performed audio.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sheet_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        x = F.normalize(sheet_emb, dim=1)
        y = F.normalize(audio_emb, dim=1)
        features = torch.cat([x, y, x * y, (x - y).abs()], dim=1)
        return self.net(features)


class Task2Model(nn.Module):
    """Dorfer encoders + fusion classifier for performance error detection."""

    def __init__(
        self,
        sheet_encoder: nn.Module,
        audio_encoder: nn.Module,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sheet = sheet_encoder
        self.audio = audio_encoder
        self.classifier = FusionClassifier(embed_dim, hidden_dim, num_classes, dropout)

    def forward(self, score_image: torch.Tensor, audio_spec: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.sheet(score_image), self.audio(audio_spec))

    def forward_from_embeddings(
        self,
        score_emb: torch.Tensor,
        audio_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(score_emb, audio_emb)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def embed_image_rows(
    encoder: nn.Module,
    crops_path: str | Path,
    crop_rows: Sequence[int],
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    normalize: bool = False,
) -> torch.Tensor:
    crops = np.load(crops_path, mmap_mode="r")
    crop_rows = np.asarray(crop_rows, dtype=np.int64)
    encoder.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(crop_rows), batch_size):
        rows = crop_rows[start:start + batch_size]
        batch = np.stack([crops[int(row)] for row in rows])
        tensor = preprocess_image_batch(batch).to(device)
        z = encoder(tensor).float().cpu()
        if normalize:
            z = F.normalize(z, dim=1)
        outputs.append(z)
    if not outputs:
        return torch.empty(0, 32)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def embed_audio_rows(
    encoder: nn.Module,
    windows_path: str | Path,
    window_rows: Sequence[int],
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    normalize: bool = False,
) -> torch.Tensor:
    windows = np.load(windows_path, mmap_mode="r")
    window_rows = np.asarray(window_rows, dtype=np.int64)
    encoder.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(window_rows), batch_size):
        rows = window_rows[start:start + batch_size]
        batch = np.stack([windows[int(row)] for row in rows]).astype(np.float32)
        tensor = torch.from_numpy(batch).permute(0, 2, 1).unsqueeze(1).to(device)
        z = encoder(tensor).float().cpu()
        if normalize:
            z = F.normalize(z, dim=1)
        outputs.append(z)
    if not outputs:
        return torch.empty(0, 32)
    return torch.cat(outputs, dim=0)


def project_cca(
    hidden: torch.Tensor,
    mean: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed CCA projection and L2-normalize."""
    mean = mean.to(hidden.device)
    projection = projection.to(hidden.device)
    return F.normalize((hidden - mean) @ projection, dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ranks_from_scores(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rank of one correct target per query (1 = best)."""
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=np.int64)
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("scores and targets must have the same number of rows.")
    correct = scores[np.arange(scores.shape[0]), targets]
    return (scores > correct[:, None]).sum(axis=1) + 1


def ranks_from_multi_scores(
    scores: np.ndarray,
    target_lists: Sequence[Sequence[int]],
) -> np.ndarray:
    """Rank of the best correct target when a query has several valid targets."""
    scores = np.asarray(scores)
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    for index, targets in enumerate(target_lists):
        targets = np.asarray(targets, dtype=np.int64)
        if targets.size == 0:
            raise ValueError(f"Query {index} has no valid target.")
        correct_score = scores[index, targets].max()
        ranks[index] = int((scores[index] > correct_score).sum()) + 1
    return ranks


def retrieval_metrics(ranks: Sequence[int], ks: Sequence[int] = (1, 5, 10, 25)) -> dict:
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        return {"n_queries": 0}
    result = {f"R@{int(k)}": float((ranks <= int(k)).mean()) for k in ks}
    result["MRR"] = float((1.0 / ranks).mean())
    result["median_rank"] = float(np.median(ranks))
    result["mean_rank"] = float(ranks.mean())
    result["n_queries"] = int(ranks.size)
    return result


def hit_rate_at_k(ranks: Sequence[int], ks: Sequence[int] = (1, 5, 10)) -> dict:
    ranks = np.asarray(ranks, dtype=np.float64)
    result = {}
    for k in ks:
        hits = int((ranks <= int(k)).sum())
        result[f"Rk@{int(k)}"] = hits
        result[f"Rk@{int(k)}_pct"] = float(hits / max(len(ranks), 1))
    result["n_queries"] = int(len(ranks))
    return result


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
    num_classes: int = 4,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)

    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(confusion[index].sum()),
        }

    return {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "n": int(confusion.sum()),
    }


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Positive class = error (label != 0)."""
    y_true = (np.asarray(y_true) != 0).astype(np.int64)
    y_pred = (np.asarray(y_pred) != 0).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def topk_accuracy(probabilities: np.ndarray, y_true: Sequence[int], k: int = 2) -> float:
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true, dtype=np.int64)
    top = np.argsort(-probabilities, axis=1)[:, :int(k)]
    return float((top == y_true[:, None]).any(axis=1).mean())
