"""Corruption wrappers for Task 2 audio and score inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def add_gaussian_noise(
    spec: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = spec.astype(np.float64)
    power = float(np.mean(values * values))
    if power <= 0.0:
        return spec.copy()
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=values.shape)
    return (values + noise).astype(np.float32)


def time_mask(
    spec: np.ndarray,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = spec.copy()
    frames = result.shape[0]
    width = max(1, min(int(width), frames))
    start = int(rng.integers(0, max(1, frames - width + 1)))
    result[start : start + width, :] = 0.0
    return result


def frequency_mask(
    spec: np.ndarray,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = spec.copy()
    bins = result.shape[1]
    width = max(1, min(int(width), bins))
    start = int(rng.integers(0, max(1, bins - width + 1)))
    result[:, start : start + width] = 0.0
    return result


def score_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    return (image.astype(np.float64) + noise).astype(np.float32)


def score_cutout(
    image: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[-2], result.shape[-1]
    size = max(1, min(int(size), height, width))
    top = int(rng.integers(0, max(1, height - size + 1)))
    left = int(rng.integers(0, max(1, width - size + 1)))
    result[..., top : top + size, left : left + size] = 0.0
    return result


class Task2CorruptionDataset(Dataset):
    """Wrap a Task 2 dataset and corrupt its audio or score inputs.

    Eval mode applies a fixed corruption spec deterministically per sample.
    Train mode applies random mild augmentations.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        mode: str = "eval",
        audio_noise_snr: float | None = None,
        time_mask_width: int | None = None,
        frequency_mask_width: int | None = None,
        score_noise_sigma: float | None = None,
        score_cutout_size: int | None = None,
        train_config: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        if mode not in {"eval", "train"}:
            raise ValueError("mode must be 'eval' or 'train'")
        self.base = base
        self.mode = mode
        self.audio_noise_snr = audio_noise_snr
        self.time_mask_width = time_mask_width
        self.frequency_mask_width = frequency_mask_width
        self.score_noise_sigma = score_noise_sigma
        self.score_cutout_size = score_cutout_size
        self.train_config = dict(train_config or {})
        self.seed = int(seed)
        self._train_rng = np.random.default_rng(self.seed + 7919)

    def __len__(self) -> int:
        return len(self.base)

    def _corrupt_eval(self, image: torch.Tensor, audio: torch.Tensor, index: int):
        rng = np.random.default_rng(self.seed + int(index))
        spec = audio.squeeze(0).numpy().astype(np.float32)
        if self.audio_noise_snr is not None:
            spec = add_gaussian_noise(spec, self.audio_noise_snr, rng)
        if self.time_mask_width is not None:
            spec = time_mask(spec, self.time_mask_width, rng)
        if self.frequency_mask_width is not None:
            spec = frequency_mask(spec, self.frequency_mask_width, rng)
        image_np = image.numpy().astype(np.float32)
        if self.score_noise_sigma is not None:
            image_np = score_noise(image_np, self.score_noise_sigma, rng)
        if self.score_cutout_size is not None:
            image_np = score_cutout(image_np, self.score_cutout_size, rng)
        return torch.from_numpy(image_np), torch.from_numpy(spec).unsqueeze(0)

    def _corrupt_train(self, image: torch.Tensor, audio: torch.Tensor):
        config = self.train_config
        rng = self._train_rng
        spec = audio.squeeze(0).numpy().astype(np.float32)
        if rng.random() < float(config.get("audio_noise_prob", 0.0)):
            low, high = config.get("audio_noise_snr", (10.0, 20.0))
            spec = add_gaussian_noise(spec, float(rng.uniform(low, high)), rng)
        if rng.random() < float(config.get("time_mask_prob", 0.0)):
            low, high = config.get("time_mask_width", (1, 5))
            spec = time_mask(spec, int(rng.integers(low, high + 1)), rng)
        if rng.random() < float(config.get("frequency_mask_prob", 0.0)):
            low, high = config.get("frequency_mask_width", (1, 5))
            spec = frequency_mask(spec, int(rng.integers(low, high + 1)), rng)
        image_np = image.numpy().astype(np.float32)
        if rng.random() < float(config.get("score_noise_prob", 0.0)):
            low, high = config.get("score_noise_sigma", (0.02, 0.05))
            image_np = score_noise(image_np, float(rng.uniform(low, high)), rng)
        if rng.random() < float(config.get("score_cutout_prob", 0.0)):
            low, high = config.get("score_cutout_size", (8, 16))
            image_np = score_cutout(image_np, int(rng.integers(low, high + 1)), rng)
        return torch.from_numpy(image_np), torch.from_numpy(spec).unsqueeze(0)

    def __getitem__(self, index: int):
        image, audio, label = self.base[index]
        if self.mode == "train":
            image, audio = self._corrupt_train(image, audio)
        else:
            image, audio = self._corrupt_eval(image, audio, index)
        return image, audio, label


class ZeroScoreDataset(Dataset):
    """Wrap a Task 2 dataset and replace score crops with zeros."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, spec, label = self.base[index]
        return torch.zeros_like(image), spec, label

