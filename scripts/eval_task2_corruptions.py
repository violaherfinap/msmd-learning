"""Evaluate a Task 2 checkpoint on clean and corrupted test inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baseline_dorfer import (
    CLASS_NAMES,
    AudioEncoder,
    SheetEncoder,
    Task2Dataset,
    Task2Model,
    binary_metrics,
    classification_metrics,
    configure_torch,
    load_checkpoint,
    resolve_device,
    save_json,
    set_seed,
    topk_accuracy,
)
from scripts.task2_corruption import Task2CorruptionDataset, ZeroScoreDataset

CONDITIONS = {
    "clean": {},
    "audio_noise_snr20": {"audio_noise_snr": 20.0},
    "audio_noise_snr10": {"audio_noise_snr": 10.0},
    "audio_noise_snr5": {"audio_noise_snr": 5.0},
    "time_mask10": {"time_mask_width": 10},
    "frequency_mask10": {"frequency_mask_width": 10},
    "combined_mild": {
        "audio_noise_snr": 10.0,
        "time_mask_width": 5,
        "frequency_mask_width": 5,
    },
    "score_noise_0.1": {"score_noise_sigma": 0.1},
    "score_cutout_16": {"score_cutout_size": 16},
}


def predict(model, loader, device):
    probabilities, labels = [], []
    with torch.no_grad():
        for images, specs, batch_labels in loader:
            logits = model(images.to(device), specs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 corruption evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--task1-checkpoint",
        default=str(ROOT / "checkpoints" / "T1-C1_seed42" / "best.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    configure_torch(None)
    device = resolve_device("auto")

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    variant = checkpoint.get("variant", config.get("variant", "multimodal"))

    task1 = load_checkpoint(args.task1_checkpoint, map_location="cpu")
    channels = tuple(config["channels"])
    sheet_encoder = SheetEncoder(embed_dim=config["embed_dim"], channels=channels)
    audio_encoder = AudioEncoder(embed_dim=config["embed_dim"], channels=channels)
    sheet_encoder.load_state_dict(task1["sheet_encoder"])
    audio_encoder.load_state_dict(task1["audio_encoder"])

    model = Task2Model(
        sheet_encoder=sheet_encoder,
        audio_encoder=audio_encoder,
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_dir = ROOT / "data" / "processed"
    meta = pd.read_parquet(data_dir / "task2_windows_test.parquet").reset_index(
        drop=True
    )
    holdout_soundfont = config.get("holdout_soundfont")
    if holdout_soundfont:
        meta = meta[meta["soundfont"] == holdout_soundfont].reset_index(drop=True)
    noteheads = pd.read_parquet(data_dir / "noteheads_test.parquet")
    base = Task2Dataset(
        data_dir / "score_crops_test.npy",
        data_dir / "error_windows_test.npy",
        meta,
        noteheads,
        score_bank_indices=None,
        one_variant_per_source=False,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "task1_checkpoint": str(Path(args.task1_checkpoint)),
        "variant": variant,
        "holdout_soundfont": holdout_soundfont,
        "seed": int(args.seed),
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "conditions": {},
    }

    for name, corruption in CONDITIONS.items():
        dataset = base
        if corruption:
            dataset = Task2CorruptionDataset(
                base,
                mode="eval",
                seed=args.seed,
                **corruption,
            )
        if variant == "audio-only":
            dataset = ZeroScoreDataset(dataset)
        loader = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        probabilities, labels = predict(model, loader, device)
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(
            labels,
            predictions,
            class_names=CLASS_NAMES,
            num_classes=config["num_classes"],
        )
        metrics["top2_accuracy"] = topk_accuracy(probabilities, labels, k=2)
        binary = binary_metrics(labels, predictions)
        per_class = metrics["per_class"]
        row = {
            "condition": name,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "top2_accuracy": float(metrics["top2_accuracy"]),
            "binary_accuracy": float(binary["accuracy"]),
            "binary_f1": float(binary["f1"]),
            "f1_correct": float(per_class["correct"]["f1"]),
            "f1_wrong_pitch": float(per_class["wrong_pitch"]["f1"]),
            "f1_missing_note": float(per_class["missing_note"]["f1"]),
            "f1_extra_note": float(per_class["extra_note"]["f1"]),
            "n": int(len(labels)),
        }
        rows.append(row)
        payload["conditions"][name] = {
            "corruption": corruption,
            "metrics": metrics,
            "binary": binary,
        }
        print(
            f"{name:20s} | macro-F1 {row['macro_f1']:.4f} | "
            f"acc {row['accuracy']:.4f} | top2 {row['top2_accuracy']:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "corruption_metrics.csv", index=False)
    save_json(out_dir / "corruption_metrics.json", payload)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(table["condition"], table["macro_f1"])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("macro-F1")
    axis.set_title("Task 2 corruption robustness")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "corruption_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_dir / "corruption_metrics.csv")


if __name__ == "__main__":
    main()
