"""Train a Task 2 model with mild audio/score corruption augmentation."""

from __future__ import annotations

import argparse
import json
import sys
import time
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

AUGMENTATION_PRESETS = {
    "none": {},
    "mild": {
        "audio_noise_prob": 0.5,
        "audio_noise_snr": (10.0, 20.0),
        "time_mask_prob": 0.5,
        "time_mask_width": (1, 5),
        "frequency_mask_prob": 0.5,
        "frequency_mask_width": (1, 8),
        "score_noise_prob": 0.3,
        "score_noise_sigma": (0.02, 0.05),
        "score_cutout_prob": 0.3,
        "score_cutout_size": (8, 16),
    },
    "strong": {
        "audio_noise_prob": 0.7,
        "audio_noise_snr": (5.0, 15.0),
        "time_mask_prob": 0.7,
        "time_mask_width": (1, 8),
        "frequency_mask_prob": 0.7,
        "frequency_mask_width": (1, 12),
        "score_noise_prob": 0.3,
        "score_noise_sigma": (0.02, 0.08),
        "score_cutout_prob": 0.3,
        "score_cutout_size": (8, 20),
    },
}

AUGMENTATION = AUGMENTATION_PRESETS["mild"]


def build_split(data_dir: Path, split: str):
    meta = pd.read_parquet(data_dir / f"task2_windows_{split}.parquet").reset_index(
        drop=True
    )
    noteheads = pd.read_parquet(data_dir / f"noteheads_{split}.parquet")
    return meta, noteheads


def subsample_by_source(meta: pd.DataFrame, n_sources: int) -> pd.DataFrame:
    source_ids = meta["source_window_id"].drop_duplicates().iloc[: int(n_sources)]
    return meta[meta["source_window_id"].isin(source_ids)].reset_index(drop=True)


def predict_task2(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, specs, batch_labels in loader:
            logits = model(images.to(device), specs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, specs, labels in loader:
        logits = model(images.to(device), specs.to(device))
        labels = labels.to(device)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Task 2 loss is not finite.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            grad_clip,
        )
        optimizer.step()
        total_loss += float(loss.detach()) * labels.size(0)
        total_items += labels.size(0)
    return total_loss / max(total_items, 1)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    config,
    device,
    checkpoint_path,
    history_csv_path,
    augmentation,
):
    history = []
    best_f1 = -1.0
    bad_epochs = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config["grad_clip"],
        )
        probabilities, labels = predict_task2(model, val_loader, device)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Validation probabilities are not finite.")
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(
            labels,
            predictions,
            class_names=CLASS_NAMES,
            num_classes=config["num_classes"],
        )
        metrics["top2_accuracy"] = topk_accuracy(probabilities, labels, k=2)
        scheduler.step(metrics["macro_f1"])
        elapsed = time.time() - started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "seconds": elapsed,
        }
        history.append(record)
        flat = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"confusion_matrix", "per_class"}
                }
                for item in history
            ]
        )
        flat.to_csv(history_csv_path, index=False)
        print(
            f"epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val acc {metrics['accuracy']:.4f} | "
            f"val macro-F1 {metrics['macro_f1']:.4f} | "
            f"top2 {metrics['top2_accuracy']:.4f} | {elapsed:.1f}s",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "task1_checkpoint": str(
                        ROOT / "checkpoints" / "T1-C1_seed42" / "best.pt"
                    ),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "augmentation": augmentation,
                    "variant": config.get("variant", "multimodal"),
                },
                checkpoint_path,
            )
            print(f"  -> checkpoint saved (best val macro-F1 {best_f1:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["early_stop_patience"]):
                print(f"Early stop after {bad_epochs} epochs without improvement.")
                break
    return history, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 robust training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-name", default="T2-robust_seed42")
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument(
        "--augmentation",
        choices=sorted(AUGMENTATION_PRESETS),
        default="mild",
    )
    parser.add_argument(
        "--variant",
        choices=["multimodal", "audio-only"],
        default="multimodal",
    )
    parser.add_argument("--holdout-soundfont", default=None)
    args = parser.parse_args()

    augmentation = AUGMENTATION_PRESETS[args.augmentation]
    if args.variant == "audio-only":
        augmentation = dict(augmentation)
        augmentation["score_noise_prob"] = 0.0
        augmentation["score_cutout_prob"] = 0.0
    set_seed(args.seed)
    configure_torch(None)
    device = resolve_device("auto")

    data_dir = ROOT / "data" / "processed"
    out_dir = ROOT / "reports" / "experiments" / args.out_name
    figure_dir = out_dir / "figures"
    ckpt_dir = ROOT / "checkpoints" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_checkpoint(
        ROOT / "checkpoints" / "T2-baseline_seed42" / "best.pt",
        map_location="cpu",
    )
    config = dict(baseline["config"])
    config["seed"] = int(args.seed)
    config["epochs"] = int(args.epochs)
    config["fast_dev"] = bool(args.fast_dev)
    config["augmentation"] = augmentation
    config["variant"] = args.variant
    config["freeze_sheet"] = args.variant == "audio-only"
    config["holdout_soundfont"] = args.holdout_soundfont
    if args.fast_dev:
        config["epochs"] = 1
        config["early_stop_patience"] = 1

    task1 = load_checkpoint(
        ROOT / "checkpoints" / "T1-C1_seed42" / "best.pt",
        map_location="cpu",
    )
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

    train_meta, train_noteheads = build_split(data_dir, "train")
    val_meta, val_noteheads = build_split(data_dir, "val")
    test_meta, test_noteheads = build_split(data_dir, "test")
    if args.holdout_soundfont:
        train_meta = train_meta[
            train_meta["soundfont"] != args.holdout_soundfont
        ].reset_index(drop=True)
        val_meta = val_meta[
            val_meta["soundfont"] != args.holdout_soundfont
        ].reset_index(drop=True)
        test_meta = test_meta[
            test_meta["soundfont"] == args.holdout_soundfont
        ].reset_index(drop=True)
        if train_meta.empty or val_meta.empty or test_meta.empty:
            raise ValueError("Holdout soundfont filtering produced an empty split.")
    if args.fast_dev:
        train_meta = subsample_by_source(train_meta, 2000)
        val_meta = subsample_by_source(val_meta, 500)

    train_base = Task2Dataset(
        data_dir / "score_crops_train.npy",
        data_dir / "error_windows_train.npy",
        train_meta,
        train_noteheads,
        score_bank_indices=None,
        one_variant_per_source=True,
    )
    if augmentation:
        train_dataset = Task2CorruptionDataset(
            train_base,
            mode="train",
            train_config=augmentation,
            seed=args.seed,
        )
    else:
        train_dataset = train_base
    val_dataset = Task2Dataset(
        data_dir / "score_crops_val.npy",
        data_dir / "error_windows_val.npy",
        val_meta,
        val_noteheads,
        score_bank_indices=None,
        one_variant_per_source=False,
    )
    test_dataset = Task2Dataset(
        data_dir / "score_crops_test.npy",
        data_dir / "error_windows_test.npy",
        test_meta,
        test_noteheads,
        score_bank_indices=None,
        one_variant_per_source=False,
    )
    if args.variant == "audio-only":
        train_dataset = ZeroScoreDataset(train_dataset)
        val_dataset = ZeroScoreDataset(val_dataset)
        test_dataset = ZeroScoreDataset(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    if args.variant == "audio-only":
        for parameter in model.sheet.parameters():
            parameter.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_params = [
        {"params": model.classifier.parameters(), "lr": config["lr_head"]},
        {"params": model.audio.parameters(), "lr": config["lr_encoder"]},
    ]
    if args.variant != "audio-only":
        optimizer_params.append(
            {"params": model.sheet.parameters(), "lr": config["lr_encoder"]}
        )
    optimizer = torch.optim.Adam(
        optimizer_params,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    checkpoint_path = ckpt_dir / "best.pt"
    history_csv_path = out_dir / "history.csv"
    print(f"Training {args.out_name} -> {out_dir}", flush=True)
    history, best_f1 = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config,
        device,
        checkpoint_path,
        history_csv_path,
        augmentation,
    )
    print(f"Best validation macro-F1: {best_f1:.4f}", flush=True)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities, labels = predict_task2(model, test_loader, device)
    predictions = probabilities.argmax(axis=1)
    test_metrics = classification_metrics(
        labels,
        predictions,
        class_names=CLASS_NAMES,
        num_classes=config["num_classes"],
    )
    test_metrics["top2_accuracy"] = topk_accuracy(probabilities, labels, k=2)
    binary = binary_metrics(labels, predictions)
    error_mask = labels != 0
    detected_mask = predictions != 0
    both_error = error_mask & detected_mask
    error_type_metrics = classification_metrics(
        labels[both_error] - 1,
        predictions[both_error] - 1,
        class_names=CLASS_NAMES[1:],
        num_classes=3,
    )
    error_type_metrics["coverage_of_true_errors"] = float(
        both_error.sum() / max(error_mask.sum(), 1)
    )

    payload = {
        "config": config,
        "variant": args.variant,
        "holdout_soundfont": args.holdout_soundfont,
        "augmentation": augmentation,
        "seed": int(args.seed),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "task1_checkpoint": str(
            ROOT / "checkpoints" / "T1-C1_seed42" / "best.pt"
        ),
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "test_metrics": test_metrics,
        "binary_correct_vs_error": binary,
        "error_type_3class": error_type_metrics,
        "history": history,
    }
    save_json(out_dir / "metrics.json", payload)
    save_json(out_dir / "config.json", config)
    print(
        f"clean test macro-F1: {test_metrics['macro_f1']:.6f} | "
        f"accuracy: {test_metrics['accuracy']:.6f}",
        flush=True,
    )
    print("saved:", out_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
