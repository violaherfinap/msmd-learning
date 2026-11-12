"""Run reproducible Task 1 experiments from YAML configs.

Usage:
    python scripts/run_experiment.py --config configs/experiments/t1_c1_baseline.yaml
    python scripts/run_experiment.py --config configs/experiments/t1_c1_baseline.yaml --fast-dev
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from scripts.baseline_dorfer import (
    DualEncoder,
    PositivePairDataset,
    RankingLoss,
    build_positive_pairs,
    configure_torch,
    embed_audio_rows,
    embed_image_rows,
    hit_rate_at_k,
    load_checkpoint,
    load_split_frames,
    one_window_per_notehead,
    project_cca,
    ranks_from_multi_scores,
    ranks_from_scores,
    resolve_device,
    retrieval_metrics,
    save_json,
    set_seed,
)

DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "experiments"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "experiments"

_LOG_HANDLE = None


def log(message: str) -> None:
    print(message, flush=True)
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.write(message + "\n")
        _LOG_HANDLE.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def normalize_config(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = dict(config)
    resolved.setdefault("task", "task1")
    resolved.setdefault("experiment", "dual_encoder")
    resolved.setdefault("seed", 42)
    resolved.setdefault("device", "auto")
    resolved.setdefault("threads", None)
    resolved.setdefault("batch_size", 128)
    resolved.setdefault("eval_batch_size", 256)
    resolved.setdefault("epochs", 6)
    resolved.setdefault("early_stop_patience", 3)
    resolved.setdefault("lr", 2e-3)
    resolved.setdefault("weight_decay", 1e-4)
    resolved.setdefault("grad_clip", 5.0)
    resolved.setdefault("margin", 0.7)
    resolved.setdefault("embed_dim", 32)
    resolved.setdefault("channels", [16, 32, 64])
    resolved.setdefault("use_cca", False)
    resolved.setdefault("cca", {"r1": 1e-3, "r2": 1e-3, "rT": 1e-3, "alpha": 1.0})
    resolved.setdefault(
        "augment",
        {
            "image_scale": [0.95, 1.05],
            "image_translate": 5.0,
            "audio_time_shift": 1,
            "audio_tempo_jitter": 0.1,
        },
    )
    resolved.setdefault("num_workers", 0)
    resolved.setdefault("fast_dev", False)
    resolved.setdefault("data_dir", str(DEFAULT_DATA_DIR))
    resolved["channels"] = [int(value) for value in resolved["channels"]]
    resolved["augment"] = dict(resolved["augment"])

    if args.fast_dev:
        resolved["fast_dev"] = True
    if args.seed is not None:
        resolved["seed"] = int(args.seed)
    if args.epochs is not None:
        resolved["epochs"] = int(args.epochs)
    if args.device is not None:
        resolved["device"] = str(args.device)
    if args.threads is not None:
        resolved["threads"] = int(args.threads)

    if resolved["fast_dev"]:
        resolved["epochs"] = 2
        resolved["early_stop_patience"] = 2
        resolved["train_limit"] = 2000
        resolved["val_limit"] = 500
    else:
        resolved.setdefault("train_limit", None)
        resolved.setdefault("val_limit", None)
    return resolved


def create_run_dir(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, Path]:
    requested = args.run_id or config.get("run_id") or f"task1_{datetime.now():%Y%m%d_%H%M%S}"
    output_root = args.output_dir or DEFAULT_OUTPUT_DIR
    output_root = Path(output_root)
    run_dir = output_root / requested
    if run_dir.exists() and not args.force:
        requested = f"{requested}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = output_root / requested
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return requested, run_dir


def build_task1_loaders(
    config: dict[str, Any],
    data_dir: Path,
) -> tuple[DataLoader, DataLoader, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    frames = {
        split: load_split_frames(data_dir, split)
        for split in ("train", "val", "test")
    }
    positives = {
        split: build_positive_pairs(frames[split][2], frames[split][0], frames[split][1])
        for split in ("train", "val")
    }
    train_pairs = positives["train"]
    val_pairs = one_window_per_notehead(positives["val"])

    train_dataset = PositivePairDataset(
        data_dir / "score_crops_train.npy",
        data_dir / "audio_windows_train.npy",
        train_pairs["crop_row"].to_numpy(),
        train_pairs["window_row_eval"].to_numpy(),
        augment=True,
        **config["augment"],
    )
    val_dataset = PositivePairDataset(
        data_dir / "score_crops_val.npy",
        data_dir / "audio_windows_val.npy",
        val_pairs["crop_row"].to_numpy(),
        val_pairs["window_row_eval"].to_numpy(),
        augment=False,
        **config["augment"],
    )

    train_limit = config.get("train_limit")
    val_limit = config.get("val_limit")
    if train_limit:
        train_dataset = Subset(train_dataset, range(min(int(train_limit), len(train_dataset))))
    if val_limit:
        val_dataset = Subset(val_dataset, range(min(int(val_limit), len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    return train_loader, val_loader, frames


def train_one_epoch(
    model: DualEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RankingLoss,
    device: torch.device,
    grad_clip: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, specs in loader:
        images = images.to(device, non_blocking=True)
        specs = specs.to(device, non_blocking=True)
        z_sheet, z_audio = model(images, specs)
        loss = criterion(z_sheet, z_audio)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Check learning rate, data, or CCA stability.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += float(loss.detach()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_val_mrr(
    model: DualEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    images_all: list[torch.Tensor] = []
    specs_all: list[torch.Tensor] = []
    for images, specs in loader:
        images_all.append(images)
        specs_all.append(specs)
    images_all = torch.cat(images_all, dim=0).to(device)
    specs_all = torch.cat(specs_all, dim=0).to(device)
    z_sheet, z_audio = model(images_all, specs_all)
    scores = (z_sheet @ z_audio.T).float().cpu().numpy()
    if not np.isfinite(scores).all():
        return {
            "mrr_a2s": float("nan"),
            "mrr_s2a": float("nan"),
            "mrr_mean": float("nan"),
            "n": int(scores.shape[0]),
        }
    targets = np.arange(scores.shape[0])
    ranks_a2s = ranks_from_scores(scores, targets)
    ranks_s2a = ranks_from_scores(scores.T, targets)
    mrr_a2s = float((1.0 / ranks_a2s).mean())
    mrr_s2a = float((1.0 / ranks_s2a).mean())
    return {
        "mrr_a2s": mrr_a2s,
        "mrr_s2a": mrr_s2a,
        "mrr_mean": 0.5 * (mrr_a2s + mrr_s2a),
        "n": int(scores.shape[0]),
    }


def train_model(
    model: DualEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: RankingLoss,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], float, float]:
    history: list[dict[str, Any]] = []
    best_mrr = -1.0
    bad_epochs = 0
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.get("grad_clip"),
        )
        val_metrics = evaluate_val_mrr(model, val_loader, device)
        if not np.isfinite(val_metrics["mrr_mean"]):
            log("Validation MRR is not finite; stopping.")
            break
        scheduler.step(val_metrics["mrr_mean"])
        elapsed = time.time() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": elapsed,
        }
        history.append(record)
        log(
            f"epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val MRR A2S {val_metrics['mrr_a2s']:.4f} | "
            f"val MRR S2A {val_metrics['mrr_s2a']:.4f} | "
            f"lr {record['lr']:.2e} | {elapsed:.1f}s"
        )
        if val_metrics["mrr_mean"] > best_mrr:
            best_mrr = val_metrics["mrr_mean"]
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "sheet_encoder": model.sheet.state_dict(),
                    "audio_encoder": model.audio.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            log(f"  -> checkpoint saved (best val MRR {best_mrr:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["early_stop_patience"]):
                log(f"Early stop after {bad_epochs} epochs without improvement.")
                break
    return history, best_mrr, time.time() - started


def _vote_rank(
    labels_per_candidate: Sequence[str],
    true_label: str,
    label_to_id: dict[str, int],
    num_labels: int,
) -> int:
    votes = np.zeros(num_labels, dtype=np.int64)
    for label in labels_per_candidate:
        votes[label_to_id[label]] += 1
    true_id = label_to_id[true_label]
    order = np.argsort(-votes, kind="stable")
    return int(np.flatnonzero(order == true_id)[0]) + 1


def evaluate_task1(
    model: DualEncoder,
    config: dict[str, Any],
    data_dir: Path,
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    noteheads_test, windows_test, pairs_test = frames["test"]
    noteheads_test = noteheads_test.sort_values("crop_row").reset_index(drop=True)
    windows_test = windows_test.sort_values("window_row").reset_index(drop=True)
    notehead_ids = noteheads_test["notehead_id"].tolist()
    notehead_to_index = {
        notehead_id: index for index, notehead_id in enumerate(notehead_ids)
    }

    sheet_hidden = embed_image_rows(
        model.sheet,
        data_dir / "score_crops_test.npy",
        noteheads_test["crop_row"].to_numpy(),
        batch_size=int(config["eval_batch_size"]),
        device=device,
    )
    audio_hidden = embed_audio_rows(
        model.audio,
        data_dir / "audio_windows_test.npy",
        windows_test["window_row"].to_numpy(),
        batch_size=int(config["eval_batch_size"]),
        device=device,
    )
    if model.cca is not None:
        z_sheet = project_cca(
            sheet_hidden.to(device),
            model.cca.mean1.detach(),
            model.cca.U.detach(),
        ).cpu()
        z_audio = project_cca(
            audio_hidden.to(device),
            model.cca.mean2.detach(),
            model.cca.V.detach(),
        ).cpu()
    else:
        z_sheet = F.normalize(sheet_hidden, dim=1)
        z_audio = F.normalize(audio_hidden, dim=1)

    similarity = (z_audio @ z_sheet.T).numpy()
    if not np.isfinite(similarity).all():
        raise RuntimeError("Non-finite test similarity matrix.")

    window_targets = windows_test["notehead_id"].map(notehead_to_index).to_numpy()
    if np.isnan(window_targets).any():
        raise ValueError("Some test windows have no matching notehead.")
    window_targets = window_targets.astype(int)

    paired = one_window_per_notehead(
        build_positive_pairs(pairs_test, noteheads_test, windows_test)
    )
    paired_window_rows = paired["window_row_eval"].to_numpy()
    paired_targets = (
        paired["notehead_id"].map(notehead_to_index).to_numpy().astype(int)
    )
    similarity_paired = similarity[paired_window_rows]
    ranks_paired_a2s = ranks_from_scores(similarity_paired, paired_targets)
    inverse_paired_targets = np.empty_like(paired_targets)
    inverse_paired_targets[paired_targets] = np.arange(len(paired_targets))
    ranks_paired_s2a = ranks_from_scores(
        similarity_paired.T,
        inverse_paired_targets,
    )
    ranks_extended_a2s = ranks_from_scores(similarity, window_targets)
    target_lists = [
        np.flatnonzero(window_targets == index) for index in range(len(noteheads_test))
    ]
    ranks_extended_s2a = ranks_from_multi_scores(similarity.T, target_lists)
    rng = np.random.default_rng(int(config["seed"]))
    shuffled_queries = rng.permutation(len(paired))
    ranks_random = ranks_from_scores(
        similarity_paired[shuffled_queries],
        paired_targets,
    )

    snippet_metrics = {
        "paired_2018_audio_to_sheet": retrieval_metrics(ranks_paired_a2s),
        "paired_2018_sheet_to_audio": retrieval_metrics(ranks_paired_s2a),
        "extended_audio_to_sheet": retrieval_metrics(ranks_extended_a2s),
        "extended_sheet_to_audio": retrieval_metrics(ranks_extended_s2a),
        "random_paired_audio_to_sheet": retrieval_metrics(ranks_random),
    }

    nh_systems = pd.read_parquet(data_dir / "notehead_systems_test.parquet")
    systems = pd.read_parquet(data_dir / "systems_test.parquet")
    pairs_system = pd.read_parquet(data_dir / "pairs_system_test.parquet")
    system_ids = systems["system_id"].tolist()
    system_to_col = {
        system_id: index for index, system_id in enumerate(system_ids)
    }
    nh_systems["notehead_index"] = nh_systems["notehead_id"].map(notehead_to_index)
    nh_systems["notehead_index"] = nh_systems["notehead_index"].astype(int)
    system_notehead_indices = {
        system_id: nh_systems.loc[
            nh_systems["system_id"] == system_id, "notehead_index"
        ].to_numpy()
        for system_id in system_ids
    }
    system_scores = np.full(
        (len(windows_test), len(system_ids)),
        -np.inf,
        dtype=np.float32,
    )
    for system_id, indices in system_notehead_indices.items():
        if len(indices):
            system_scores[:, system_to_col[system_id]] = similarity[:, indices].max(axis=1)

    notehead_to_system = nh_systems.set_index("notehead_id")["system_id"].to_dict()
    window_true_system = windows_test["notehead_id"].map(notehead_to_system)
    if window_true_system.isna().any():
        raise ValueError("Some test windows have no system assignment.")
    window_true_system_col = (
        window_true_system.map(system_to_col).to_numpy().astype(int)
    )

    ranks_system_full = ranks_from_scores(system_scores, window_true_system_col)
    page_to_systems = systems.groupby("page_id")["system_id"].apply(list).to_dict()
    window_pages = windows_test["page_id"].to_numpy()
    ranks_system_page = np.empty(len(windows_test), dtype=np.int64)
    pred_system_page = np.empty(len(windows_test), dtype=object)
    for index in range(len(windows_test)):
        candidate_systems = page_to_systems[window_pages[index]]
        candidate_cols = np.asarray(
            [system_to_col[system_id] for system_id in candidate_systems],
            dtype=int,
        )
        true_col = window_true_system_col[index]
        scores = system_scores[index, candidate_cols]
        true_score = system_scores[index, true_col]
        ranks_system_page[index] = int((scores > true_score).sum()) + 1
        pred_system_page[index] = system_ids[int(candidate_cols[int(np.argmax(scores))])]

    ranks_system_5 = []
    pred_system_5 = []
    query_ids_5 = []
    window_rows_5 = []
    true_system_5 = []
    for _, group in pairs_system.groupby("query_id", sort=False):
        row = int(group["window_row"].iloc[0])
        candidate_cols = group["candidate_system_id"].map(system_to_col).to_numpy()
        scores = system_scores[row, candidate_cols]
        positive_id = group.loc[
            group["label"] == 1, "candidate_system_id"
        ].iloc[0]
        positive_position = int(
            np.flatnonzero(
                group["candidate_system_id"].to_numpy() == positive_id
            )[0]
        )
        ranks_system_5.append(
            int((scores > scores[positive_position]).sum()) + 1
        )
        pred_system_5.append(system_ids[int(candidate_cols[int(np.argmax(scores))])])
        query_ids_5.append(group["query_id"].iloc[0])
        window_rows_5.append(row)
        true_system_5.append(positive_id)
    ranks_system_5 = np.asarray(ranks_system_5)

    system_metrics = {
        "page_conditioned": retrieval_metrics(ranks_system_page),
        "full_collection": retrieval_metrics(ranks_system_full),
        "five_candidate": retrieval_metrics(ranks_system_5),
    }

    notehead_piece = noteheads_test["piece"].to_numpy()
    notehead_page = noteheads_test["page_id"].to_numpy()
    window_piece = windows_test["piece"].to_numpy()
    window_page = windows_test["page_id"].to_numpy()
    piece_names = sorted(set(window_piece))
    piece_to_id = {name: index for index, name in enumerate(piece_names)}
    page_names = sorted(set(window_page))
    page_to_id = {name: index for index, name in enumerate(page_names)}

    top_k = 25
    top_noteheads = np.argsort(-similarity, axis=1)[:, :top_k]
    ranks_piece_a2s, ranks_page_a2s = [], []
    for index in range(len(windows_test)):
        candidate_ids = top_noteheads[index]
        ranks_piece_a2s.append(
            _vote_rank(
                notehead_piece[candidate_ids],
                window_piece[index],
                piece_to_id,
                len(piece_names),
            )
        )
        ranks_page_a2s.append(
            _vote_rank(
                notehead_page[candidate_ids],
                window_page[index],
                page_to_id,
                len(page_names),
            )
        )

    top_windows = np.argsort(-similarity.T, axis=1)[:, :top_k]
    ranks_piece_s2a, ranks_page_s2a = [], []
    for index in range(len(noteheads_test)):
        candidate_ids = top_windows[index]
        ranks_piece_s2a.append(
            _vote_rank(
                window_piece[candidate_ids],
                notehead_piece[index],
                piece_to_id,
                len(piece_names),
            )
        )
        ranks_page_s2a.append(
            _vote_rank(
                window_page[candidate_ids],
                notehead_page[index],
                page_to_id,
                len(page_names),
            )
        )

    identification_metrics = {
        "piece_audio_to_sheet_top25": hit_rate_at_k(ranks_piece_a2s),
        "piece_sheet_to_audio_top25": hit_rate_at_k(ranks_piece_s2a),
        "page_audio_to_sheet_top25": hit_rate_at_k(ranks_page_a2s),
        "page_sheet_to_audio_top25": hit_rate_at_k(ranks_page_s2a),
    }

    top1_a2s = np.argmax(similarity, axis=1)
    pred_system_full = np.asarray(system_ids, dtype=object)[
        np.argmax(system_scores, axis=1)
    ]
    snippet_predictions = pd.DataFrame(
        {
            "window_id": windows_test["window_id"].to_numpy(),
            "true_notehead_id": noteheads_test["notehead_id"].to_numpy()[window_targets],
            "pred_notehead_id": noteheads_test["notehead_id"].to_numpy()[top1_a2s],
            "rank": ranks_extended_a2s,
            "score": similarity[np.arange(len(windows_test)), window_targets],
        }
    )
    system_predictions = pd.DataFrame(
        {
            "window_id": windows_test["window_id"].to_numpy(),
            "true_system_id": np.asarray(system_ids, dtype=object)[window_true_system_col],
            "pred_system_id_full": pred_system_full,
            "rank_full": ranks_system_full,
            "pred_system_id_page": pred_system_page,
            "rank_page": ranks_system_page,
        }
    )
    system_5cand_predictions = pd.DataFrame(
        {
            "query_id": query_ids_5,
            "window_row": window_rows_5,
            "true_system_id": true_system_5,
            "pred_system_id": pred_system_5,
            "rank": ranks_system_5,
        }
    )

    metrics = {
        "snippet_retrieval": snippet_metrics,
        "system_localization": system_metrics,
        "identification": identification_metrics,
        "rank_distributions": {
            "snippet_extended_audio_to_sheet": _rank_summary(ranks_extended_a2s),
            "system_page_conditioned": _rank_summary(ranks_system_page),
            "system_five_candidate": _rank_summary(ranks_system_5),
        },
    }
    predictions = {
        "snippet": snippet_predictions,
        "system": system_predictions,
        "system_5cand": system_5cand_predictions,
    }
    return metrics, predictions


def _rank_summary(ranks: Sequence[int]) -> dict[str, Any]:
    ranks_array = np.asarray(ranks, dtype=np.float64)
    return {
        "n": int(ranks_array.size),
        "mean": float(ranks_array.mean()),
        "median": float(np.median(ranks_array)),
        "p90": float(np.percentile(ranks_array, 90)),
        "max": float(ranks_array.max()),
    }


def plot_history(history: list[dict[str, Any]], fig_dir: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("ranking loss")
    axes[0].set_title("Training loss")
    axes[1].plot(frame["epoch"], frame["mrr_a2s"], marker="o", label="audio-to-sheet")
    axes[1].plot(frame["epoch"], frame["mrr_s2a"], marker="o", label="sheet-to-audio")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MRR")
    axes[1].set_title("Validation MRR")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snippet_metrics(metrics: dict[str, Any], fig_dir: Path) -> None:
    protocols = list(metrics.keys())
    metric_keys = ["R@1", "R@5", "R@10", "R@25"]
    x = np.arange(len(metric_keys))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 4))
    for index, protocol in enumerate(protocols):
        values = [metrics[protocol].get(metric, 0.0) for metric in metric_keys]
        ax.bar(x + (index - 2) * width, values, width, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_title("Task 1 - snippet retrieval")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "snippet_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_system_metrics(metrics: dict[str, Any], fig_dir: Path) -> None:
    protocols = list(metrics.keys())
    metric_keys = ["R@1", "R@5", "R@10", "R@25"]
    x = np.arange(len(metric_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, protocol in enumerate(protocols):
        values = [metrics[protocol].get(metric, 0.0) for metric in metric_keys]
        ax.bar(x + (index - 1) * width, values, width, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_title("Task 1 - system localization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "system_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_distribution(
    predictions: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        np.clip(predictions["rank"].to_numpy(), 1, 100),
        bins=50,
        color="#4c72b0",
    )
    ax.set_xlabel("rank (clipped at 100)")
    ax.set_ylabel("windows")
    ax.set_title("Task 1 - audio-to-sheet rank distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "retrieval_ranks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    global _LOG_HANDLE
    args = parse_args()
    config = normalize_config(load_config(args.config), args)
    if config["task"] != "task1":
        raise ValueError(f"Unsupported task: {config['task']}")

    run_id, run_dir = create_run_dir(config, args)
    _LOG_HANDLE = open(run_dir / "run.log", "w", encoding="utf-8")
    config["run_id"] = run_id
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    config["data_dir"] = str(data_dir)

    seed = int(config["seed"])
    set_seed(seed)
    configure_torch(config.get("threads"))
    device = resolve_device(str(config["device"]))
    log(f"run_id : {run_id}")
    log(f"config : {args.config}")
    log(f"device : {device}")
    log(f"torch  : {torch.__version__}")

    if config["experiment"] == "ranking_head":
        from scripts.ranking_head import run_ranking_head

        try:
            run_ranking_head(config, run_id, run_dir, device, data_dir, log)
        finally:
            _LOG_HANDLE.close()
            _LOG_HANDLE = None
        return
    if config["experiment"] != "dual_encoder":
        raise ValueError(f"Unsupported experiment: {config['experiment']}")

    train_loader, val_loader, frames = build_task1_loaders(config, data_dir)
    log(
        f"train batches {len(train_loader):,} | val pairs {len(val_loader.dataset):,}"
    )

    model = DualEncoder(
        embed_dim=int(config["embed_dim"]),
        use_cca=bool(config["use_cca"]),
        cca_kwargs=config["cca"],
        channels=config["channels"],
    ).to(device)
    criterion = RankingLoss(margin=float(config["margin"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    checkpoint_path = DEFAULT_CHECKPOINT_DIR / f"{run_id}.pt"
    save_json(run_dir / "config.json", {**config, "checkpoint": str(checkpoint_path)})

    history, best_mrr, training_seconds = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config,
        device,
        checkpoint_path,
    )
    write_history_csv(history, run_dir / "history.csv")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log(f"loaded epoch {checkpoint.get('epoch')} | val {checkpoint.get('val_metrics')}")

    test_metrics, predictions = evaluate_task1(
        model,
        config,
        data_dir,
        frames,
        device,
    )
    predictions["snippet"].to_csv(run_dir / "predictions.csv", index=False)
    predictions["system"].to_csv(run_dir / "predictions_system.csv", index=False)
    predictions["system_5cand"].to_csv(
        run_dir / "predictions_system_5cand.csv",
        index=False,
    )

    plot_history(history, run_dir / "figures")
    plot_snippet_metrics(test_metrics["snippet_retrieval"], run_dir / "figures")
    plot_system_metrics(test_metrics["system_localization"], run_dir / "figures")
    plot_rank_distribution(predictions["snippet"], run_dir / "figures")

    payload = {
        "run_id": run_id,
        "task": config["task"],
        "experiment": config["experiment"],
        "config": config,
        "seed": seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "best_val_mrr": best_mrr,
        "training_seconds": training_seconds,
        "test": test_metrics,
        "history": history,
    }
    save_json(run_dir / "metrics.json", payload)
    log(f"test system page R@1 {test_metrics['system_localization']['page_conditioned']['R@1']:.4f}")
    log(f"test system page R@5 {test_metrics['system_localization']['page_conditioned']['R@5']:.4f}")
    log(f"test 5-candidate R@1 {test_metrics['system_localization']['five_candidate']['R@1']:.4f}")
    log(f"test snippet MRR {test_metrics['snippet_retrieval']['paired_2018_audio_to_sheet']['MRR']:.4f}")
    log(f"saved : {run_dir}")
    _LOG_HANDLE.close()
    _LOG_HANDLE = None


if __name__ == "__main__":
    main()
