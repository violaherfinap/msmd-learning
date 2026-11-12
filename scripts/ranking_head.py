"""Learned listwise ranking head for Task 1 system localization."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.baseline_dorfer import (
    DualEncoder,
    embed_audio_rows,
    embed_image_rows,
    load_checkpoint,
    load_split_frames,
    project_cca,
    ranks_from_scores,
    retrieval_metrics,
    save_json,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCALARS = 5


class CandidateRanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _load_source_model(
    source_checkpoint: Path,
    device: torch.device,
) -> tuple[DualEncoder, dict[str, Any]]:
    checkpoint = load_checkpoint(source_checkpoint, map_location=device)
    source_config = checkpoint.get("config", {})
    model = DualEncoder(
        embed_dim=int(source_config.get("embed_dim", 32)),
        use_cca=bool(source_config.get("use_cca", False)),
        cca_kwargs=source_config.get("cca"),
        channels=source_config.get("channels", (24, 48, 96, 96)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def _embed_split(
    model: DualEncoder,
    data_dir: Path,
    split: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    noteheads, windows, pairs = load_split_frames(data_dir, split)
    noteheads = noteheads.sort_values("crop_row").reset_index(drop=True)
    windows = windows.sort_values("window_row").reset_index(drop=True)
    notehead_ids = noteheads["notehead_id"].tolist()
    notehead_to_index = {
        notehead_id: index for index, notehead_id in enumerate(notehead_ids)
    }

    sheet_hidden = embed_image_rows(
        model.sheet,
        data_dir / f"score_crops_{split}.npy",
        noteheads["crop_row"].to_numpy(),
        batch_size=int(config["eval_batch_size"]),
        device=device,
    )
    audio_hidden = embed_audio_rows(
        model.audio,
        data_dir / f"audio_windows_{split}.npy",
        windows["window_row"].to_numpy(),
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

    systems = pd.read_parquet(data_dir / f"systems_{split}.parquet")
    nh_systems = pd.read_parquet(data_dir / f"notehead_systems_{split}.parquet")
    pairs_system = pd.read_parquet(data_dir / f"pairs_system_{split}.parquet")
    system_ids = systems["system_id"].tolist()
    system_to_col = {
        system_id: index for index, system_id in enumerate(system_ids)
    }
    nh_systems["notehead_index"] = nh_systems["notehead_id"].map(notehead_to_index)
    if nh_systems["notehead_index"].isna().any():
        raise ValueError(f"{split}: notehead_systems contains unknown noteheads.")
    nh_systems["notehead_index"] = nh_systems["notehead_index"].astype(int)
    system_members = {
        system_id: nh_systems.loc[
            nh_systems["system_id"] == system_id, "notehead_index"
        ].to_numpy(dtype=np.int64)
        for system_id in system_ids
    }
    page_to_systems = systems.groupby("page_id")["system_id"].apply(list).to_dict()
    notehead_to_system = nh_systems.set_index("notehead_id")["system_id"].to_dict()
    window_true_system = windows["notehead_id"].map(notehead_to_system)
    if window_true_system.isna().any():
        raise ValueError(f"{split}: some windows have no system assignment.")
    window_true_system_col = (
        window_true_system.map(system_to_col).to_numpy().astype(int)
    )

    return {
        "split": split,
        "noteheads": noteheads,
        "windows": windows,
        "pairs": pairs,
        "pairs_system": pairs_system,
        "sheet": z_sheet.numpy().astype(np.float32),
        "audio": z_audio.numpy().astype(np.float32),
        "system_ids": system_ids,
        "system_to_col": system_to_col,
        "system_members": system_members,
        "page_to_systems": page_to_systems,
        "window_page": windows["page_id"].to_numpy(),
        "window_true_system_col": window_true_system_col,
    }


def _candidate_features(
    audio_vec: np.ndarray,
    member_indices: np.ndarray,
    sheet_embeddings: np.ndarray,
    embed_dim: int,
) -> np.ndarray:
    if len(member_indices) == 0:
        return np.zeros(FEATURE_SCALARS + 4 * embed_dim, dtype=np.float32)
    similarities = sheet_embeddings[member_indices] @ audio_vec
    best_member = int(member_indices[int(np.argmax(similarities))])
    best_sheet = sheet_embeddings[best_member]
    top5_mean = float(np.sort(similarities)[-5:].mean())
    return np.concatenate(
        [
            np.asarray(
                [
                    float(similarities.max()),
                    float(similarities.mean()),
                    top5_mean,
                    float(similarities.min()),
                    float(similarities.std()),
                ],
                dtype=np.float32,
            ),
            best_sheet,
            audio_vec,
            best_sheet * audio_vec,
            np.abs(best_sheet - audio_vec),
        ]
    ).astype(np.float32)


def _get_features(
    split_data: dict[str, Any],
    row: int,
    system_id: str,
    embed_dim: int,
    cache: dict[tuple[int, str], np.ndarray] | None,
) -> np.ndarray:
    key = (int(row), str(system_id))
    if cache is not None and key in cache:
        return cache[key]
    features = _candidate_features(
        split_data["audio"][int(row)],
        split_data["system_members"].get(system_id, np.empty(0, dtype=np.int64)),
        split_data["sheet"],
        embed_dim,
    )
    if cache is not None:
        cache[key] = features
    return features


def _build_fixed_groups(
    split_data: dict[str, Any],
    embed_dim: int,
    cache: dict[tuple[int, str], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for _, group in split_data["pairs_system"].groupby("query_id", sort=False):
        row = int(group["window_row"].iloc[0])
        candidate_ids = group["candidate_system_id"].tolist()
        target = int(np.flatnonzero(group["label"].to_numpy() == 1)[0])
        features = np.stack(
            [
                _get_features(split_data, row, system_id, embed_dim, cache)
                for system_id in candidate_ids
            ]
        )
        groups.append(
            {
                "query_id": group["query_id"].iloc[0],
                "window_row": row,
                "system_ids": candidate_ids,
                "features": features,
                "target": target,
            }
        )
    return groups


def _build_hard_groups(
    model: CandidateRanker,
    base_groups: Sequence[dict[str, Any]],
    split_data: dict[str, Any],
    embed_dim: int,
    cache: dict[tuple[int, str], np.ndarray],
    device: torch.device,
    num_negatives: int = 4,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    model.eval()
    for base in base_groups:
        row = int(base["window_row"])
        true_id = base["system_ids"][int(base["target"])]
        original_negatives = [
            system_id
            for system_id in base["system_ids"]
            if system_id != true_id
        ]
        page_systems = [
            system_id
            for system_id in split_data["page_to_systems"].get(
                split_data["window_page"][row], []
            )
            if system_id != true_id
        ]
        candidate_ids = list(dict.fromkeys(page_systems + original_negatives))
        if not candidate_ids:
            groups.append(base)
            continue
        features = np.stack(
            [
                _get_features(split_data, row, system_id, embed_dim, cache)
                for system_id in candidate_ids
            ]
        )
        with torch.no_grad():
            scores = model(
                torch.from_numpy(features).to(device)
            ).detach().cpu().numpy()
        order = np.argsort(-scores)
        negatives = [candidate_ids[int(index)] for index in order[:num_negatives]]
        selected_ids = [true_id] + negatives
        selected_features = np.stack(
            [
                _get_features(split_data, row, system_id, embed_dim, cache)
                for system_id in selected_ids
            ]
        )
        groups.append(
            {
                "query_id": base["query_id"],
                "window_row": row,
                "system_ids": selected_ids,
                "features": selected_features,
                "target": 0,
            }
        )
    return groups


def _batch_tensors(
    groups: Sequence[dict[str, Any]],
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(groups)
    max_candidates = max(len(group["system_ids"]) for group in groups)
    features = torch.zeros(batch, max_candidates, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch, max_candidates, dtype=torch.bool)
    targets = torch.zeros(batch, dtype=torch.long)
    for index, group in enumerate(groups):
        count = len(group["system_ids"])
        features[index, :count] = torch.from_numpy(group["features"])
        mask[index, :count] = True
        targets[index] = int(group["target"])
    return features.to(device), mask.to(device), targets.to(device)


def _train_epoch(
    model: CandidateRanker,
    groups: Sequence[dict[str, Any]],
    feature_dim: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_queries: int,
    grad_clip: float | None,
) -> float:
    model.train()
    order = np.random.permutation(len(groups))
    total_loss = 0.0
    total_items = 0
    for start in range(0, len(order), batch_queries):
        batch = [groups[int(index)] for index in order[start:start + batch_queries]]
        features, mask, targets = _batch_tensors(batch, feature_dim, device)
        batch_size, max_candidates, _ = features.shape
        scores = model(features.reshape(-1, feature_dim)).reshape(
            batch_size, max_candidates
        )
        scores = scores.masked_fill(~mask, -1e9)
        loss = F.cross_entropy(scores, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite ranking loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
        total_items += len(batch)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _evaluate_groups(
    model: CandidateRanker,
    groups: Sequence[dict[str, Any]],
    feature_dim: int,
    device: torch.device,
    batch_queries: int = 256,
) -> dict[str, Any]:
    model.eval()
    ranks: list[int] = []
    for start in range(0, len(groups), batch_queries):
        batch = groups[start:start + batch_queries]
        features, mask, targets = _batch_tensors(batch, feature_dim, device)
        batch_size, max_candidates, _ = features.shape
        scores = model(features.reshape(-1, feature_dim)).reshape(
            batch_size, max_candidates
        )
        scores = scores.masked_fill(~mask, -1e9).cpu().numpy()
        for index, group in enumerate(batch):
            target = int(group["target"])
            ranks.append(int((scores[index] > scores[index, target]).sum()) + 1)
    ranks_array = np.asarray(ranks, dtype=np.float64)
    return {
        **retrieval_metrics(ranks_array),
        "accuracy": float((ranks_array <= 1).mean()),
    }


@torch.no_grad()
def _system_score_matrix(
    model: CandidateRanker,
    split_data: dict[str, Any],
    embed_dim: int,
    feature_dim: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    system_ids = split_data["system_ids"]
    scores = np.empty(
        (len(split_data["windows"]), len(system_ids)),
        dtype=np.float32,
    )
    for row in range(len(split_data["windows"])):
        features = np.stack(
            [
                _get_features(split_data, row, system_id, embed_dim, None)
                for system_id in system_ids
            ]
        )
        scores[row] = model(
            torch.from_numpy(features).to(device)
        ).detach().cpu().numpy()
    return scores


def _evaluate_system_localization(
    model: CandidateRanker,
    split_data: dict[str, Any],
    embed_dim: int,
    feature_dim: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    system_scores = _system_score_matrix(
        model,
        split_data,
        embed_dim,
        feature_dim,
        device,
    )
    true_cols = split_data["window_true_system_col"]
    ranks_full = ranks_from_scores(system_scores, true_cols)
    system_ids = split_data["system_ids"]
    windows = split_data["windows"]
    page_ranks = np.empty(len(windows), dtype=np.int64)
    page_pred = np.empty(len(windows), dtype=object)
    for row in range(len(windows)):
        candidate_systems = split_data["page_to_systems"].get(
            split_data["window_page"][row], []
        )
        candidate_cols = np.asarray(
            [split_data["system_to_col"][system_id] for system_id in candidate_systems],
            dtype=int,
        )
        true_col = int(true_cols[row])
        candidate_scores = system_scores[row, candidate_cols]
        page_ranks[row] = int((candidate_scores > system_scores[row, true_col]).sum()) + 1
        page_pred[row] = system_ids[int(candidate_cols[int(np.argmax(candidate_scores))])]

    five_ranks: list[int] = []
    five_rows: list[dict[str, Any]] = []
    for _, group in split_data["pairs_system"].groupby("query_id", sort=False):
        row = int(group["window_row"].iloc[0])
        candidate_ids = group["candidate_system_id"].tolist()
        candidate_cols = np.asarray(
            [split_data["system_to_col"][system_id] for system_id in candidate_ids],
            dtype=int,
        )
        candidate_scores = system_scores[row, candidate_cols]
        positive_id = group.loc[
            group["label"] == 1, "candidate_system_id"
        ].iloc[0]
        positive_position = int(
            np.flatnonzero(group["candidate_system_id"].to_numpy() == positive_id)[0]
        )
        rank = int((candidate_scores > candidate_scores[positive_position]).sum()) + 1
        five_ranks.append(rank)
        five_rows.append(
            {
                "query_id": group["query_id"].iloc[0],
                "window_id": windows["window_id"].iloc[row],
                "true_system_id": positive_id,
                "pred_system_id": candidate_ids[int(np.argmax(candidate_scores))],
                "rank": rank,
            }
        )

    metrics = {
        "five_candidate": retrieval_metrics(np.asarray(five_ranks)),
        "page_conditioned": retrieval_metrics(page_ranks),
        "full_collection": retrieval_metrics(ranks_full),
    }
    predictions = {
        "five_candidate": pd.DataFrame(five_rows),
        "system": pd.DataFrame(
            {
                "window_id": windows["window_id"].to_numpy(),
                "true_system_id": np.asarray(system_ids, dtype=object)[true_cols],
                "pred_system_id_full": np.asarray(system_ids, dtype=object)[
                    np.argmax(system_scores, axis=1)
                ],
                "rank_full": ranks_full,
                "pred_system_id_page": page_pred,
                "rank_page": page_ranks,
            }
        ),
    }
    return metrics, predictions


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _plot_history(history: list[dict[str, Any]], fig_dir: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("listwise loss")
    axes[0].set_title("Training loss")
    axes[1].plot(frame["epoch"], frame["val_mrr"], marker="o", label="val MRR")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], marker="o", label="val top-1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Validation ranking")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_system_metrics(metrics: dict[str, Any], fig_dir: Path) -> None:
    protocols = ["five_candidate", "page_conditioned", "full_collection"]
    labels = ["5-candidate", "page-conditioned", "full collection"]
    metric_keys = ["R@1", "R@5", "R@10", "R@25"]
    x = np.arange(len(metric_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, (key, label) in enumerate(zip(protocols, labels)):
        values = [metrics[key].get(metric, 0.0) for metric in metric_keys]
        ax.bar(x + (index - 1) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_title("Task 1 - learned ranking head")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "system_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_ranking_head(
    config: dict[str, Any],
    run_id: str,
    run_dir: Path,
    device: torch.device,
    data_dir: Path,
    log_fn: Callable[[str], None],
) -> None:
    started = time.time()
    source_checkpoint = Path(config["source_checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    log_fn(f"source checkpoint : {source_checkpoint}")
    log_fn(f"data dir          : {data_dir}")

    encoder, source_meta = _load_source_model(source_checkpoint, device)
    embed_dim = int(source_meta.get("config", {}).get("embed_dim", 32))
    feature_dim = FEATURE_SCALARS + 4 * embed_dim
    log_fn(f"embed dim         : {embed_dim}")
    log_fn(f"feature dim       : {feature_dim}")

    splits = {}
    for split in ("train", "val", "test"):
        log_fn(f"embedding {split}...")
        splits[split] = _embed_split(encoder, data_dir, split, config, device)

    train_cache: dict[tuple[int, str], np.ndarray] = {}
    val_cache: dict[tuple[int, str], np.ndarray] = {}
    log_fn("building candidate features...")
    train_groups = _build_fixed_groups(splits["train"], embed_dim, train_cache)
    val_groups = _build_fixed_groups(splits["val"], embed_dim, val_cache)
    test_groups = _build_fixed_groups(splits["test"], embed_dim, {})

    train_limit = config.get("train_limit")
    val_limit = config.get("val_limit")
    if train_limit:
        train_groups = train_groups[: int(train_limit)]
    if val_limit:
        val_groups = val_groups[: int(val_limit)]
    base_train_groups = train_groups
    log_fn(f"train queries {len(train_groups):,} | val queries {len(val_groups):,}")

    model = CandidateRanker(
        input_dim=feature_dim,
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
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
    checkpoint_path = ROOT / "checkpoints" / "experiments" / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_mrr = -1.0
    bad_epochs = 0
    mining = config.get("hard_negative_mining", {}) or {}
    mining_enabled = bool(mining.get("enabled", False))
    mining_every = max(int(mining.get("every_epochs", 2)), 1)
    batch_queries = int(config.get("batch_queries", 64))
    grad_clip = config.get("grad_clip")
    patience = int(config.get("early_stop_patience", 3))

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        if mining_enabled and epoch > 1 and epoch % mining_every == 0:
            log_fn(f"epoch {epoch:02d}: rebuilding hard negatives")
            train_groups = _build_hard_groups(
                model,
                base_train_groups,
                splits["train"],
                embed_dim,
                train_cache,
                device,
                num_negatives=int(config.get("num_candidates", 5)) - 1,
            )
        train_loss = _train_epoch(
            model,
            train_groups,
            feature_dim,
            optimizer,
            device,
            batch_queries,
            grad_clip,
        )
        val_metrics = _evaluate_groups(
            model,
            val_groups,
            feature_dim,
            device,
            batch_queries=256,
        )
        if not np.isfinite(val_metrics["MRR"]):
            log_fn("Validation MRR is not finite; stopping.")
            break
        scheduler.step(val_metrics["MRR"])
        elapsed = time.time() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mrr": val_metrics["MRR"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mrr_at_5": val_metrics.get("R@5", float("nan")),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": elapsed,
        }
        history.append(record)
        log_fn(
            f"epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val MRR {val_metrics['MRR']:.4f} | "
            f"val top-1 {val_metrics['accuracy']:.4f} | "
            f"lr {record['lr']:.2e} | {elapsed:.1f}s"
        )
        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "source_checkpoint": str(source_checkpoint),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            log_fn(f"  -> checkpoint saved (best val MRR {best_mrr:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log_fn(f"Early stop after {bad_epochs} epochs without improvement.")
                break

    _write_history(history, run_dir / "history.csv")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log_fn(f"loaded epoch {checkpoint.get('epoch')} | val {checkpoint.get('val_metrics')}")

    test_metrics, predictions = _evaluate_system_localization(
        model,
        splits["test"],
        embed_dim,
        feature_dim,
        device,
    )
    predictions["five_candidate"].to_csv(run_dir / "predictions.csv", index=False)
    predictions["system"].to_csv(run_dir / "predictions_system.csv", index=False)
    _plot_history(history, run_dir / "figures")
    _plot_system_metrics(test_metrics, run_dir / "figures")

    payload = {
        "run_id": run_id,
        "task": config["task"],
        "experiment": config["experiment"],
        "config": {**config, "checkpoint": str(checkpoint_path)},
        "seed": int(config["seed"]),
        "device": str(device),
        "source_checkpoint": str(source_checkpoint),
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "best_val_mrr": best_mrr,
        "training_seconds": time.time() - started,
        "test": test_metrics,
        "history": history,
    }
    save_json(run_dir / "metrics.json", payload)
    save_json(
        run_dir / "config.json",
        {**config, "checkpoint": str(checkpoint_path)},
    )
    log_fn(
        f"test 5-candidate R@1 "
        f"{test_metrics['five_candidate']['R@1']:.4f}"
    )
    log_fn(
        f"test page R@1 "
        f"{test_metrics['page_conditioned']['R@1']:.4f} | "
        f"R@5 {test_metrics['page_conditioned']['R@5']:.4f}"
    )
    log_fn(
        f"test full R@1 "
        f"{test_metrics['full_collection']['R@1']:.4f}"
    )
    log_fn(f"saved : {run_dir}")
