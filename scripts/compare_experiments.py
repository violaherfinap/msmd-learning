"""Build the Task 1 experiment comparison table and figure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "reports" / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "comparison"


def _nested(payload: dict[str, Any], *keys: str, default: float | None = None):
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_metrics(path: Path) -> dict[str, Any] | None:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "test" not in payload:
        return None
    run_id = payload.get("run_id", path.parent.name)
    experiment = payload.get("experiment", "unknown")
    test = payload["test"]
    if "system_localization" in test:
        page = test["system_localization"].get("page_conditioned", {})
        five = test["system_localization"].get("five_candidate", {})
        snippet = _nested(test, "snippet_retrieval", "paired_2018_audio_to_sheet", default={}) or {}
        full = test["system_localization"].get("full_collection", {})
    else:
        page = test.get("page_conditioned", {})
        five = test.get("five_candidate", {})
        full = test.get("full_collection", {})
        snippet = {}
    return {
        "run_id": run_id,
        "experiment": experiment,
        "seed": payload.get("seed"),
        "best_epoch": payload.get("best_epoch"),
        "val_mrr": payload.get("best_val_mrr"),
        "val_protocol": "5-cand" if experiment == "ranking_head" else "snippet",
        "page_r1": page.get("R@1"),
        "page_r5": page.get("R@5"),
        "page_mrr": page.get("MRR"),
        "five_r1": five.get("R@1"),
        "five_mrr": five.get("MRR"),
        "full_r1": full.get("R@1"),
        "snippet_mrr": snippet.get("MRR"),
        "training_seconds": payload.get("training_seconds"),
        "source_checkpoint": payload.get("source_checkpoint"),
        "metrics_path": str(path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(EXPERIMENTS_DIR.glob("*/metrics.json")):
        if path.parent.name == "comparison":
            continue
        if path.parent.name.endswith("_smoke"):
            continue
        row = extract_metrics(path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No experiment metrics found.")

    frame = pd.DataFrame(rows)
    preference = {
        "t1_c1_baseline": 1,
        "t1_c2_tuned": 2,
        "t1_c3_architecture": 3,
        "t1_c4_ranking_head": 4,
    }
    frame["order"] = frame["run_id"].map(preference).fillna(99)
    frame = frame.sort_values(["order", "run_id"]).drop(columns=["order"])
    frame.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    columns = [
        "run_id",
        "experiment",
        "val_mrr",
        "val_protocol",
        "page_r1",
        "page_r5",
        "five_r1",
        "five_mrr",
        "snippet_mrr",
        "training_seconds",
    ]
    display = frame[columns].copy()
    for column in ("val_mrr", "page_r1", "page_r5", "five_r1", "five_mrr", "snippet_mrr"):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )
    display["training_seconds"] = display["training_seconds"].map(
        lambda value: "" if pd.isna(value) else f"{float(value) / 60.0:.1f}"
    )
    display = display.rename(
        columns={
            "run_id": "Run",
            "experiment": "Experiment",
            "val_mrr": "Val MRR",
            "val_protocol": "Val protocol",
            "page_r1": "Page R@1",
            "page_r5": "Page R@5",
            "five_r1": "5-cand R@1",
            "five_mrr": "5-cand MRR",
            "snippet_mrr": "Snippet MRR",
            "training_seconds": "Minutes",
        }
    )
    lines = [
        "# Task 1 experiment comparison",
        "",
        "| " + " | ".join(display.columns) + " |",
        "|" + "|".join(["---"] * len(display.columns)) + "|",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    (OUTPUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics = ["page_r1", "page_r5", "five_r1", "five_mrr"]
    labels = ["Page R@1", "Page R@5", "5-cand R@1", "5-cand MRR"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(frame), 1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for index, (_, row) in enumerate(frame.iterrows()):
        values = [0.0 if row[metric] is None else float(row[metric]) for metric in metrics]
        ax.bar(x + (index - (len(frame) - 1) / 2) * width, values, width, label=row["run_id"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Task 1 - system localization comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    snippet = frame.dropna(subset=["snippet_mrr"])
    if not snippet.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(snippet["run_id"], snippet["snippet_mrr"], color="#4c72b0")
        ax.set_ylabel("MRR")
        ax.set_title("Task 1 - snippet retrieval (paired 2018)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "snippet_mrr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(display.to_string(index=False))
    print("saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
