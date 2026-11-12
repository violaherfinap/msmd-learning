"""Error analysis for Task 1 system localization runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
EXPERIMENTS_DIR = ROOT / "reports" / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "comparison"
RUNS = ["t1_c1_baseline", "t1_c2_tuned", "t1_c3_architecture", "t1_c4_ranking_head"]


def _page_label(row: pd.Series) -> str:
    piece = str(row["piece"]).split("__")[0]
    page_num = int(row["page_num"])
    return f"{piece}-p{page_num:02d}"


def _rank_metrics(values: pd.Series) -> pd.Series:
    ranks = values.to_numpy(dtype=np.float64)
    return pd.Series(
        {
            "n": int(ranks.size),
            "R@1": float((ranks <= 1).mean()),
            "R@5": float((ranks <= 5).mean()),
            "MRR": float((1.0 / ranks).mean()),
            "median_rank": float(np.median(ranks)),
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = pd.read_parquet(DATA_DIR / "systems_test.parquet")
    page_system_counts = systems.groupby("page_id").size()
    page_by_system = systems.set_index("system_id")["page_id"].to_dict()
    page_labels = (
        systems.drop_duplicates("page_id")
        .assign(label=lambda frame: frame.apply(_page_label, axis=1))
        .set_index("page_id")["label"]
        .to_dict()
    )

    page_rows = []
    size_rows = []
    confusion_rows = []
    for run_id in RUNS:
        path = EXPERIMENTS_DIR / run_id / "predictions_system.csv"
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        predictions["true_page"] = predictions["true_system_id"].map(page_by_system)
        predictions["n_systems_page"] = predictions["true_page"].map(page_system_counts)
        predictions["correct"] = predictions["pred_system_id_page"] == predictions["true_system_id"]

        for page, group in predictions.groupby("true_page"):
            metrics = _rank_metrics(group["rank_page"])
            page_rows.append(
                {
                    "run_id": run_id,
                    "page": page,
                    "page_label": page_labels.get(page, page),
                    **metrics.to_dict(),
                }
            )
        for count, group in predictions.groupby("n_systems_page"):
            metrics = _rank_metrics(group["rank_page"])
            size_rows.append(
                {
                    "run_id": run_id,
                    "n_systems_page": int(count),
                    **metrics.to_dict(),
                }
            )
        wrong = predictions.loc[~predictions["correct"]]
        confusions = (
            wrong.groupby(["true_system_id", "pred_system_id_page"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        for _, row in confusions.iterrows():
            confusion_rows.append(
                {
                    "run_id": run_id,
                    "true_system_id": row["true_system_id"],
                    "pred_system_id": row["pred_system_id_page"],
                    "count": int(row["count"]),
                }
            )

    page_frame = pd.DataFrame(page_rows)
    size_frame = pd.DataFrame(size_rows)
    confusion_frame = pd.DataFrame(confusion_rows)
    page_frame.to_csv(OUTPUT_DIR / "error_by_page.csv", index=False)
    size_frame.to_csv(OUTPUT_DIR / "error_by_page_size.csv", index=False)
    confusion_frame.to_csv(OUTPUT_DIR / "top_confusions.csv", index=False)

    if not page_frame.empty:
        pivot = page_frame.pivot(index="page_label", columns="run_id", values="R@1")
        fig, ax = plt.subplots(figsize=(11, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Page-conditioned R@1")
        ax.set_ylim(0, 1.02)
        ax.set_title("Task 1 - page-conditioned R@1 by test page")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "page_r1_by_page.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not size_frame.empty:
        pivot = size_frame.pivot(index="n_systems_page", columns="run_id", values="R@1")
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("systems on the page")
        ax.set_ylabel("Page-conditioned R@1")
        ax.set_ylim(0, 1.02)
        ax.set_title("Task 1 - R@1 by page size")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "page_r1_by_page_size.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(page_frame.to_string(index=False))
    print()
    print(size_frame.to_string(index=False))
    print()
    print(confusion_frame.to_string(index=False))


if __name__ == "__main__":
    main()
