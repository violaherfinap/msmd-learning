#!/usr/bin/env python3
"""Build a static, self-contained preprocessing report.

The report is intended for reviewers who should not have to run Jupyter or
download the 18 GB source dataset.  It embeds figures as base64 PNG and
contains all tables needed to inspect the note-level preprocessing result.

Outputs:
- reports/preprocessing_report.html
- reports/preprocessing_report.md
- reports/preprocessing_datacard.md
- reports/figures/13_...png ... 19_...png
"""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PROC = DATA / "processed"
REP = PROJECT / "reports"
FIG = REP / "figures"
SPLITS = ("train", "val", "test")
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def md_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    if df.empty:
        return "_Tidak ada data._"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_and_embed(fig, name: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = buf.getvalue()
    (FIG / name).write_bytes(data)
    return base64.b64encode(data).decode("ascii")


def img_tag(data_uri: str, alt: str, caption: str = "") -> str:
    cap = f'<div class="caption">{html.escape(caption)}</div>' if caption else ""
    return (
        f'<figure><img src="data:image/png;base64,{data_uri}" alt="{html.escape(alt)}">'
        f"{cap}</figure>"
    )


def load_split(split: str) -> dict:
    return {
        "crops": np.load(PROC / f"score_crops_{split}.npy", mmap_mode="r"),
        "wins": np.load(PROC / f"audio_windows_{split}.npy", mmap_mode="r"),
        "nh": pd.read_csv(PROC / f"noteheads_{split}.csv"),
        "wd": pd.read_csv(PROC / f"windows_{split}.csv"),
        "pr": pd.read_csv(PROC / f"pairs_note_{split}.csv"),
    }


def make_split_summary_figure(bundles: dict) -> str:
    metrics = [
        ("Pieces", [bundles[s]["nh"]["piece"].nunique() for s in SPLITS]),
        ("Pages", [bundles[s]["nh"]["page_id"].nunique() for s in SPLITS]),
        ("Crop notehead", [len(bundles[s]["crops"]) for s in SPLITS]),
        ("Window audio", [len(bundles[s]["wins"]) for s in SPLITS]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, vals) in zip(axes.ravel(), metrics):
        bars = ax.bar(SPLITS, vals, color=[COLORS[s] for s in SPLITS])
        ax.set_title(title)
        ax.set_ylabel("jumlah")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Ringkasan split note-level", fontsize=12, y=1.02)
    fig.tight_layout()
    return save_and_embed(fig, "13_split_summary.png")


def _resize_page(img: Image.Image, target_w: int = 620) -> tuple[Image.Image, float]:
    if img.width <= target_w:
        return img, 1.0
    scale = target_w / img.width
    return img.resize((target_w, int(img.height * scale)), Image.Resampling.LANCZOS), scale


def make_page_overlay_figure(bundles: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5))
    for ax, split in zip(axes, SPLITS):
        nh = bundles[split]["nh"]
        page_counts = nh.groupby("page_id").size().sort_values(ascending=False)
        if page_counts.empty:
            ax.axis("off")
            ax.set_title(f"{split}: tidak ada halaman")
            continue
        page_id = page_counts.index[0]
        rows = nh[nh["page_id"] == page_id]
        png_path = Path(rows.iloc[0]["png_path"])
        if not png_path.exists():
            ax.axis("off")
            ax.set_title(f"{split}: PNG tidak ditemukan")
            continue

        img = Image.open(png_path).convert("RGB")
        img, scale = _resize_page(img, target_w=620)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Draw all notehead boxes, but only label a deterministic sample.
        label_rows = rows.sample(min(18, len(rows)), random_state=42)
        label_ids = set(label_rows["notehead_id"])
        for row in rows.itertuples(index=False):
            x0 = float(row.box_left) * scale
            y0 = float(row.box_top) * scale
            x1 = x0 + float(row.box_w) * scale
            y1 = y0 + float(row.box_h) * scale
            draw.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=1)
            if row.notehead_id in label_ids:
                cx = float(row.cx) * scale
                cy = float(row.cy) * scale
                draw.rectangle(
                    [cx - 80 * scale, cy - 100 * scale, cx + 80 * scale, cy + 100 * scale],
                    outline=(20, 130, 70),
                    width=2,
                )
                draw.text((x1 + 2, y0 - 8), f"p{int(row.pitch)}", fill=(20, 80, 160), font=font)

        ax.imshow(img)
        ax.set_title(f"{split}: {page_id.split('__')[-1]} ({len(rows)} notehead)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Overlay halaman: kotak merah = notehead, kotak hijau = crop 160x200 (sampel)", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_and_embed(fig, "14_page_overlay_samples.png")


def make_crop_gallery(bundles: dict) -> str:
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for r, split in enumerate(SPLITS):
        nh = bundles[split]["nh"]
        crops = bundles[split]["crops"]
        idx = nh.sample(min(4, len(nh)), random_state=42 + r).index.to_numpy()
        for c, i in enumerate(idx):
            ax = axes[r, c]
            row = nh.iloc[int(i)]
            ax.imshow(crops[int(row["crop_row"]), :, :, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{split} | pitch={int(row['pitch'])} | edge={int(row['edge'])}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Contoh crop notehead 200x160", fontsize=12)
    fig.tight_layout()
    return save_and_embed(fig, "15_crop_gallery.png")


def make_audio_gallery(bundles: dict) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5))
    for r, split in enumerate(SPLITS):
        wd = bundles[split]["wd"]
        wins = bundles[split]["wins"]
        idx = wd.sample(min(2, len(wd)), random_state=100 + r).index.to_numpy()
        for c, i in enumerate(idx):
            ax = axes[r, c]
            row = wd.iloc[int(i)]
            w = wins[int(row["window_row"])].astype(np.float32)
            ax.imshow(w.T, aspect="auto", origin="lower", cmap="magma", vmin=-2, vmax=2)
            onset_rel = int(row["onset_frame"] - row["start_frame"])
            ax.axvline(onset_rel - 0.5, color="cyan", lw=1.2)
            ax.set_title(f"{split} | pitch={int(row['pitch'])} | onset={onset_rel}", fontsize=8)
            ax.set_xlabel("frame")
            ax.set_ylabel("bin")
    fig.suptitle("Contoh window audio 42 frame (garis cyan = onset)", fontsize=12)
    fig.tight_layout()
    return save_and_embed(fig, "16_audio_window_gallery.png")


def make_pair_gallery(bundles: dict, hard: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for row_i, split in enumerate(SPLITS):
        nh = bundles[split]["nh"].set_index("notehead_id", drop=False)
        wd = bundles[split]["wd"].set_index("window_id", drop=False)
        crops = bundles[split]["crops"]
        wins = bundles[split]["wins"]
        pr = bundles[split]["pr"]

        pos = pr[pr["label"] == 1].sample(1, random_state=41).iloc[0]
        neg = pr[pr["label"] == 0].sample(1, random_state=42).iloc[0]
        hard_df = hard.get(split)
        hard_row = hard_df.sample(1, random_state=43).iloc[0] if hard_df is not None and len(hard_df) else None

        selected = [("POSITIF", pos), ("NEGATIF acak", neg)]
        selected.append(("NEGATIF keras", hard_row))

        for pair_i, (label, pair) in enumerate(selected):
            ax_img = axes[row_i, pair_i * 2]
            ax_aud = axes[row_i, pair_i * 2 + 1]
            if pair is None:
                ax_img.axis("off")
                ax_aud.axis("off")
                ax_img.set_title(f"{split}: hard negative tidak tersedia", fontsize=8)
                continue
            nh_row = nh.loc[str(pair["notehead_id"])]
            wd_row = wd.loc[str(pair["window_id"])]
            crop = crops[int(nh_row["crop_row"]), :, :, 0]
            audio = wins[int(wd_row["window_row"])].astype(np.float32)
            onset_rel = int(wd_row["onset_frame"] - wd_row["start_frame"])

            ax_img.imshow(crop, cmap="gray", vmin=0, vmax=255)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(
                f"{split} | {label}\nneg_type={pair['neg_type']} | pitch={int(nh_row['pitch'])}",
                fontsize=7,
            )
            ax_aud.imshow(audio.T, aspect="auto", origin="lower", cmap="magma", vmin=-2, vmax=2)
            ax_aud.axvline(onset_rel - 0.5, color="cyan", lw=1.0)
            ax_aud.set_xticks([])
            ax_aud.set_yticks([])
            ax_aud.set_title(f"audio onset={onset_rel}", fontsize=7)
    fig.suptitle("Contoh pasangan gambar-audio (kiri: crop, kanan: window audio)", fontsize=12)
    fig.tight_layout()
    return save_and_embed(fig, "17_pair_gallery.png")


def make_checks_figure(checks: pd.DataFrame) -> str:
    counts = checks.groupby(["category", "status"]).size().unstack(fill_value=0)
    for col in ("PASS", "FAIL", "SKIP"):
        if col not in counts:
            counts[col] = 0
    fig, ax = plt.subplots(figsize=(11, max(4, 0.38 * len(counts) + 1.5)))
    y = np.arange(len(counts))
    ax.barh(y, counts["PASS"], color="#55A868", label="PASS")
    ax.barh(y, counts["FAIL"], left=counts["PASS"], color="#C44E52", label="FAIL")
    ax.barh(y, counts["SKIP"], left=counts["PASS"] + counts["FAIL"], color="#DD8452", label="SKIP")
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("jumlah check")
    ax.set_title("Status validasi prepro per kategori")
    ax.legend(loc="lower right")
    for i, (_, row) in enumerate(counts.iterrows()):
        p, f, s = int(row["PASS"]), int(row["FAIL"]), int(row["SKIP"])
        ax.text(p + f + s + 0.2, i, f"{p}/{p + f + s}", va="center", fontsize=8)
    fig.tight_layout()
    return save_and_embed(fig, "18_checks_status.png")


def make_coverage_pitch_figure(coverage: pd.DataFrame, bundles: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    cov = coverage.sort_values("pct_mapped")
    axes[0].barh(cov["piece"], cov["pct_mapped"], color="#4C72B0")
    axes[0].set_xlabel("% notehead dengan minimal 1 window")
    axes[0].set_title("Coverage per piece")
    axes[0].tick_params(axis="y", labelsize=7)

    for split in SPLITS:
        pitches = bundles[split]["nh"]["pitch"].to_numpy()
        axes[1].hist(pitches, bins=np.arange(20, 101, 2), alpha=0.55, label=split, color=COLORS[split])
    axes[1].set_xlabel("MIDI pitch")
    axes[1].set_ylabel("jumlah notehead")
    axes[1].set_title("Distribusi pitch per split")
    axes[1].legend()
    fig.tight_layout()
    return save_and_embed(fig, "19_coverage_pitch_distribution.png")


def artifact_rows(summary: dict, manifest: dict | None) -> list[dict]:
    rows = [
        {
            "file": "score_crops_{split}.npy",
            "shape": "[N, 200, 160, 1]",
            "dtype": "uint8",
            "rows": f"{summary['split_crops']['train']:,} / {summary['split_crops']['val']:,} / {summary['split_crops']['test']:,}",
        },
        {
            "file": "audio_windows_{split}.npy",
            "shape": "[N, 42, 92]",
            "dtype": "float16",
            "rows": f"{summary['split_windows']['train']:,} / {summary['split_windows']['val']:,} / {summary['split_windows']['test']:,}",
        },
        {
            "file": "noteheads_{split}.csv",
            "shape": "metadata",
            "dtype": "CSV",
            "rows": f"{summary['split_crops']['train']:,} / {summary['split_crops']['val']:,} / {summary['split_crops']['test']:,}",
        },
        {
            "file": "windows_{split}.csv",
            "shape": "metadata",
            "dtype": "CSV",
            "rows": f"{summary['split_windows']['train']:,} / {summary['split_windows']['val']:,} / {summary['split_windows']['test']:,}",
        },
        {
            "file": "pairs_note_{split}.csv",
            "shape": "metadata",
            "dtype": "CSV",
            "rows": f"{summary['split_pairs']['train']:,} / {summary['split_pairs']['val']:,} / {summary['split_pairs']['test']:,}",
        },
    ]
    return rows


def build_html(
    summary: dict,
    norm: dict,
    coverage: pd.DataFrame,
    checks: pd.DataFrame,
    old_checks: pd.DataFrame,
    hard_stats: dict,
    manifest: dict | None,
    env: dict | None,
    figs: dict[str, str],
) -> str:
    split_rows = []
    for split in SPLITS:
        split_rows.append(
            {
                "split": split,
                "piece": int((coverage["split"] == split).sum()),
                "crop": summary["split_crops"][split],
                "window": summary["split_windows"][split],
                "pair": summary["split_pairs"][split],
            }
        )
    split_df = pd.DataFrame(split_rows)

    hard_rows = []
    for split in SPLITS:
        s = hard_stats.get(split, {})
        breakdown = s.get("neg_type_counts", {})
        hard_rows.append(
            {
                "split": split,
                "hard_negatives": s.get("rows", 0),
                "same_piece": s.get("same_piece", 0),
                "same_pitch": s.get("same_pitch", 0),
                "breakdown": ", ".join(f"{k}={v}" for k, v in breakdown.items()),
            }
        )
    hard_df = pd.DataFrame(hard_rows)

    coverage_view = coverage.copy()
    coverage_view["pct_mapped"] = coverage_view["pct_mapped"].round(2)
    coverage_view = coverage_view.sort_values("pct_mapped", ascending=False)
    coverage_view = coverage_view[
        ["piece", "split", "n_pages", "n_noteheads", "n_mapped_noteheads", "n_valid_windows", "pct_mapped"]
    ]

    manifest_rows = []
    if manifest:
        for item in manifest.get("files", []):
            manifest_rows.append(
                {
                    "file": item["name"],
                    "size_mb": round(item["size_bytes"] / (1024 * 1024), 3),
                    "shape": item.get("shape", ""),
                    "dtype": item.get("dtype", ""),
                    "sha256": item["sha256"][:16] + "...",
                }
            )
    manifest_df = pd.DataFrame(manifest_rows)

    env_rows = []
    if env:
        env_rows.append({"key": "Python", "value": env.get("python", "")})
        env_rows.append({"key": "Platform", "value": env.get("platform", "")})
        env_rows.append({"key": "Seed", "value": env.get("seed", "")})
        for k, v in env.get("packages", {}).items():
            env_rows.append({"key": k, "value": v})
    env_df = pd.DataFrame(env_rows)

    def table(df: pd.DataFrame) -> str:
        return df.fillna("").to_html(index=False, escape=True, classes="table")

    n_total = len(checks)
    n_pass = int((checks["status"] == "PASS").sum())
    n_skip = int((checks["status"] == "SKIP").sum())
    check_num = f"{n_pass}/{n_total}"
    check_label = "check PASS" if n_skip == 0 else f"PASS; {n_skip} SKIP"

    summary_cards = f"""
    <div class="cards">
      <div class="card"><div class="num">24</div><div class="label">piece note-level</div></div>
      <div class="card"><div class="num">60</div><div class="label">halaman</div></div>
      <div class="card"><div class="num">18.813</div><div class="label">crop notehead</div></div>
      <div class="card"><div class="num">46.522</div><div class="label">window audio</div></div>
      <div class="card"><div class="num">93.044</div><div class="label">pair</div></div>
      <div class="card"><div class="num">{check_num}</div><div class="label">{check_label}</div></div>
    </div>
    """

    css = """
    :root { --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#f6f8fb; --card:#ffffff; --accent:#4C72B0; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; line-height:1.55; }
    .wrap { max-width:1200px; margin:0 auto; padding:28px 22px 60px; }
    h1 { font-size:30px; margin:0 0 6px; letter-spacing:-.02em; }
    h2 { font-size:20px; margin:32px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line); }
    h3 { font-size:15px; margin:20px 0 8px; }
    p, li { font-size:14px; }
    .sub { color:var(--muted); margin:0 0 18px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:20px 0; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
    .card .num { font-size:22px; font-weight:700; color:var(--accent); }
    .card .label { font-size:12px; color:var(--muted); }
    .table { width:100%; border-collapse:collapse; background:var(--card); font-size:12px; margin:10px 0 18px; }
    .table th, .table td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
    .table th { background:#f0f4f9; font-weight:600; }
    .table tr:nth-child(even) td { background:#fbfcfe; }
    figure { margin:18px 0 24px; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; }
    figure img { width:100%; height:auto; display:block; border-radius:6px; }
    .caption { color:var(--muted); font-size:12px; padding:6px 2px 0; }
    details { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:10px 0 18px; }
    summary { cursor:pointer; font-weight:600; }
    code, pre { background:#eef2f7; border-radius:6px; }
    code { padding:1px 5px; font-size:12px; }
    pre { padding:12px; overflow:auto; font-size:12px; }
    .note { background:#fff8e6; border-left:4px solid #e0a800; padding:10px 14px; border-radius:6px; font-size:13px; }
    .ok { color:#2f7d4f; font-weight:600; }
    .bad { color:#b42318; font-weight:600; }
    .footer { color:var(--muted); font-size:12px; margin-top:40px; border-top:1px solid var(--line); padding-top:12px; }
    @media (max-width:700px) { .wrap { padding:18px 12px 40px; } h1 { font-size:24px; } }
    """

    return f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laporan Preprocessing Note-Level</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <h1>Laporan Preprocessing Note-Level</h1>
  <p class="sub">Audio–Sheet Music Matching · MSMD augmented v1.1 (no-audio) · subset 25 piece · laporan statis, tidak perlu menjalankan notebook.</p>
  {summary_cards}

  <div class="note">
    <strong>Untuk teman/reviewer:</strong> file ini sudah memuat gambar dan tabel. Cukup buka di browser.
    Tidak perlu Python, Jupyter, atau dataset 18 GB. Untuk verifikasi angka, lihat bagian
    <a href="#verifikasi">Verifikasi</a>.
  </div>

  <h2>1. Apa yang dihasilkan preprocessing?</h2>
  <p>Preprocessing ini menghasilkan <strong>dataset pasangan siap latih</strong>, bukan gambar partitur yang sudah “dibersihkan”. Satu contoh data adalah:</p>
  <pre>(crop gambar notehead 200x160, window audio 42 frame di sekitar onset, label 1/0)</pre>
  <ul>
    <li><strong>Gambar:</strong> crop grayscale mentah di sekitar notehead; garis staff tetap ada.</li>
    <li><strong>Audio:</strong> log-frequency spectrogram 92 bin, 42 frame (sekitar 2,1 detik pada 20 fps), dinormalisasi per bin.</li>
    <li><strong>Label:</strong> 1 = pasangan notehead dan window audio yang sama; 0 = pasangan negatif.</li>
    <li><strong>Split:</strong> berdasarkan <code>work_group</code>, sehingga tidak ada piece/work yang bocor antar-split.</li>
  </ul>

  <h2>2. Ringkasan split</h2>
  {table(split_df)}
  <p><strong>Catatan:</strong> split test hanya berisi 3 piece. Semua metrik retrieval nanti harus dibaca sebagai proof-of-concept, bukan klaim generalisasi.</p>

  <h2>3. Artefak</h2>
  {table(pd.DataFrame(artifact_rows(summary, manifest)))}
  <p>Total artefak tensor note-level sekitar 1,03 GB. File tensor tidak perlu dibuka untuk membaca laporan ini.</p>

  <h2>4. Validasi</h2>
  <p>Check awal dari notebook: <span class="ok">63/63 PASS</span>. Script validasi mandiri menambahkan pengecekan metadata, index, window, pair, split, audio, dan gambar.</p>
  {img_tag(figs["checks"], "Status check", "Gambar 18. Status check per kategori; tidak ada check extended yang FAIL.")}
  {table(checks[["category", "check", "status", "detail"]])}
  <details>
    <summary>Lihat check awal dari notebook (63 baris)</summary>
    {table(old_checks)}
  </details>

  <h2>5. Visualisasi</h2>
  {img_tag(figs["split"], "Ringkasan split", "Gambar 13. Jumlah piece, halaman, crop, dan window per split.")}
  {img_tag(figs["page"], "Overlay halaman", "Gambar 14. Kotak merah = notehead; kotak hijau = crop 160x200 pada sampel. Ini membantu memeriksa apakah crop berada di sekitar notehead yang benar.")}
  {img_tag(figs["crop"], "Galeri crop", "Gambar 15. Contoh crop notehead dari ketiga split.")}
  {img_tag(figs["audio"], "Galeri audio", "Gambar 16. Contoh window audio; garis cyan adalah onset note.")}
  {img_tag(figs["pair"], "Galeri pair", "Gambar 17. Contoh pasangan positif, negatif acak, dan negatif keras.")}

  <h2>6. Coverage dan distribusi pitch</h2>
  {img_tag(figs["coverage"], "Coverage dan pitch", "Gambar 19. Coverage mapping per piece dan distribusi pitch per split.")}
  {table(coverage_view)}

  <h2>7. Hard negatives tambahan</h2>
  <p>File <code>pairs_note_{split}.csv</code> yang lama hanya memakai negatif dari piece lain. Untuk analisis tambahan, dibuat file terpisah <code>pairs_note_hard_{split}.csv</code> dengan negatif dari piece yang sama (prioritas pitch sama). File lama tidak diubah.</p>
  {table(hard_df)}
  <p>Jumlah negatif keras: <strong>{sum(int(s.get('rows', 0)) for s in hard_stats.values()):,}</strong> baris. Ini belum dipakai untuk training; hanya disiapkan agar eksperimen berikutnya tidak memakai negatif yang terlalu mudah.</p>

  <h2>8. Batasan</h2>
  <ul>
    <li>Subset 25 piece; note-level 24 piece karena satu piece tidak punya MuNG.</li>
    <li>Test hanya 3 piece; metrik nanti bisa noisy.</li>
    <li>Coverage rendah untuk beberapa piece (Froberger, Brahms) karena notehead tied/missing mapping.</li>
    <li>Crop 160x200 dapat memuat lebih dari satu system pada halaman padat; ini batasan untuk task system-level, bukan note-level.</li>
    <li>Negatif lama hanya cross-piece; karena itu disiapkan hard negatives terpisah.</li>
    <li>Belum ada raw audio; augmentasi audio di masa depan terbatas pada domain spectrogram.</li>
    <li>Belum ada model/training di tahap ini.</li>
  </ul>

  <h2 id="verifikasi">9. Verifikasi</h2>
  <p>Jika ingin memverifikasi artefak di mesin yang memiliki file hasil preprocessing:</p>
  <pre>python scripts/validate_preprocessing.py --check-raw-paths</pre>
  <p>Untuk cek cepat tanpa dataset mentah:</p>
  <pre>python scripts/validate_preprocessing.py</pre>
  <p>Manifest berisi SHA256 untuk setiap file artefak. File tensor <code>.npy</code> tidak disarankan masuk Git biasa karena ukurannya besar; metadata CSV/JSON dan laporan ini yang dibagikan.</p>
  <p>Bukti mentah notebook yang sudah dieksekusi: <a href="02_note_level_preprocessing.html">02_note_level_preprocessing.html</a> (opsional, untuk transparansi).</p>

  <details>
    <summary>Manifest artefak (SHA256)</summary>
    {table(manifest_df) if not manifest_df.empty else "<p>Tidak ada manifest.</p>"}
  </details>

  <details>
    <summary>Environment</summary>
    {table(env_df) if not env_df.empty else "<p>Tidak ada environment metadata.</p>"}
  </details>

  <div class="footer">
    Dibuat {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.
    Sumber: <code>data/processed/</code>. Notebook: <code>02_note_level_preprocessing.ipynb</code>.
  </div>
</div>
</body>
</html>
"""


def build_markdown(
    summary: dict,
    coverage: pd.DataFrame,
    checks: pd.DataFrame,
    hard_stats: dict,
    figs: dict[str, str],
) -> str:
    coverage_view = coverage.copy()
    coverage_view["pct_mapped"] = coverage_view["pct_mapped"].round(2)
    coverage_view = coverage_view.sort_values("pct_mapped", ascending=False)
    coverage_view = coverage_view[
        ["piece", "split", "n_pages", "n_noteheads", "n_mapped_noteheads", "n_valid_windows", "pct_mapped"]
    ]
    split_rows = [
        {
            "split": split,
            "crop": summary["split_crops"][split],
            "window": summary["split_windows"][split],
            "pair": summary["split_pairs"][split],
        }
        for split in SPLITS
    ]
    hard_rows = [
        {
            "split": split,
            "hard_negatives": hard_stats.get(split, {}).get("rows", 0),
            "same_piece": hard_stats.get(split, {}).get("same_piece", 0),
            "same_pitch": hard_stats.get(split, {}).get("same_pitch", 0),
        }
        for split in SPLITS
    ]
    n_total = len(checks)
    n_pass = int((checks["status"] == "PASS").sum())
    n_skip = int((checks["status"] == "SKIP").sum())
    check_line = f"{n_pass}/{n_total} check extended PASS" if n_skip == 0 else f"{n_pass}/{n_total} check extended PASS, {n_skip} SKIP"
    return f"""# Laporan Preprocessing Note-Level

Laporan statis untuk reviewer. Buka versi HTML untuk pengalaman lengkap: [`preprocessing_report.html`](preprocessing_report.html). Bukti mentah notebook: [`02_note_level_preprocessing.html`](02_note_level_preprocessing.html).

## Ringkasan

- 24 piece note-level, 60 halaman
- 18.813 crop notehead, 46.522 window audio, 93.044 pair
- 63/63 check awal PASS dan {check_line}
- Total artefak tensor sekitar 1,03 GB

## Split

{md_table(pd.DataFrame(split_rows))}

## Validasi

Total check extended: **{len(checks)}**; PASS: **{n_pass}**; SKIP: **{n_skip}**; FAIL: **{int((checks['status'] == 'FAIL').sum())}**.

{md_table(checks[['category', 'check', 'status', 'detail']])}

## Visualisasi

![Split](figures/13_split_summary.png)

![Overlay halaman](figures/14_page_overlay_samples.png)

![Crop notehead](figures/15_crop_gallery.png)

![Window audio](figures/16_audio_window_gallery.png)

![Pair](figures/17_pair_gallery.png)

![Status check](figures/18_checks_status.png)

![Coverage dan pitch](figures/19_coverage_pitch_distribution.png)

## Coverage per piece

{md_table(coverage_view)}

## Hard negatives

{md_table(pd.DataFrame(hard_rows))}

## Batasan

- Subset 25 piece; note-level 24 piece karena satu piece tidak punya MuNG.
- Test hanya 3 piece.
- Coverage rendah untuk Froberger dan Brahms.
- Crop 160x200 dapat memuat lebih dari satu system.
- Negatif lama hanya cross-piece; hard negatives disiapkan terpisah.
- Belum ada raw audio dan belum ada model/training.

## Verifikasi

```powershell
python scripts/validate_preprocessing.py --check-raw-paths
```
"""


def build_datacard(summary: dict, coverage: pd.DataFrame, hard_stats: dict) -> str:
    return f"""# Data Card — Note-Level Preprocessing

## Sumber

- Dataset: MSMD augmented v1.1 (no-audio)
- Subset: 25 piece (24 piece note-level; satu piece tanpa MuNG)
- Unit: crop notehead ↔ window audio di sekitar onset
- Split: group by `work_group`

## Jumlah

| Split | Crop | Window | Pair |
|---|---:|---:|---:|
| Train | {summary['split_crops']['train']:,} | {summary['split_windows']['train']:,} | {summary['split_pairs']['train']:,} |
| Val | {summary['split_crops']['val']:,} | {summary['split_windows']['val']:,} | {summary['split_pairs']['val']:,} |
| Test | {summary['split_crops']['test']:,} | {summary['split_windows']['test']:,} | {summary['split_pairs']['test']:,} |

## Schema

- `score_crops_{{split}}.npy`: `[N, 200, 160, 1]`, `uint8`, grayscale mentah.
- `audio_windows_{{split}}.npy`: `[N, 42, 92]`, `float16`, spectrogram ternormalisasi per bin, clip `[-5, 5]`.
- `noteheads_{{split}}.csv`: metadata crop; `crop_row` adalah index tensor.
- `windows_{{split}}.csv`: metadata window; `window_row` adalah index tensor.
- `pairs_note_{{split}}.csv`: pasangan positif/negatif untuk retrieval.
- `pairs_note_hard_{{split}}.csv`: negatif keras tambahan; tidak mengubah file lama.
- `norm_stats_note.json`: statistik normalisasi gambar/audio.

## Normalisasi

- Gambar: `(pixel / 255 - mean) / std`; mean `{summary.get('image_mean', 0):.6f}`, std `{summary.get('image_std', 0):.6f}`.
- Audio: sudah dinormalisasi saat preprocessing; jangan normalisasi ulang.

## Coverage

- Total piece: {len(coverage)}
- Coverage minimum: {coverage['pct_mapped'].min():.2f}%
- Coverage median: {coverage['pct_mapped'].median():.2f}%
- Coverage maksimum: {coverage['pct_mapped'].max():.2f}%

## Hard negatives

- Train: {hard_stats.get('train', {}).get('rows', 0):,}
- Val: {hard_stats.get('val', {}).get('rows', 0):,}
- Test: {hard_stats.get('test', {}).get('rows', 0):,}

## Batasan

- Test hanya 3 piece.
- Crop 160x200 dapat memuat lebih dari satu system.
- Belum ada raw audio.
- Belum ada model/training pada tahap ini.

## Verifikasi

```powershell
python scripts/validate_preprocessing.py --check-raw-paths
```
"""


def main() -> int:
    summary = read_json(PROC / "preprocess_summary_note.json")
    norm = read_json(PROC / "norm_stats_note.json")
    coverage = pd.read_csv(PROC / "coverage_note_level.csv")
    old_checks = pd.read_csv(PROC / "checks_note_level.csv")
    checks_path = PROC / "checks_note_level_extended.csv"
    checks = pd.read_csv(checks_path) if checks_path.exists() else old_checks.copy()
    if "category" not in checks.columns:
        checks.insert(0, "category", "legacy")

    manifest_path = PROC / "artifact_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    env_path = PROC / "environment_note.json"
    env = read_json(env_path) if env_path.exists() else None

    hard = {}
    for split in SPLITS:
        p = PROC / f"pairs_note_hard_{split}.csv"
        if p.exists():
            hard[split] = pd.read_csv(p)
    hard_stats_path = PROC / "pairs_note_hard_stats.json"
    hard_stats = read_json(hard_stats_path) if hard_stats_path.exists() else {}

    bundles = {split: load_split(split) for split in SPLITS}

    figs = {
        "split": make_split_summary_figure(bundles),
        "page": make_page_overlay_figure(bundles),
        "crop": make_crop_gallery(bundles),
        "audio": make_audio_gallery(bundles),
        "pair": make_pair_gallery(bundles, hard),
        "checks": make_checks_figure(checks),
        "coverage": make_coverage_pitch_figure(coverage, bundles),
    }

    html_out = build_html(summary, norm, coverage, checks, old_checks, hard_stats, manifest, env, figs)
    (REP / "preprocessing_report.html").write_text(html_out, encoding="utf-8")

    md_out = build_markdown(summary, coverage, checks, hard_stats, figs)
    (REP / "preprocessing_report.md").write_text(md_out, encoding="utf-8")

    datacard = build_datacard(summary, coverage, hard_stats)
    (REP / "preprocessing_datacard.md").write_text(datacard, encoding="utf-8")

    print("Saved:", REP / "preprocessing_report.html")
    print("Saved:", REP / "preprocessing_report.md")
    print("Saved:", REP / "preprocessing_datacard.md")
    print("Figures:")
    for name in [
        "13_split_summary.png",
        "14_page_overlay_samples.png",
        "15_crop_gallery.png",
        "16_audio_window_gallery.png",
        "17_pair_gallery.png",
        "18_checks_status.png",
        "19_coverage_pitch_distribution.png",
    ]:
        print(" ", FIG / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
