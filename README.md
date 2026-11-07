# Audio–Sheet Music Matching

Proyek machine learning untuk mencocokkan **gambar partitur** dengan **audio piano**. Tujuannya menghasilkan skor kemiripan antara potongan not dan potongan audio, lalu mengevaluasinya dengan protokol retrieval.

Dataset yang dipakai: **MSMD augmented v1.1 (no-audio)** di `msmd_aug_v1-1_no-audio/`.

> **Lihat hasil preprocessing tanpa menjalankan notebook:**
> buka [`reports/preprocessing_report.html`](reports/preprocessing_report.html).
> Laporan ini sudah memuat tabel, validasi, dan contoh visual; tidak butuh Python,
> Jupyter, atau dataset 18 GB.
>
> Ringkasan untuk reviewer juga tersedia di
> [`reports/preprocessing_report.md`](reports/preprocessing_report.md) dan
> [`reports/preprocessing_datacard.md`](reports/preprocessing_datacard.md).
> Bukti mentah notebook yang sudah dieksekusi:
> [`reports/02_note_level_preprocessing.html`](reports/02_note_level_preprocessing.html).

## Dua jalur preprocessing

| Jalur | Unit matching | Notebook | Status |
|---|---|---|---|
| Page-level baseline | 1 halaman ↔ segmen audio 5 detik dari piece yang sama | `notebooks/01_eda_preprocessing.ipynb` | Sudah ada |
| Note-level | 1 crop notehead ↔ window audio di sekitar onset note | `notebooks/02_note_level_preprocessing.ipynb` | Sudah ada |

Jalur note-level adalah jalur utama untuk pemodelan. Baseline page-level tetap disimpan sebagai pembanding.

## Ringkasan note-level

- Crop gambar: **160×200 px** di sekitar notehead, tetap memuat garis staff.
- Window audio: **42 frame** (≈2,1 detik pada 20 fps) di sekitar onset.
- Train: maksimal **2 performance per notehead**.
- Val/test: semua performance valid.
- Split: `group_by_work_group`, tidak ada `work_group` yang muncul di dua split.
- Crop disimpan **raw grayscale uint8**; normalisasi gambar dilakukan on-the-fly.
- Audio disimpan **float16 ternormalisasi per bin**, clip `[-5, 5]`.

### Angka aktual

| Split | Piece | Halaman | Crop notehead | Window audio | Pair (pos + neg) |
|---|---:|---:|---:|---:|---:|
| Train | 17 | 44 | 14.365 | 28.730 | 57.460 |
| Val | 4 | 8 | 2.430 | 9.720 | 19.440 |
| Test | 3 | 8 | 2.018 | 8.072 | 16.144 |
| **Total** | **24** | **60** | **18.813** | **46.522** | **93.044** |

- `image_mean = 0.8936307`, `image_std = 0.2651336` (skala `[0,1]`, crop train).
- Checks: **63/63 PASS** (check awal notebook) dan **195/195 PASS** (validasi extended).
- Total artefak note-level: ±1,0 GB.

Catatan: `GobbaertsL__Les_Etoiles_dOr_No1__Les_Etoiles_dOr_No1` tidak punya MuNG XML, sehingga dikecualikan dari jalur note-level. Piece ini tetap ada di baseline page-level.

## Validasi dan laporan statis

Preprocessing note-level divalidasi ulang oleh script mandiri:

```powershell
python scripts\validate_preprocessing.py --check-raw-paths
```

Script ini menghasilkan:

- `data/processed/checks_note_level_extended.csv` — **195 check**, 0 FAIL.
- `data/processed/artifact_manifest.json` — shape, dtype, ukuran, dan SHA256 setiap artefak.
- `data/processed/environment_note.json` — versi Python/package, seed, platform.

Jika dataset mentah tidak tersedia, jalankan tanpa `--check-raw-paths`; check keberadaan PNG/MuNG akan ditandai **SKIP** (total tetap 195: 189 PASS, 6 SKIP, 0 FAIL).

Laporan untuk reviewer:

```powershell
python scripts\build_preprocessing_report.py
```

Output:

- `reports/preprocessing_report.html` — **self-contained**, gambar di-embed base64.
- `reports/preprocessing_report.md` — versi GitHub.
- `reports/preprocessing_datacard.md` — data card ringkas.
- `reports/figures/13_*.png` sampai `19_*.png` — split, overlay halaman, crop, audio, pair, checks, coverage.

## Hard negatives tambahan

File `pairs_note_{split}.csv` yang lama hanya memakai negatif dari piece lain. Untuk eksperimen berikutnya, dibuat file terpisah:

- `data/processed/pairs_note_hard_{split}.csv`
- `data/processed/pairs_note_hard_stats.json`

Hard negative dipilih dengan prioritas: same piece + same pitch, same piece random, cross piece same pitch, cross piece random. File lama tidak diubah.

## Struktur direktori

```text
PROJECT/
├── README.md
├── requirements.txt
├── .gitignore
├── Note.docx
├── Note (1).docx
├── msmd_aug_v1-1_no-audio/              # tidak di-commit (18 GB)
├── data/
│   ├── subset_pieces.csv
│   ├── subset_performances.csv
│   ├── splits.csv
│   └── processed/
│       ├── score_pages_*.npy             # baseline page-level
│       ├── audio_segments_*.npy          # baseline page-level
│       ├── pairs_*.csv                   # baseline page-level
│       ├── score_crops_*.npy             # note-level, tidak di-commit (besar)
│       ├── audio_windows_*.npy           # note-level, tidak di-commit (besar)
│       ├── noteheads_*.csv               # metadata note-level
│       ├── windows_*.csv                 # metadata note-level
│       ├── pairs_note_*.csv              # pair retrieval
│       ├── pairs_note_hard_*.csv         # hard negative tambahan
│       ├── pairs_note_hard_stats.json
│       ├── norm_stats_note.json
│       ├── coverage_note_level.csv
│       ├── checks_note_level.csv
│       ├── checks_note_level_extended.csv
│       ├── artifact_manifest.json
│       ├── environment_note.json
│       └── preprocess_summary_note.json
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_note_level_preprocessing.ipynb
├── scripts/
│   ├── validate_preprocessing.py
│   ├── add_hard_negatives.py
│   ├── build_preprocessing_report.py
│   └── export_notebook_html.py
└── reports/
    ├── methodology_dataset.md
    ├── preprocessing_report.html          # untuk reviewer, self-contained
    ├── preprocessing_report.md
    ├── preprocessing_datacard.md
    ├── 02_note_level_preprocessing.html   # output notebook mentah
    └── figures/
        ├── 01_*.png ... 08_*.png
        ├── 09_notehead_crops.png
        ├── 10_audio_windows.png
        ├── 11_pair_demo.png
        ├── 12_coverage.png
        └── 13_*.png ... 19_*.png
```

## Skema artefak note-level

### `score_crops_{split}.npy`

- Shape: `[N, 200, 160, 1]`
- Dtype: `uint8`
- Isi: crop grayscale mentah; background putih `255`.
- Baris ke-`i` sesuai dengan `crop_row == i` di `noteheads_{split}.csv`.

### `audio_windows_{split}.npy`

- Shape: `[N, 42, 92]`
- Dtype: `float16`
- Isi: log-frequency spectrogram yang sudah dinormalisasi per bin dan di-clip.
- Baris ke-`i` sesuai dengan `window_row == i` di `windows_{split}.csv`.

### `noteheads_{split}.csv`

Kolom penting:

`notehead_id, page_id, piece, composer, work_group, split, page_num, mung_id, cx, cy, box_top, box_left, box_w, box_h, pitch, tied, edge, pad_frac, ink_ratio, png_path, mung_path, crop_row`

### `windows_{split}.csv`

Kolom penting:

`window_id, notehead_id, page_id, piece, composer, work_group, split, perf_name, note_event_idx, onset_s, onset_frame, start_frame, end_frame, pad_left, pad_right, n_frames, pitch, window_row`

### `pairs_note_{split}.csv`

`pair_id, notehead_id, window_id, label, piece, work_group, split, neg_type, window_piece, window_work_group`

- `label = 1`: pasangan positif; `window_piece == piece`.
- `label = 0`: pasangan negatif; `window_piece != piece`.
- `neg_type`: `positive` atau `random` (negatif dari piece lain).
- Hard negative tambahan ada di `pairs_note_hard_{split}.csv` dengan tipe `same_piece_same_pitch`, `same_piece_random`, `cross_piece_same_pitch`, atau `cross_piece_random`. File ini terpisah dan tidak mengubah `pairs_note_*`.

### `norm_stats_note.json`

Berisi statistik gambar crop train dan statistik audio yang dipakai untuk normalisasi. Gunakan file ini saat training dan inference agar konsisten.

## Cara menjalankan

Install dependency:

```powershell
python -m pip install -r requirements.txt
```

Regenerate artefak note-level (butuh dataset mentah):

```powershell
python -m nbconvert --execute --inplace notebooks\02_note_level_preprocessing.ipynb
```

Validasi artefak yang sudah ada:

```powershell
python scripts\validate_preprocessing.py --check-raw-paths
```

Buat hard negatives tambahan:

```powershell
python scripts\add_hard_negatives.py
```

Buat laporan statis untuk reviewer:

```powershell
python scripts\build_preprocessing_report.py
```

Export notebook `02` sebagai HTML mentah (opsional, path lokal disanitasi):

```powershell
python scripts\export_notebook_html.py
```

Notebook `01` tidak perlu dijalankan ulang kecuali ingin meregenerasi baseline page-level. Notebook `03` untuk training belum dibuat.

## Cara memakai di model

```python
import numpy as np
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")
split = "train"

crops = np.load(PROC / f"score_crops_{split}.npy", mmap_mode="r")
wins = np.load(PROC / f"audio_windows_{split}.npy", mmap_mode="r")
nh = pd.read_csv(PROC / f"noteheads_{split}.csv")
wd = pd.read_csv(PROC / f"windows_{split}.csv")
pairs = pd.read_csv(PROC / f"pairs_note_{split}.csv")

crop_by_id = nh.set_index("notehead_id")["crop_row"].to_dict()
win_by_id = wd.set_index("window_id")["window_row"].to_dict()

row = pairs.iloc[0]
img = crops[crop_by_id[row["notehead_id"]]]          # [200,160,1] uint8
aud = wins[win_by_id[row["window_id"]]]              # [42,92] float16
label = int(row["label"])
```

Normalisasi gambar:

```python
img_f = (img[..., 0].astype(np.float32) / 255.0 - 0.8936307) / 0.2651336
```

Audio sudah ternormalisasi; tidak perlu normalisasi ulang.

## Model yang direncanakan

- **Image encoder:** ResNet18 1-channel atau CNN kecil → vektor 128/256.
- **Audio encoder:** CNN 2D pada `[42, 92]` → vektor 128/256.
- L2-normalize kedua vektor; cosine similarity.
- Loss: InfoNCE/contrastive atau triplet.
- Untuk eksperimen yang lebih sulit, gabungkan `pairs_note_*` dengan `pairs_note_hard_*`; jangan menimpa file pair lama.
- Batch 64; Adam `1e-4`; early stopping pada validation MRR/R@1.
- Augmentasi:
  - Gambar: crop jitter, scaling, rotasi kecil, kontras, blur. **Jangan flip.**
  - Audio: SpecAugment, time shift, gain kecil. **Jangan pitch-shift/time-stretch.**
- Output persentase: kalibrasi cosine pada validation; sebut sebagai “similarity score”, bukan probabilitas.

## Evaluasi

- **Snippet retrieval:** window audio → ranking semua crop notehead test. Metrik R@1, R@5, R@10, MRR, median rank.
- **Arah sebaliknya:** crop → ranking window audio.
- **Page/piece identification:** voting top-k hasil snippet.
- **Toleransi chord/repeated note:** terima jika notehead berada di onset/page yang sama; laporkan exact dan tolerant.
- **Pembanding:** multimodal vs audio-only vs image-only, plus random baseline.

## Batasan

- Subset hanya 25 piece; hasil retrieval bisa noisy.
- Satu piece (`GobbaertsL`) tidak punya MuNG dan dikecualikan dari note-level.
- Banyak notehead tied/missing mapping, terutama Froberger dan Brahms. Coverage minimum ±45%.
- Crop 160×200 dapat memuat lebih dari satu system pada halaman yang padat. Jika model kesulitan, pertimbangkan staff-aware crop.
- Chord dan repeated note membuat evaluasi exact-match ambigu.
- Belum ada raw audio; augmentasi audio terbatas pada domain spektrogram.
- `torch`/`torchvision` belum terpasang; perlu instalasi sebelum training.
- Notebook training/model belum dibuat pada tahap ini.

## GitHub dan file besar

Repo ini siap di-push ke GitHub dengan catatan:

- `msmd_aug_v1-1_no-audio/` dan `msmd_aug_v1-1_no-audio.zip` **tidak di-commit**.
- File tensor `data/processed/*.npy` **tidak di-commit** karena totalnya ±1 GB.
- Yang di-commit: kode, notebook, metadata CSV/JSON, laporan HTML/Markdown, figures, manifest, dan checks.
- Reviewer cukup membuka `reports/preprocessing_report.html`; tidak perlu dataset atau artefak tensor.
- Jika ingin membagikan artefak tensor, gunakan Hugging Face/Zenodo/GitHub Release, bukan Git biasa.

Contoh inisialisasi repo (jalankan setelah yakin):

```powershell
git init
git add .
git status --short
git commit -m "Add note-level preprocessing, validation, and static report"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

## Referensi

- Dorfer, M., Hajič, J. jr., Arzt, A., Frostel, H., & Widmer, G. (2018). *Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification.* TISMIR. https://doi.org/10.5334/tismir.12
- Dorfer, M., Arzt, A., & Widmer, G. (2017). *Learning Audio–Sheet Music Correspondences for Score Identification and Offline Alignment.* ISMIR.
- Carvalho, L., & Widmer, G. (2023). *Passage Summarization with Recurrent Models for Audio-Sheet Music Retrieval.* ISMIR. https://arxiv.org/abs/2309.12111
- Repo: https://github.com/CPJKU/audio_sheet_retrieval
- Dataset: https://github.com/CPJKU/msmd

## Next step

Tahap preprocessing sudah dilengkapi validasi dan laporan statis. Setelah ini:

1. Review `reports/preprocessing_report.html`.
2. Putuskan apakah akan memakai `pairs_note_*` (negatif cross-piece) atau menambahkan `pairs_note_hard_*` untuk eksperimen retrieval yang lebih sulit.
3. Baru setelah itu buat `notebooks/03_baseline_model.ipynb` untuk training dual encoder, evaluasi R@1/R@5/R@10, dan contoh top-5 retrieval.
