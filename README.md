# Audio-Sheet Music Matching (MSMD)

Preprocessing untuk mencocokkan **gambar partitur** dengan **audio piano** pada dataset
MSMD augmented v1.1 (no-audio). Repo ini sekarang fokus pada dua task:

| Task | Input | Output |
|---|---|---|
| **Task 1** | window audio + halaman partitur | lokasi **system** partitur yang cocok |
| **Task 2** | window audio | klasifikasi 4 kelas: **clean, wrong_pitch, delete, insert** |

Status: pipeline preprocessing **selesai dan tervalidasi**. Training, arsitektur model,
loss, dan evaluasi model **tidak termasuk** dalam scope repo ini.

---

## Quick start: memakai artefak tanpa dataset mentah

Artefak besar didistribusikan lewat **GitHub Release**, bukan lewat Git.

1. Unduh empat asset Release:
   - `preproc-v1-core.zip`
   - `preproc-v1-task1.zip`
   - `preproc-v1-task2.zip`
   - `preproc-v1-meta.zip`
2. Ekstrak semuanya ke **root repo**, pertahankan struktur folder. Zip sudah berisi
   path seperti `data/processed/...`, `configs/...`, dan `README.md`.
3. Install dependency:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Validasi artefak **tanpa** membutuhkan dataset 18 GB:

   ```powershell
   python scripts\run_preprocessing.py --verify-only
   ```

   Output akhir harus berisi `"ok": true` dan `Validation passed`.
   Laporan lengkap ditulis ke `reports/validation_preprocessing.json`.

Jika hanya ingin memakai artefak di kode training, mulai dari:

```python
import numpy as np
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")
split = "train"

# Task 1
windows = np.load(PROC / f"audio_windows_{split}.npy", mmap_mode="r")
systems = np.load(PROC / f"system_crops_{split}.npy", mmap_mode="r")
pairs_t1 = pd.read_parquet(PROC / f"pairs_system_{split}.parquet")

# Task 2
error_windows = np.load(PROC / f"error_windows_{split}.npy", mmap_mode="r")
task2 = pd.read_parquet(PROC / f"task2_windows_{split}.parquet")
```

---

## Hasil aktual

### Core note-level

| Split | Piece | Crop notehead | Window audio | Pair retrieval |
|---|---:|---:|---:|---:|
| Train | 17 | 14.365 | 28.730 | 57.460 |
| Val | 4 | 2.430 | 9.720 | 19.440 |
| Test | 3 | 2.018 | 8.072 | 16.144 |
| **Total** | **24** | **18.813** | **46.522** | **93.044** |

- Crop gambar: `uint8` raw grayscale, shape `[N, 200, 160, 1]`.
- Window audio: `float16` ternormalisasi, shape `[N, 42, 92]`.
- `image_mean = 0.8936307`, `image_std = 0.2651336` (skala `[0,1]`, crop train).
- Alignment audio: `T = min(spec_T, midi_T)`; 72/100 performance memiliki
  `spec_T = midi_T + 1`.
- `GobbaertsL__Les_Etoiles_dOr_No1__Les_Etoiles_dOr_No1` dikecualikan karena tidak
  memiliki MuNG XML.

### Task 1: system localization

| Split | Halaman | System | Query | Pair (1 pos + 4 neg) |
|---|---:|---:|---:|---:|
| Train | 44 | 240 | 28.730 | 143.650 |
| Val | 8 | 38 | 9.720 | 48.600 |
| Test | 8 | 36 | 8.072 | 40.360 |
| **Total** | **60** | **314** | **46.522** | **232.610** |

- Assign notehead ke system: **16.491** inside polygon, **2.322** fallback nearest
  y-center, **0** unassigned.
- `coords/bars_*.npy` kosong, sehingga Task 1 hanya sampai level **system**, bukan bar.
- Halaman: `uint8 [N, 256, 256, 1]`.
- System crop: `uint8 [N, 128, 512, 1]`.
- Statistik train: page `mean=0.9429144`, `std=0.1269404`; system
  `mean=0.9099208`, `std=0.2053268`.

### Task 2: 4-class synthetic errors

| Split | Source window | Clean | Wrong pitch | Delete | Insert | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 28.730 | 28.730 | 28.730 | 28.730 | 28.730 | 114.920 |
| Val | 9.720 | 9.720 | 9.720 | 9.720 | 9.720 | 38.880 |
| Test | 8.072 | 8.072 | 8.072 | 8.072 | 8.072 | 32.288 |
| **Total** | **46.522** | **46.522** | **46.522** | **46.522** | **46.522** | **186.088** |

- Shape: `float16 [N, 42, 92]`; `error_row` kontinu.
- Template per `(soundfont, pitch)` dari **train saja**; tidak ada leakage split.
- Template coverage: 52/47/29/43 pitch direct untuk ElectricPiano, YamahaGrandPiano,
  acoustic_piano_imis_1, dan grand-piano-YDP-20160804.
- Pitch direct global: 31-91. Pitch yang kurang memakai **ridge fallback**, lalu
  **nearest**, lalu **global**. Hasil validasi: 42 varian memakai `fallback_overlap`,
  0 skip.
- Error dibentuk dengan mengubah window asli: wrong pitch menekan note sumber lalu
  menambahkan template pitch target; delete menekan region note; insert menambahkan
  attack note tambahan.

---

## Cara regenerate dari dataset mentah

Butuh folder `msmd_aug_v1-1_no-audio/` (tidak di-commit) dan long path Windows aktif.

```powershell
# Semua stage: core -> task1 -> task2
python scripts\run_preprocessing.py --stage all

# Per stage
python scripts\run_preprocessing.py --stage core
python scripts\run_preprocessing.py --stage task1
python scripts\run_preprocessing.py --stage task2

# Validasi artefak yang sudah ada
python scripts\run_preprocessing.py --verify-only
```

Log ditulis ke `logs/preprocess_YYYYMMDD_HHMMSS.log`. Konfigurasi utama ada di
`configs/preprocess.yaml`.

Windows long path:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
git config core.longpaths true
```

---

## Pipeline

### Core

1. Baca `subset_pieces.csv`, `subset_performances.csv`, dan `splits.csv`.
2. Pilih performance train maksimal 2 per notehead, prioritaskan soundfont berbeda.
3. Validasi notehead MuNG terhadap `notes.npy`; skip notehead tanpa note event.
4. Crop notehead `160x200` dan window audio `42x92` di sekitar onset.
5. Buat `pairs_note_*`: satu positif dan satu negatif dari piece lain.

### Task 1

1. Load halaman dan polygon system dari `systems/*.npy`.
2. Assign notehead ke system dengan point-in-polygon; fallback nearest y-center.
3. Buat page tensor, system crop, dan `pairs_system_*` dengan 4 negatif:
   same page, same piece other page, cross piece, dan filler.
4. Simpan metadata halaman, system, dan assignment.

### Task 2

1. Bangun template pitch dari window train yang tidak terpotong padding.
2. Direct template jika minimal 20 frame; fallback ridge/nearest/global jika kurang.
3. Untuk setiap window bersih, buat 4 varian: clean, wrong_pitch, delete, insert.
4. Simpan `error_windows_*` dan metadata kelas.

---

## Skema artefak

### Core

| File | Isi |
|---|---|
| `score_crops_{split}.npy` | `uint8 [N,200,160,1]`, `crop_row == i` |
| `audio_windows_{split}.npy` | `float16 [N,42,92]`, `window_row == i` |
| `noteheads_{split}.parquet` | metadata crop + `crop_row` |
| `windows_{split}.parquet` | metadata window + `window_row` |
| `pairs_note_{split}.parquet` | pair retrieval note-level |
| `coverage_{split}.parquet` | coverage per piece |
| `norm_stats_note.json` | statistik gambar/audio |
| `preprocess_summary_note.json` | ringkasan count |

Kolom penting `windows_*`: `window_id, notehead_id, page_id, piece, composer,
work_group, split, perf_name, soundfont, tempo, note_event_idx, onset_s, onset_frame,
start_frame, end_frame, pad_left, pad_right, n_frames, spec_T, midi_T, aligned_T,
pitch, window_row`.

### Task 1

| File | Isi |
|---|---|
| `score_pages_{split}.npy` | `uint8 [N,256,256,1]`, `page_row == i` |
| `system_crops_{split}.npy` | `uint8 [N,128,512,1]`, `crop_row == i` |
| `pages_{split}.parquet` | metadata halaman |
| `systems_{split}.parquet` | metadata system + bbox + `crop_row` |
| `notehead_systems_{split}.parquet` | assignment notehead ke system |
| `pairs_system_{split}.parquet` | query window ke kandidat system |
| `norm_stats_task1.json` | statistik page/system |
| `preprocess_summary_task1.json` | ringkasan count |

### Task 2

| File | Isi |
|---|---|
| `error_windows_{split}.npy` | `float16 [N,42,92]`, `error_row == i` |
| `task2_windows_{split}.parquet` | metadata varian + label |
| `templates_task2.npz` | template bank per soundfont |
| `templates_task2.json` | coverage template |
| `task2_class_map.json` | mapping class id |
| `preprocess_summary_task2.json` | ringkasan count |

Label Task 2: `0=clean`, `1=wrong_pitch`, `2=delete`, `3=insert`.

---

## Validasi dan release

```powershell
# Validasi artefak
python scripts\run_preprocessing.py --verify-only

# Manifest + SHA256 semua artefak
python scripts\build_manifest.py

# Zip untuk GitHub Release
python scripts\build_release.py
```

Output:

- `reports/validation_preprocessing.json`
- `data/processed/manifest_preprocessing.json`
- `data/processed/SHA256SUMS.txt`
- `dist/preproc-v1-core.zip`
- `dist/preproc-v1-task1.zip`
- `dist/preproc-v1-task2.zip`
- `dist/preproc-v1-meta.zip`
- `dist/SHA256SUMS.txt`

Ukuran artefak total sekitar **2,4 GB**; file `.npy` tidak di-commit ke Git.

---

## Dataset dan exclusion

- 697 piece, 9.015 performance.
- 600 piece lengkap.
- 96 piece memiliki folder `performances/` kosong.
- 1 piece Beethoven (`BeethovenLv__O111__lvb_sonate_111_1`) memiliki 15 performance
  tetapi semua `features/` kosong.
- Total 97 piece tidak lengkap; tidak ada yang masuk subset 25 piece.
- Subset: 25 piece; 24 piece memiliki MuNG; 1 piece dikecualikan dari core/Task 1.
- Split memakai `group_by_work_group`, sehingga tidak ada `work_group` yang muncul
  di dua split.

---

## Struktur repo

```text
configs/preprocess.yaml
data/
  subset_pieces.csv
  subset_performances.csv
  splits.csv
  processed/                 # artefak kecil di Git, tensor besar di Release
notebooks/
  01_eda_preprocessing.ipynb
  02_note_level_preprocessing.ipynb
scripts/
  run_preprocessing.py
  build_manifest.py
  build_release.py
  build_dataset_inventory.py
src/msmd_preprocess/
  config.py  dataset.py  io.py  mung.py  paths.py
  core.py    task1.py    templates.py  task2.py  validate.py  bundles.py
reports/
  validation_preprocessing.json
```

Notebook `01`/`02` adalah referensi historis. Pipeline baru tidak menjalankan notebook;
gunakan `scripts/run_preprocessing.py`.

---

## Branch

- `main`: stable.
- `dev`: integrasi.
- `feature/preprocessing-dhea`: branch kerja preprocessing.
- `01-preprocessing`: arsip pipeline lama; **tidak dihapus**.

Alur: `feature/preprocessing-dhea` -> `dev` -> `main`.

---

## Batasan

- Task 1 hanya level **system** karena anotasi bar kosong.
- Task 2 memakai error **sintetis**, bukan label human; kualitasnya perlu dicek saat
  training.
- Delete note bisa halus secara energi jika window berisi banyak note; model perlu
  memakai bukti spektral lokal, bukan hanya energi total.
- Template pitch di luar range direct memakai fallback ridge/nearest/global.
- Tidak ada training/model di repo ini.
