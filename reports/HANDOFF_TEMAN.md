# Handoff Preprocessing MSMD

Panduan ini untuk teman yang akan memakai artefak preprocessing dan melakukan
training. Training bukan bagian dari repo preprocessing ini.

## 1. Yang diterima

Dataset pilot:

- 25 piece terpilih.
- 24 piece usable.
- 96 performance usable.
- 18.813 crop notehead.
- 46.522 window audio.
- 232.610 pair Task 1.
- 186.088 window Task 2.

Data besar ada di `data/processed/`, tetapi file `.npy`/`.npz` tidak masuk Git.
Distribusinya lewat GitHub Release atau folder `dist/`.

Yang **tidak** diperlukan:

- dataset mentah 18 GB,
- regenerate preprocessing,
- notebook lama (sudah dihapus; arsip di branch `01-preprocessing`),
- HTML lama (sudah dihapus; gunakan `reports/preprocessing_report_current.html`).

## 2. Prasyarat

- Python 3.11 atau lebih baru; pipeline diuji di Python 3.14.
- Git.
- Ruang disk sekitar 5 GB jika zip dan hasil ekstrak disimpan bersamaan.
- GPU opsional; CPU masih bisa untuk model kecil pada 25 piece.
- PyTorch/torchvision belum ada di `requirements.txt`; training perlu memasangnya
  terpisah.

## 3. Ambil source code

Sementara ini pakai branch kerja:

```powershell
git clone https://github.com/violaherfinap/msmd-learning.git
cd msmd-learning
git checkout feature/preprocessing-dhea
```

Commit terakhir preprocessing:

```text
dfa4fa5 feat: complete task1/task2 preprocessing and validation
```

Kalau repo private, minta akses collaborator.

Setelah PR ke `dev` selesai, teman bisa memakai branch `dev` atau `main`.

## 4. Download artefak

Kalau Release sudah diupload:

```text
https://github.com/violaherfinap/msmd-learning/releases/tag/preproc-v1
```

Asset:

- `preproc-v1-core.zip`
- `preproc-v1-task1.zip`
- `preproc-v1-task2.zip`
- `preproc-v1-meta.zip`
- `SHA256SUMS.txt`

Kalau Release belum ada, minta folder `dist/` dari pemilik repo.

## 5. Extract

Cara yang disarankan: jalankan dari root repo.

```powershell
.\scripts\extract_release.ps1 -DryRun
.\scripts\extract_release.ps1
```

Kalau zip ada di folder lain:

```powershell
.\scripts\extract_release.ps1 -SourceDir "D:\path\ke\zip"
```

Kalau hanya ingin extract tanpa validasi:

```powershell
.\scripts\extract_release.ps1 -SkipVerify
```

Jangan extract ke dalam `data/processed`. Script ini sudah mengarahkan ke root repo
dan akan menolak folder tujuan yang salah.

Struktur yang benar:

```text
msmd-learning\
  data\processed\audio_windows_train.npy
  data\processed\score_crops_train.npy
  data\processed\system_crops_train.npy
  data\processed\error_windows_train.npy
```

Struktur yang salah:

```text
msmd-learning\data\processed\data\processed\audio_windows_train.npy
```

## 6. Install dependency

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pasang PyTorch sesuai mesin:

```powershell
python -m pip install torch torchvision
```

Ikuti instruksi resmi PyTorch jika butuh versi CUDA tertentu.

## 7. Verifikasi

```powershell
python scripts\run_preprocessing.py --verify-only
```

Harus muncul:

```json
"ok": true
```

dan:

```text
Validation passed
```

Jangan jalankan:

```powershell
python scripts\run_preprocessing.py --stage all
```

Itu membutuhkan dataset mentah dan akan menulis ulang artefak.

## 8. Cek QA visual

Buka:

```text
reports/qa_v2/
```

File penting:

- `01_notehead_page_overlay.png`: overlay notehead di halaman.
- `02_notehead_crops.png`: contoh crop notehead.
- `03_audio_windows.png`: spectrogram dan onset.
- `04_task1_system_overlay.png`: system positif/negatif di halaman.
- `05_task1_system_crops.png`: kandidat Task 1.
- `06_task2_class_grid.png`: clean/wrong_pitch/delete/insert.
- `07_task2_diff_examples.png`: difference map Task 2.
- `qa_summary.json`: ringkasan check.
- `trace_samples.json`: trace 3 sampel raw ke artifact.

QA visual ini sudah diperiksa. Kalau ada yang aneh di lingkungan baru, laporkan
sebelum training.

## 9. Skema data untuk training

### Core

| File | Isi |
|---|---|
| `score_crops_{split}.npy` | `uint8 [N,200,160,1]`, crop gambar mentah |
| `audio_windows_{split}.npy` | `float16 [N,42,92]`, sudah normalized |
| `noteheads_{split}.parquet` | metadata crop + `crop_row` |
| `windows_{split}.parquet` | metadata window + `window_row` |
| `pairs_note_{split}.parquet` | pair note-level |

### Task 1

| File | Isi |
|---|---|
| `score_pages_{split}.npy` | `uint8 [N,256,256,1]` |
| `system_crops_{split}.npy` | `uint8 [N,128,512,1]` |
| `pages_{split}.parquet` | metadata halaman |
| `systems_{split}.parquet` | metadata system + bbox + `crop_row` |
| `notehead_systems_{split}.parquet` | assignment notehead ke system |
| `pairs_system_{split}.parquet` | 5 kandidat per query |

### Task 2

| File | Isi |
|---|---|
| `error_windows_{split}.npy` | `float16 [N,42,92]` |
| `task2_windows_{split}.parquet` | metadata varian + `class_id` |
| `task2_class_map.json` | mapping kelas |
| `templates_task2.npz` | hanya untuk generasi, bukan training |

## 10. Loader Task 1

```python
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
split = "train"

audio = np.load(PROC / f"audio_windows_{split}.npy", mmap_mode="r")
systems = np.load(PROC / f"system_crops_{split}.npy", mmap_mode="r")
pairs = pd.read_parquet(PROC / f"pairs_system_{split}.parquet")

a = audio[pairs["window_row"].to_numpy()]
s = systems[pairs["candidate_crop_row"].to_numpy()]
y = pairs["label"].to_numpy()
```

Aturan:

- 5 baris per `query_id`: 1 positif dan 4 negatif.
- `label=1` adalah system yang cocok.
- `audio_windows` sudah normalized.
- `system_crops` masih `uint8` mentah; normalisasi dengan
  `norm_stats_task1.json`:
  - page mean/std `0.9429144 / 0.1269404`,
  - system mean/std `0.9099208 / 0.2053268`.
- Jangan split ulang secara acak; gunakan split yang tersedia.
- Untuk evaluasi retrieval, ranking kandidat per `query_id`.

## 11. Loader Task 2

```python
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
split = "train"

errors = np.load(PROC / f"error_windows_{split}.npy", mmap_mode="r")
meta = pd.read_parquet(PROC / f"task2_windows_{split}.parquet")

x = errors[meta["error_row"].to_numpy()]
y = meta["class_id"].to_numpy()
```

Aturan:

- `class_id`: `0=clean`, `1=wrong_pitch`, `2=delete`, `3=insert`.
- Setiap source window punya 4 varian.
- `error_windows` sudah normalized.
- Jangan split baris secara acak; gunakan split piece yang sudah ada.
- `templates_task2.npz` tidak perlu untuk training classifier.

## 12. Normalisasi

- `score_crops`: `(pixel / 255 - 0.8936307) / 0.2651336`.
- `system_crops`: `(pixel / 255 - 0.9099208) / 0.2053268`.
- `score_pages`: `(pixel / 255 - 0.9429144) / 0.1269404`.
- `audio_windows`: sudah normalized; jangan normalisasi ulang.
- `error_windows`: sudah normalized; jangan normalisasi ulang.

Gunakan `norm_stats_note.json` dan `norm_stats_task1.json` sebagai acuan angka.

## 13. Saran baseline training

Karena ini pilot, mulai dari model kecil.

### Task 2 dulu

- Input `[1, 42, 92]`.
- CNN 2D kecil, 3 blok konvolusi, global average pooling, linear 4 kelas.
- Loss cross-entropy.
- Metrik: accuracy, macro F1, per-class precision/recall, confusion matrix.
- Tuning hanya di val; test sekali di akhir.

### Task 1

- Audio encoder + system image encoder.
- Skor dot product atau pair classifier.
- Metrik: Recall@1, Recall@5, MRR per `query_id`.
- Tuning hanya di val; test sekali di akhir.

Gunakan batch kecil dulu, `mmap_mode="r"`, dan seed tetap.

## 14. Batasan yang harus ditulis di laporan

- Pilot 25 piece, 24 usable; test hanya 3 piece.
- Hasil belum bisa diklaim untuk seluruh MSMD 697 piece.
- Task 1 hanya level system, bukan bar.
- Task 2 memakai error sintetis, bukan label human.
- Tidak ada audio mentah; hanya spektrogram dan MIDI.
- `delete` bisa halus secara energi.
- `insert` bisa menambah pitch yang sudah ada.
- `norm_stats_note.json` punya field legacy `segment_frames=100`; artifact window
  sekarang 42 frame. Field itu tidak dipakai loader.

## 15. Troubleshooting

### `audio_windows_train.npy` tidak ditemukan

Zip belum diekstrak ke root repo, atau ekstraksi terjadi di folder yang salah.

### `Validation failed`

Cek:

```powershell
python scripts\run_preprocessing.py --verify-only
```

Pastikan semua zip diekstrak dan tidak ada file yang terpotong.

### GPU tidak terdeteksi

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Kalau `False`, pakai CPU atau environment Colab/Kaggle.

### Training lambat

- Gunakan model lebih kecil.
- Kurangi `num_workers`.
- Pastikan memakai `mmap_mode="r"` dan tidak menyalin seluruh tensor ke RAM.

### Hasil terlalu bagus

Curigai overfit atau leakage. Cek:

- split tidak diubah,
- test tidak dipakai untuk tuning,
- tidak ada performance dari piece yang sama di dua split.

## 16. Checklist teman

- [ ] Clone branch `feature/preprocessing-dhea`.
- [ ] Download empat zip Release atau terima folder `dist/`.
- [ ] Extract ke root repo.
- [ ] `Test-Path data\processed\audio_windows_train.npy` bernilai `True`.
- [ ] Install `requirements.txt`.
- [ ] Install PyTorch.
- [ ] `python scripts\run_preprocessing.py --verify-only` → `ok: true`.
- [ ] Buka `reports/qa_v2/`.
- [ ] Mulai loader Task 2.
- [ ] Latih baseline Task 2.
- [ ] Setelah itu baseline Task 1.
- [ ] Jangan commit `.npy`/`.npz`.
- [ ] Tulis batasan pilot 25 piece di laporan.

## 17. Dokumen acuan

- `reports/preprocessing_report_current.html`
- `reports/PREPROCESSING_SUMMARY.md`
- `reports/qa_v2/qa_summary.json`
- `reports/qa_v2/trace_samples.json`
- `reports/validation_preprocessing.json`
- `data/processed/manifest_preprocessing.json`
- `data/processed/SHA256SUMS.txt`

Dokumen/HTML lama sudah dihapus dan diarsipkan di branch `01-preprocessing`.
Jangan dipakai sebagai acuan current.
