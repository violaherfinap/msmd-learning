# Laporan Eksperimen MSMD — Task 1 dan Task 2

Tanggal: 20 November 2026  
Seed: 42 (satu seed, eksploratif)  
Device: CPU  
Referensi: Dorfer et al. (2018), TISMIR 1(1):22–33

> Catatan konfigurasi: laporan ini fokus pada C1–C4, Task 2 baseline, dan Task 2 robust mild. Konfigurasi lengkap semua arm (termasuk robust-strong, audio-only, cross-soundfont) ada di [`FULL_CONFIG_TASK1_TASK2.md`](FULL_CONFIG_TASK1_TASK2.md).

## 1. Ringkasan eksekutif

- **Task 1** — C1 (baseline) berhasil mereproduksi baseline historis dalam toleransi Gate A. C3 (hard-negative mining) adalah kandidat terbaik: val MRR +1,10 pp, full MRR +2,65 pp, snippet A→S R@1 +0,40 pp dibanding C1. C2 (tanpa CCA) mendekati C3 pada metrik sistem. C4 (dropout + weight decay) menurun. Tidak ada arm yang melewati ambang +5 pp pada metrik sistem, sehingga hasil masih eksploratif.
- **Task 2** — Baseline berhasil direproduksi (test macro-F1 0,9929 vs historis 0,9954). Robust training dengan augmentasi audio ringan menurunkan clean macro-F1 sebesar 0,58 pp, tetapi meningkatkan robustness secara besar: noise SNR 10 dB +60,59 pp, SNR 5 dB +58,54 pp, frequency mask 10 +22,04 pp, combined mild +54,73 pp. Corruption pada score crop hampir tidak berpengaruh pada kedua model; model sangat bergantung pada audio.
- **Rekomendasi** — C3 layak divalidasi multi-seed sebelum dijadikan konfigurasi Task 1 baru. Robust training direkomendasikan untuk skenario audio bising; baseline tetap lebih baik untuk audio bersih.

## 2. Data dan preprocessing

### 2.1 Sumber data

- Raw dataset: `E:\SMT 5\msmd_aug_v1-1_extracted\msmd_aug_v1-1_no-audio` (18,16 GB, 697 direktori top-level).
- Junction repo: `msmd_aug_v1-1_no-audio` → raw dataset.
- Backup metadata dan report historis: `data/backups/historical_v1/`.

### 2.2 Regenerasi artefak

Preprocessing core, Task 1, dan Task 2 diregenerasi. Verifikasi SHA256 terhadap snapshot historis:

- 57 file cocok.
- 0 file hilang.
- 3 file berbeda, semuanya dokumentasi/konfigurasi: `configs/preprocess.yaml`, `requirements.txt`, `README.md`.
- Seluruh artefak data (termasuk `error_windows_*.npy` dan `templates_task2.npz`) cocok dengan snapshot historis.

Jumlah sampel:

| Split | Crops | Windows | Pairs | Task 2 error windows |
|---|---:|---:|---:|---:|
| train | 14.365 | 28.730 | 57.460 | 114.920 |
| val | 2.430 | 9.720 | 19.440 | 38.880 |
| test | 2.018 | 8.072 | 16.144 | 32.288 |
| total | — | — | — | 186.088 |

Task 2: 0 source window terlewat, 42 fallback (26 train, 12 val, 4 test). Setiap source window memiliki 4 varian: `clean`, `wrong_pitch`, `delete`, `insert`.

### 2.3 Catatan teknis

- `src/msmd_preprocess/paths.py` diperbaiki agar `relative_posix` memakai `os.path.abspath` (bukan `Path.resolve()`), sehingga junction tidak resolve ke luar repo dan path metadata tetap relatif.
- Artefak Task 2 sebelumnya ditunda; sekarang sudah lengkap dan tervalidasi.

## 3. Task 1 — Score–audio alignment

### 3.1 Konfigurasi

- Model: `DualEncoder` Dorfer full; sheet `80×100`, audio `92×42`; 4 blok `24/48/96/96`; Conv1×1-32 + GAP.
- CCA dibekukan: `alpha=0.05`, `warmup_steps=200`, `r1=r2=rT=1e-3`.
- Loss: ranking hinge margin `0.7`.
- Optimizer: Adam `2e-3`, weight decay `1e-4`, batch `100`, maksimum 8 epoch, early stop 3.
- Seed 42; semua arm dilatih dari awal.
- Notebook: `notebooks/05_task1_experiments.ipynb` dengan parameter `VARIANT`.

Arm:

| Arm | Perubahan |
|---|---|
| C1 | Baseline; CCA aktif |
| C2 | Ablasi CCA (`use_cca=False`) |
| C3 | Hard-negative mining satu arah, `k=10`, warmup 1 epoch |
| C4 | Dropout `0.2` setelah CCA + weight decay `1e-3` |

### 3.2 Gate A — reproduksi baseline

| Metrik | Historis (%) | C1 (%) | Delta (pp) | Toleransi (pp) | Status |
|---|---:|---:|---:|---:|---|
| val MRR | 1,001 | 1,168 | +0,167 | 0,200 | PASS |
| snippet A→S R@1 | 0,099 | 0,099 | 0,000 | — | n/a |
| full R@1 | 6,925 | 7,507 | +0,582 | 1,000 | PASS |
| 5-candidate R@1 | 27,329 | 29,125 | +1,796 | 2,000 | PASS |

C1 lolos Gate A. Lintasan training berbeda dari historis (best epoch 7 vs 3) karena perbedaan environment/numerik; metrik utama tetap dalam toleransi.

### 3.3 Hasil utama (%)

| Arm | Best epoch | Val MRR | Snippet A→S R@1 | Snippet A→S MRR | Page R@1 | Page MRR | Full R@1 | Full MRR | 5-cand R@1 | 5-cand MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 7 | 1,168 | 0,099 | 0,682 | 30,785 | 53,949 | 7,507 | 20,601 | 29,125 | 54,950 |
| C2 | 7 | 1,377 | 0,297 | 1,199 | 30,104 | 54,358 | 9,093 | 21,591 | 29,993 | 55,105 |
| C3 | 4 | 2,264 | 0,496 | 1,744 | 28,147 | 53,043 | 10,803 | 23,252 | 29,559 | 54,150 |
| C4 | 6 | 0,774 | 0,099 | 0,688 | 26,722 | 51,321 | 4,274 | 17,261 | 25,483 | 52,317 |

Identifikasi:

| Arm | Piece ID R@1 | Page ID A→S R@1 | Page ID S→A R@1 |
|---|---:|---:|---:|
| C1 | 46,494 | 30,017 | 24,777 |
| C2 | 52,713 | 27,837 | 32,309 |
| C3 | 48,897 | 33,461 | 34,242 |
| C4 | 55,575 | 21,977 | 17,047 |

### 3.4 Delta vs C1 (pp)

| Arm | Snippet A→S R@1 | Snippet A→S MRR | Full MRR | 5-cand R@1 |
|---|---:|---:|---:|---:|
| C2 | +0,198 | +0,517 | +0,990 | +0,867 |
| C3 | +0,396 | +1,063 | +2,652 | +0,434 |
| C4 | 0,000 | +0,006 | -3,339 | -3,642 |

Konsistensi per-piece (3 piece):

- Snippet A→S R@1: C2 dan C3 membaik di **3/3 piece**.
- System 5-candidate R@1: C2 dan C3 membaik di **1/3 piece**, menurun di **2/3 piece**.
- Full MRR: C3 membaik di **2/3 piece**, menurun di **1/3 piece**.

### 3.5 Interpretasi

- **C3** unggul pada val MRR, full MRR, dan snippet; perbaikan snippet konsisten di semua piece. Namun perbaikan metrik sistem belum melewati ambang +5 pp.
- **C2** menunjukkan bahwa CCA tidak memberikan keuntungan pada konfigurasi ini; C2 bahkan sedikit lebih baik pada 5-candidate R@1. Ini bisa dipengaruhi `alpha=0.05` dan batch 100; belum diuji sweep alpha.
- **C4** over-regularisasi: semua metrik sistem turun.
- Hasil Task 1 bersifat eksploratif (1 seed). C3 adalah kandidat, bukan konklusi final.

## 4. Task 2 — Performance error detection

### 4.1 Setup

- Encoder Task 1: checkpoint C1 (`checkpoints/T1-C1_seed42/best.pt`), sesuai Gate A.
- Model: fusion `[x, y, x*y, abs(x-y)]` → MLP → 4 kelas.
- Baseline: kedua encoder dilatih; cross-entropy; encoder `3e-4`, head `1e-3`; 8 epoch; early stop 3; batch 256.
- Robust: augmentasi audio ringan saat training:
  - Gaussian noise SNR 10–20 dB (p=0,5)
  - time mask 1–5 frame (p=0,5)
  - frequency mask 1–8 bin (p=0,5)
  - score noise σ 0,02–0,05 (p=0,3)
  - score cutout 8–16 px (p=0,3)
- Evaluasi corruption test-time memakai seed deterministik per sampel, sehingga baseline dan robust diuji pada corruption yang identik.

### 4.2 Clean test

| Model | Macro-F1 | Accuracy | Top-2 | Best epoch | Best val macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline historis | 0,9954 | 0,9954 | 1,0000 | 8 | 0,9755 |
| Baseline reproduksi | 0,9929 | 0,9929 | 1,0000 | 5 | 0,9793 |
| Robust | 0,9871 | 0,9871 | 0,9998 | 6 | 0,9656 |

Baseline reproduksi sedikit di bawah historis (-0,25 pp) dan di atas pada val (+0,38 pp). Pipeline dianggap konsisten.

Confusion matrix baseline (test):

| true \ pred | correct | wrong_pitch | missing_note | extra_note |
|---|---:|---:|---:|---:|
| correct | 7.997 | 0 | 0 | 75 |
| wrong_pitch | 0 | 8.071 | 1 | 0 |
| missing_note | 0 | 0 | 8.072 | 0 |
| extra_note | 152 | 0 | 0 | 7.920 |

Confusion matrix robust (test):

| true \ pred | correct | wrong_pitch | missing_note | extra_note |
|---|---:|---:|---:|---:|
| correct | 8.040 | 0 | 0 | 32 |
| wrong_pitch | 0 | 8.063 | 4 | 5 |
| missing_note | 178 | 0 | 7.893 | 1 |
| extra_note | 198 | 0 | 0 | 7.874 |

### 4.3 Corruption grid (macro-F1)

| Condition | Baseline | Robust | Delta (pp) |
|---|---:|---:|---:|
| clean | 0,9929 | 0,9871 | -0,58 |
| audio noise SNR 20 dB | 0,9812 | 0,9904 | +0,93 |
| audio noise SNR 10 dB | 0,3588 | 0,9647 | **+60,59** |
| audio noise SNR 5 dB | 0,1803 | 0,7657 | **+58,54** |
| time mask 10 | 0,9795 | 0,9813 | +0,19 |
| frequency mask 10 | 0,6727 | 0,8931 | **+22,04** |
| combined mild | 0,3746 | 0,9219 | **+54,73** |
| score noise 0,1 | 0,9925 | 0,9876 | -0,49 |
| score cutout 16 | 0,9929 | 0,9871 | -0,59 |

### 4.4 Interpretasi

- Robust training memberikan **trade-off** yang jelas: clean turun 0,58 pp, tetapi ketahanan terhadap noise dan frequency masking meningkat drastis.
- Baseline sangat rapuh terhadap noise SNR 10 dB dan frequency mask; pada SNR 5 dB hampir kolaps.
- Corruption score crop tidak mengubah performa baseline maupun robust. Model hampir sepenuhnya mengandalkan audio; score crop belum berkontribusi pada keputusan.
- Pada clean test, error robust lebih tersebar: `correct ↔ extra_note` dan `missing_note → correct`; baseline hanya `correct ↔ extra_note`.
- Task 2 memakai label error sintetis dan hanya 3 piece test; angka ini bukan performa pada permainan nyata.

## 5. Keterbatasan

- **Satu seed** (42); variansi antar-seed belum diukur.
- **Task 1**: CCA dibekukan pada `alpha=0.05`; belum ada sweep alpha, batch, atau arsitektur. Perbedaan environment membuat lintasan training tidak identik dengan historis.
- **Task 2**: label sintetis, bukan anotasi musisi; test hanya 3 piece; belum end-to-end (Task 1 → crop → Task 2).
- **Robust training**: satu konfigurasi augmentasi; belum diuji pada noise nyata, reverb, atau distorsi perangkat.
- **Score modality**: corruption score tidak berpengaruh; perlu analisis lebih lanjut apakah score crop terlalu lemah atau model mengabaikannya.
- Baseline Task 2 history training tidak tersimpan karena eksekusi notebook dihentikan setelah checkpoint; evaluasi memakai checkpoint terbaik (epoch 5).

## 6. Artefak

### Task 1

- Notebook: `notebooks/05_task1_experiments.ipynb`
- Run: `reports/experiments/T1-C1_seed42/` … `T1-C4_seed42/` (metrics, history, config, figures, notebook)
- Checkpoint: `checkpoints/T1-C1_seed42/best.pt` … `T1-C4_seed42/best.pt`
- Perbandingan: `reports/experiments/task1_comparison.md`, `task1_comparison.csv`, `task1_deltas_vs_C1.csv`, `task1_gate_a.csv`, `task1_per_piece.csv`

### Task 2

- Baseline: `reports/experiments/T2-baseline_seed42/` (metrics, corruption grid, predictions, error analysis, figures)
- Robust: `reports/experiments/T2-robust_seed42/` (metrics, history, config, corruption grid, figures)
- Checkpoint: `checkpoints/T2-baseline_seed42/best.pt`, `checkpoints/T2-robust_seed42/best.pt`
- Perbandingan: `reports/experiments/task2_comparison.md`, `task2_comparison.csv`, `task2_clean_summary.csv`, `task2_corruption_comparison.png`
- Script: `scripts/task2_corruption.py`, `scripts/train_task2_robust.py`, `scripts/eval_task2_corruptions.py`

### Preprocessing

- `data/processed/` (core, Task 1, Task 2)
- `data/backups/historical_v1/` (snapshot historis)
- `data/backups/task2_before_regen/` (backup sebelum regenerasi Task 2)

## 7. Rekomendasi lanjutan

1. **Validasi multi-seed** untuk C1, C2, C3 (mis. seed 123, 2024) sebelum memilih C3 sebagai konfigurasi Task 1.
2. **Robust training lanjutan**: variasi intensitas augmentasi, noise nyata/reverb, dan evaluasi pada soundfont terlemah (`ElectricPiano`).
3. **Cross-soundfont generalization**: leave-one-soundfont-out untuk mengukur generalisasi lintas instrumen.
4. **Analisis modality reliance**: mengapa score corruption tidak berpengaruh; coba score crop yang lebih informatif atau full-page reference.
5. **End-to-end evaluation**: Task 1 mencari system/page → crop → Task 2, bukan evaluasi terpisah.
6. **Laporan final** menggabungkan hasil ini dengan baseline dan batasan proyek.
