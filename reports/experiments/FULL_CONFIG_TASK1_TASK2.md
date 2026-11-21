# Konfigurasi Lengkap Task 1 dan Task 2

Dokumen ini merangkum konfigurasi eksperimen Task 1 (score-audio alignment) dan Task 2 (performance error detection) yang dijalankan di repo ini. Nilai desimal memakai titik. Sumber kebenaran tetap `config.json` per run dan checkpoint `best.pt`; dokumen ini adalah ringkasan yang dapat dibaca manusia.

> Rencana perbandingan Task 2 C1-C4 ada di [`RENCANA_TASK1_TASK2.md`](RENCANA_TASK1_TASK2.md).

- Tanggal: 21 November 2026
- Seed utama: `42`
- Scope: Task 1 C1-C4; Task 2 baseline, robust mild, robust strong, audio-only, cross-soundfont
- Dokumen ini menggantikan deck `konfigurasi_task1_task2.pptx` yang lama.

---

## 1. Data dan preprocessing

### 1.1 Sumber data

| Item | Nilai |
|---|---|
| Raw dataset | `E:\SMT 5\msmd_aug_v1-1_extracted\msmd_aug_v1-1_no-audio` |
| Junction repo | `msmd_aug_v1-1_no-audio` |
| Processed dir | `data/processed/` |
| Backup historis | `data/backups/historical_v1/` |
| Backup sebelum regenerasi Task 2 | `data/backups/task2_before_regen/` |

### 1.2 Konfigurasi `configs/preprocess.yaml`

| Parameter | Nilai |
|---|---|
| `seed` | `42` |
| `fps` | `20` |
| `spec_bins` | `92` |
| `win_frames` | `42` |
| `crop_w` / `crop_h` | `160` / `200` |
| `train_perfs_per_note` | `2` |
| `neg_per_pos` | `1` |
| Task 1 `page_size` | `256` |
| Task 1 `system_crop_h` / `system_crop_w` | `128` / `512` |
| Task 1 `system_margin` | `4` |
| Task 1 `negatives_per_positive` | `4` |
| Task 2 `template_min_frames` | `20` |
| Task 2 `ridge_lambda` | `0.001` |
| Task 2 `max_retries` | `3` |
| Task 2 `max_per_class` | `0` (tanpa batas) |
| Task 2 `fallback` | `hybrid` |
| Task 2 `window_decay_frames` | `10` |

### 1.3 Jumlah sampel

| Split | Crops | Windows | Pairs | Task 2 error windows |
|---|---:|---:|---:|---:|
| train | 14,365 | 28,730 | 57,460 | 114,920 |
| val | 2,430 | 9,720 | 19,440 | 38,880 |
| test | 2,018 | 8,072 | 16,144 | 32,288 |
| total | - | - | - | 186,088 |

Catatan Task 2:

- 0 source window terlewat.
- 42 fallback: 26 train, 12 val, 4 test.
- Setiap source window memiliki 4 varian: `clean`, `wrong_pitch`, `delete`, `insert`.

Verifikasi SHA256 terhadap snapshot historis: 57 file cocok, 0 file hilang, 3 file berbeda (`configs/preprocess.yaml`, `requirements.txt`, `README.md`). Seluruh artefak data cocok.

---

## 2. Task 1 - Score-audio alignment

### 2.1 Arsitektur

| Komponen | Konfigurasi |
|---|---|
| Model | `DualEncoder` Dorfer full |
| Sheet input | `80 x 100` |
| Audio input | log-mel `92 x 42` |
| Conv blocks | 4 blok, channel `24 / 48 / 96 / 96` |
| Embedding | Conv1x1-32 + GAP, `embed_dim=32` |
| CCA | Proyeksi CCA + L2-normalize; dibekukan |
| Loss | Ranking hinge, `margin=0.7` |
| Negatives | 4 negatif per positif |

### 2.2 Konfigurasi training bersama

| Parameter | Nilai |
|---|---|
| Seed | `42` |
| Device | `auto` (CPU pada environment ini) |
| Batch size | `100` |
| Eval batch size | `256` |
| Epochs | `8` |
| Early stop patience | `3` |
| Optimizer | Adam |
| Learning rate | `0.002` |
| Weight decay | `0.0001` (C4: `0.001`) |
| Gradient clip | `5.0` |
| CCA `alpha` | `0.05` |
| CCA `warmup_steps` | `200` |
| CCA `r1` / `r2` / `rT` | `0.001` / `0.001` / `0.001` |
| Image scale | `0.95 - 1.05` |
| Image translate | `5.0` |
| Audio time shift | `1` |
| Audio tempo jitter | `0.1` |
| Num workers | `0` |

### 2.3 Matriks arm

| Arm | Perubahan | `use_cca` | `dropout` | `mining` | Weight decay |
|---|---|---:|---:|---|---:|
| C1 | Baseline | `true` | `0.0` | `null` | `0.0001` |
| C2 | Ablasi CCA | `false` | `0.0` | `null` | `0.0001` |
| C3 | Hard-negative mining satu arah, `k=10`, warmup 1 epoch | `true` | `0.0` | `hard` | `0.0001` |
| C4 | Dropout setelah CCA + regularisasi lebih kuat | `true` | `0.2` | `null` | `0.001` |

Semua arm dilatih dari awal dan memakai split yang sama.

### 2.4 Gate A - reproduksi baseline

| Metrik | Historis (%) | C1 (%) | Delta (pp) | Toleransi (pp) | Status |
|---|---:|---:|---:|---:|---|
| val MRR | 1.001 | 1.168 | +0.167 | 0.200 | PASS |
| snippet A->S R@1 | 0.099 | 0.099 | 0.000 | - | n/a |
| full R@1 | 6.925 | 7.507 | +0.582 | 1.000 | PASS |
| 5-candidate R@1 | 27.329 | 29.125 | +1.796 | 2.000 | PASS |

C1 lolos Gate A. Lintasan training berbeda dari historis (best epoch 7 vs 3) karena perbedaan environment/numerik.

### 2.5 Hasil utama (%)

| Arm | Best epoch | Val MRR | Snippet A->S R@1 | Snippet A->S MRR | Page R@1 | Page MRR | Full R@1 | Full MRR | 5-cand R@1 | 5-cand MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 7 | 1.168 | 0.099 | 0.682 | 30.785 | 53.949 | 7.507 | 20.601 | 29.125 | 54.950 |
| C2 | 7 | 1.377 | 0.297 | 1.199 | 30.104 | 54.358 | 9.093 | 21.591 | 29.993 | 55.105 |
| C3 | 4 | 2.264 | 0.496 | 1.744 | 28.147 | 53.043 | 10.803 | 23.252 | 29.559 | 54.150 |
| C4 | 6 | 0.774 | 0.099 | 0.688 | 26.722 | 51.321 | 4.274 | 17.261 | 25.483 | 52.317 |

Delta C3 vs C1: full MRR `+2.652 pp`, snippet A->S R@1 `+0.396 pp`, 5-candidate R@1 `+0.434 pp`. Belum ada arm yang melewati ambang sistem `+5 pp`; hasil bersifat eksploratif (1 seed).

### 2.6 Artefak Task 1

| Item | Path |
|---|---|
| Notebook | `notebooks/05_task1_experiments.ipynb` |
| Run | `reports/experiments/T1-C1_seed42/` sampai `T1-C4_seed42/` |
| Config per arm | `reports/experiments/T1-C*_seed42/config.json` |
| Metrics/history | `reports/experiments/T1-C*_seed42/{metrics.json,history.csv}` |
| Checkpoint | `checkpoints/T1-C1_seed42/best.pt` sampai `T1-C4_seed42/best.pt` |
| Perbandingan | `reports/experiments/task1_comparison.md`, `task1_comparison.csv`, `task1_deltas_vs_C1.csv`, `task1_gate_a.csv`, `task1_per_piece.csv` |

---

## 3. Task 2 - Performance error detection

### 3.1 Arsitektur

| Komponen | Konfigurasi |
|---|---|
| Encoder awal | Checkpoint Task 1 C1: `checkpoints/T1-C1_seed42/best.pt` |
| Sheet encoder | Sama dengan Task 1, `embed_dim=32` |
| Audio encoder | Sama dengan Task 1, `embed_dim=32` |
| Fusion | `[x, y, x*y, abs(x-y)]` |
| Classifier | MLP `4*32 -> 128 -> 4`, dropout `0.2` |
| Kelas | `correct`, `wrong_pitch`, `missing_note`, `extra_note` |
| Loss | Cross-entropy |
| Label | Error sintetis dari preprocessing Task 2 |

### 3.2 Konfigurasi training bersama

| Parameter | Nilai |
|---|---|
| Seed | `42` |
| Embed dim | `32` |
| Channels | `24 / 48 / 96 / 96` |
| Hidden dim | `128` |
| Dropout | `0.2` |
| Num classes | `4` |
| Batch size | `256` |
| Eval batch size | `512` |
| Epochs | `8` |
| Early stop patience | `3` |
| Optimizer | Adam |
| LR encoder | `0.0003` |
| LR head | `0.001` |
| Weight decay | `0.00001` |
| Gradient clip | `5.0` |
| `freeze_sheet` | `false` (kecuali audio-only) |
| `train_one_variant_per_source` | `true` |
| Num workers | `0` |
| Scheduler | `ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-6)` |

### 3.3 Matriks arm Task 2

| Run | Variant | Augmentation | Holdout | Encoder | Best epoch | Clean test macro-F1 |
|---|---|---|---:|---:|---:|---:|
| `T2-baseline_seed42` | multimodal | none | - | sheet + audio dilatih | 5 | 0.9929 |
| `T2-robust_seed42` | multimodal | mild | - | sheet + audio dilatih | 6 | 0.9871 |
| `T2-robust-strong_seed42` | multimodal | strong | - | sheet + audio dilatih | 6 | 0.9890 |
| `T2-audio-only_seed42` | audio-only | mild audio | - | sheet dibekukan, score di-nol-kan | 6 | 0.9874 |
| `T2-cross-soundfont_seed42` | multimodal | none | `ElectricPiano` | sheet + audio dilatih | 6 | 0.8821 |

Catatan:

- Baseline Task 2 dibuat dari `notebooks/04_baseline_task2_multimodal_classifier.ipynb`; tidak memiliki `config.json` terpisah. Konfigurasi tersimpan di checkpoint.
- Audio-only memakai score crop nol dan sheet encoder dibekukan; optimizer hanya memperbarui audio encoder dan classifier. Score noise/cutout dinonaktifkan saat training.
- Audio-only berhenti setelah epoch 7; checkpoint terbaik adalah epoch 6, dan evaluasi test memakai checkpoint tersebut.
- Cross-soundfont melatih model pada 3 soundfont selain `ElectricPiano`; validasi memakai soundfont yang terlihat, test memakai `ElectricPiano` yang tidak terlihat saat training.

### 3.4 Preset augmentasi

| Parameter | Mild | Strong |
|---|---:|---:|
| Audio Gaussian noise probability | 0.5 | 0.7 |
| Audio noise SNR (dB) | 10 - 20 | 5 - 15 |
| Time mask probability | 0.5 | 0.7 |
| Time mask width (frame) | 1 - 5 | 1 - 8 |
| Frequency mask probability | 0.5 | 0.7 |
| Frequency mask width (bin) | 1 - 8 | 1 - 12 |
| Score noise probability | 0.3 | 0.3 |
| Score noise sigma | 0.02 - 0.05 | 0.02 - 0.08 |
| Score cutout probability | 0.3 | 0.3 |
| Score cutout size (px) | 8 - 16 | 8 - 20 |

Untuk audio-only, score noise/cutout probability di-set `0.0`; augmentasi audio tetap memakai preset mild.

### 3.5 Kondisi evaluasi corruption

Evaluasi test-time memakai seed deterministik per sampel, sehingga semua model diuji pada corruption yang identik.

| Kondisi | Corruption |
|---|---|
| `clean` | Tanpa corruption |
| `audio_noise_snr20` | Gaussian noise, SNR 20 dB |
| `audio_noise_snr10` | Gaussian noise, SNR 10 dB |
| `audio_noise_snr5` | Gaussian noise, SNR 5 dB |
| `time_mask10` | Time mask 10 frame |
| `frequency_mask10` | Frequency mask 10 bin |
| `combined_mild` | Noise SNR 10 dB + time mask 5 + frequency mask 5 |
| `score_noise_0.1` | Score noise sigma 0.1 |
| `score_cutout_16` | Score cutout 16 px |

### 3.6 Hasil corruption (macro-F1)

| Condition | Baseline | Robust mild | Robust strong | Audio-only | Cross-soundfont |
|---|---:|---:|---:|---:|---:|
| clean | 0.9929 | 0.9871 | 0.9890 | 0.9874 | 0.8821 |
| audio noise SNR 20 dB | 0.9812 | 0.9904 | 0.9888 | 0.9844 | 0.7174 |
| audio noise SNR 10 dB | 0.3588 | 0.9647 | 0.9725 | 0.9559 | 0.1521 |
| audio noise SNR 5 dB | 0.1803 | 0.7657 | 0.9468 | 0.8387 | 0.1185 |
| time mask 10 | 0.9795 | 0.9813 | 0.9826 | 0.9811 | 0.8486 |
| frequency mask 10 | 0.6727 | 0.8931 | 0.9230 | 0.8789 | 0.4555 |
| combined mild | 0.3746 | 0.9219 | 0.9531 | 0.9328 | 0.2066 |
| score noise 0.1 | 0.9925 | 0.9876 | 0.9888 | 0.9874 | 0.8811 |
| score cutout 16 | 0.9929 | 0.9871 | 0.9892 | 0.9874 | 0.8822 |

Interpretasi singkat:

- Robust strong memberi ketahanan terbaik pada noise berat dan frequency masking, dengan clean test yang lebih tinggi daripada robust mild.
- Audio-only hampir setara robust mild pada clean, tetapi lebih lemah pada frequency masking dan noise sedang.
- Cross-soundfont turun tajam pada test `ElectricPiano`, menunjukkan domain shift timbre yang belum tertangani.
- Score corruption hampir tidak mengubah performa model multimodal; model sangat bergantung pada audio.

### 3.7 Artefak Task 2

| Item | Path |
|---|---|
| Notebook baseline | `notebooks/04_baseline_task2_multimodal_classifier.ipynb` |
| Baseline | `reports/experiments/T2-baseline_seed42/` |
| Robust mild | `reports/experiments/T2-robust_seed42/` |
| Robust strong | `reports/experiments/T2-robust-strong_seed42/` |
| Audio-only | `reports/experiments/T2-audio-only_seed42/` |
| Cross-soundfont | `reports/experiments/T2-cross-soundfont_seed42/` |
| Config per run | `reports/experiments/T2-*/config.json` (baseline dan audio-only: config ada di checkpoint) |
| Checkpoint | `checkpoints/T2-*/best.pt` |
| Corruption eval | `reports/experiments/T2-*/corruption_metrics.{csv,json}` |
| Perbandingan | `reports/experiments/task2_comparison.md`, `task2_comparison.csv`, `task2_clean_summary.csv`, `task2_corruption_comparison.png` |
| Script | `scripts/task2_corruption.py`, `scripts/train_task2_robust.py`, `scripts/eval_task2_corruptions.py` |

---

## 4. Perintah reproduksi

### 4.1 Task 1

Jalankan `notebooks/05_task1_experiments.ipynb` dan set `VARIANT` ke `C1`, `C2`, `C3`, atau `C4`. Semua arm memakai seed 42 dan konfigurasi di atas.

### 4.2 Task 2

Baseline Task 2:

```powershell
# Jalankan notebooks/04_baseline_task2_multimodal_classifier.ipynb
```

Robust mild:

```powershell
python scripts/train_task2_robust.py --epochs 8 --augmentation mild --out-name T2-robust_seed42
```

Robust strong:

```powershell
python scripts/train_task2_robust.py --epochs 8 --augmentation strong --out-name T2-robust-strong_seed42
```

Audio-only:

```powershell
python scripts/train_task2_robust.py --epochs 8 --variant audio-only --out-name T2-audio-only_seed42
```

Cross-soundfont:

```powershell
python scripts/train_task2_robust.py --epochs 8 --augmentation none --holdout-soundfont ElectricPiano --out-name T2-cross-soundfont_seed42
```

Evaluasi corruption:

```powershell
python scripts/eval_task2_corruptions.py --checkpoint checkpoints/T2-robust-strong_seed42/best.pt --out-dir reports/experiments/T2-robust-strong_seed42
```

---

## 5. Batasan dan catatan

- Semua hasil memakai satu seed (`42`); variansi antar-seed belum diukur.
- Task 2 memakai label error sintetis, bukan anotasi musisi; test hanya 3 piece.
- Baseline Task 2 dan audio-only tidak memiliki `config.json`; konfigurasi diambil dari checkpoint `best.pt`.
- Audio-only berhenti setelah epoch 7 karena eksekusi dihentikan; checkpoint terbaik epoch 6 tetap dievaluasi.
- Cross-soundfont memakai validasi pada soundfont yang terlihat; angka val `0.9830` bukan val pada `ElectricPiano`.
- Evaluasi Task 1 dan Task 2 masih terpisah; belum ada evaluasi end-to-end Task 1 -> crop -> Task 2.
- Deck PPTX lama sudah dihapus; dokumen Markdown ini menjadi acuan konfigurasi Task 1 dan Task 2.
