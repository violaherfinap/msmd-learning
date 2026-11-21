# Rencana Eksperimen Task 1 dan Task 2

Dokumen ini menggabungkan konfigurasi Task 1 yang sudah dijalankan dengan rencana perbandingan empat konfigurasi Task 2. Konfigurasi lengkap dan hasil per arm tetap mengacu ke [`FULL_CONFIG_TASK1_TASK2.md`](FULL_CONFIG_TASK1_TASK2.md).

- Tanggal: 21 November 2026
- Seed: `42`
- Status: Task 1 C1-C4 selesai; Task 2 C1 selesai; Task 2 C2-C4 direncanakan
- Device: CPU

---

## 1. Ringkasan status

| Bagian | Status | Catatan |
|---|---|---|
| Task 1 C1-C4 | Selesai | C1 lolos Gate A; C3 kandidat terbaik |
| Task 2 C1 | Selesai | Baseline `T2-baseline_seed42`, clean test macro-F1 `0.9929` |
| Task 2 C2 | Belum | Frozen encoder, head-only |
| Task 2 C3 | Belum | Basic fusion 64-d |
| Task 2 C4 | Belum | Pitch-shift terkontrol |
| Task 2 robust/audio-only/cross-soundfont | Selesai | Eksperimen tambahan, di luar perbandingan 4 konfigurasi |

---

## 2. Konfigurasi Task 1 yang sudah dijalankan

### 2.1 Data dan preprocessing

| Item | Nilai |
|---|---|
| Seed | `42` |
| FPS | `20` |
| Spec bins | `92` |
| Window frames | `42` |
| Crop | `160 x 200` |
| Task 1 page size | `256` |
| Task 1 system crop | `128 x 512` |
| Task 1 system margin | `4` |
| Negatives per positive | `4` |
| Train pairs | `57,460` |
| Val pairs | `19,440` |
| Test pairs | `16,144` |

### 2.2 Arsitektur

| Komponen | Konfigurasi |
|---|---|
| Model | `DualEncoder` Dorfer full |
| Sheet input | `80 x 100` |
| Audio input | log-mel `92 x 42` |
| Conv blocks | 4 blok, channel `24 / 48 / 96 / 96` |
| Embedding | Conv1x1-32 + GAP, `embed_dim=32` |
| CCA | Proyeksi CCA + L2-normalize; dibekukan |
| Loss | Ranking hinge, `margin=0.7` |

### 2.3 Hyperparameter bersama

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.002` |
| Weight decay | `0.0001` |
| Batch size | `100` |
| Eval batch size | `256` |
| Epochs | `8` |
| Early stop patience | `3` |
| Gradient clip | `5.0` |
| CCA `alpha` | `0.05` |
| CCA `warmup_steps` | `200` |
| CCA `r1` / `r2` / `rT` | `0.001` / `0.001` / `0.001` |
| Image scale | `0.95 - 1.05` |
| Image translate | `5.0` |
| Audio time shift | `1` |
| Audio tempo jitter | `0.1` |
| Num workers | `0` |

### 2.4 Empat arm Task 1

| Arm | Perubahan | `use_cca` | `dropout` | `mining` | Weight decay |
|---|---|---:|---:|---|---:|
| C1 | Baseline | `true` | `0.0` | `null` | `0.0001` |
| C2 | Ablasi CCA | `false` | `0.0` | `null` | `0.0001` |
| C3 | Hard-negative mining satu arah, `k=10`, warmup 1 epoch | `true` | `0.0` | `hard` | `0.0001` |
| C4 | Dropout setelah CCA + regularisasi lebih kuat | `true` | `0.2` | `null` | `0.001` |

Semua arm dilatih dari awal dengan split dan seed yang sama.

### 2.5 Hasil ringkas Task 1 (%)

| Arm | Best epoch | Val MRR | Full MRR | Snippet A->S R@1 | 5-cand R@1 |
|---|---:|---:|---:|---:|---:|
| C1 | 7 | 1.168 | 20.601 | 0.099 | 29.125 |
| C2 | 7 | 1.377 | 21.591 | 0.297 | 29.993 |
| C3 | 4 | 2.264 | 23.252 | 0.496 | 29.559 |
| C4 | 6 | 0.774 | 17.261 | 0.099 | 25.483 |

C1 lolos Gate A. C3 unggul pada val MRR, full MRR, dan snippet, tetapi belum melewati ambang sistem `+5 pp`; hasil masih eksploratif (1 seed).

### 2.6 Artefak Task 1

| Item | Path |
|---|---|
| Notebook | `notebooks/05_task1_experiments.ipynb` |
| Run | `reports/experiments/T1-C1_seed42/` sampai `T1-C4_seed42/` |
| Config | `reports/experiments/T1-C*_seed42/config.json` |
| Checkpoint | `checkpoints/T1-C1_seed42/best.pt` sampai `T1-C4_seed42/best.pt` |
| Perbandingan | `reports/experiments/task1_comparison.md`, `task1_comparison.csv`, `task1_gate_a.csv`, `task1_deltas_vs_C1.csv`, `task1_per_piece.csv` |

---

## 3. Rencana Task 2: perbandingan empat konfigurasi

### 3.1 Definisi konfigurasi

| Konfigurasi | Encoder | Fusion | Data | Training | Fokus |
|---|---|---|---|---|---|
| **T2-C1** | Fine-tuning | Full 128-d | Baseline 4 varian | Encoder + head | Baseline |
| **T2-C2** | Frozen | Full 128-d | Baseline 4 varian | Head saja | Pengaruh fine-tuning |
| **T2-C3** | Fine-tuning | Basic 64-d | Baseline 4 varian | Encoder + head | Pengaruh fusion |
| **T2-C4** | Fine-tuning | Full 128-d | Pitch-shift terkontrol | Encoder + head | Generalisasi pitch |

### 3.2 Status dan mapping

| Konfigurasi | Status | Artefak |
|---|---|---|
| T2-C1 | Sudah ada | `T2-baseline_seed42` (dipakai ulang) |
| T2-C2 | Akan dibuat | `T2-C2_seed42` |
| T2-C3 | Akan dibuat | `T2-C3_seed42` |
| T2-C4 | Akan dibuat | `T2-C4_seed42` |

T2-C1 memakai checkpoint dan metrik `T2-baseline_seed42`; tidak perlu retrain. Jika diperlukan nama seragam, buat catatan mapping, bukan menduplikasi checkpoint.

### 3.3 Hyperparameter bersama

| Parameter | Nilai |
|---|---|
| Backbone | Checkpoint `T1-C1_seed42` |
| Embedding | `32` per encoder |
| Optimizer | Adam |
| Encoder LR | `0.0003` (tidak dipakai di C2) |
| Head LR | `0.001` |
| Weight decay | `0.00001` |
| Batch size | `256` |
| Eval batch size | `512` |
| Maximum epoch | `8` |
| Early stopping | `3` |
| Dropout | `0.2` |
| Hidden dimension | `128` |
| Loss | Cross-entropy |
| Seed | `42` |
| Scheduler | `ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-6)` |
| Num workers | `0` |

Catatan C2: encoder LR tidak digunakan karena encoder frozen. Hidden dimension head tetap `128`; C3 juga memakai hidden `128`.

### 3.4 Detail implementasi per konfigurasi

#### T2-C1 - Fine-tuned Full Fusion

- Kedua encoder dilatih ulang.
- Fusion `[x, y, x*y, abs(x-y)]` berukuran `128-d`.
- Data baseline 4 varian: `clean`, `wrong_pitch`, `delete`, `insert`.
- Tanpa augmentasi.
- Hasil existing: best epoch 5, clean test macro-F1 `0.9929`, best val macro-F1 `0.9793`.

#### T2-C2 - Frozen Encoder

- Load encoder T1-C1.
- Bekukan `sheet` dan `audio` encoder (`requires_grad=False`).
- Latih hanya classification head.
- Optimizer hanya parameter head; `lr_head=0.001`.
- Encoder harus tetap dalam mode `eval()` saat training agar BatchNorm running stats tidak berubah.
- Fusion tetap full `128-d`.
- Data baseline 4 varian, tanpa augmentasi.

#### T2-C3 - Basic Fusion

- Kedua encoder fine-tuning.
- Fusion dasar: `concat([x, y])` berukuran `64-d`.
- Head: `64 -> 128 -> 4`, dropout `0.2`.
- Data baseline 4 varian, tanpa augmentasi.
- Tujuan: menguji apakah fitur interaksi dan selisih pada full fusion benar-benar memberi peningkatan.

#### T2-C4 - Pitch-Shift Robustness

- Kedua encoder fine-tuning.
- Fusion full `128-d`.
- Filter kelas `wrong_pitch` berdasarkan `pitch_delta`.
- Kelas `clean`, `delete`, dan `insert` tetap dipakai.
- Tujuan: menguji kemampuan model mengenali kesalahan pitch pada interval yang tidak pernah dilihat saat training.

### 3.5 Usulan split pitch T2-C4

| Split | `wrong_pitch` interval | Kelas lain |
|---|---|---|
| Train | `abs(pitch_delta) <= 6` | Semua |
| Val | `abs(pitch_delta) in {7, 8}` | Semua |
| Test | `abs(pitch_delta) in {9, 10, 11, 12}` | Semua |

Default ini memberi test interval yang tidak terlihat saat training. Jumlah `wrong_pitch` pada test lebih kecil dari kelas lain; evaluasi harus menampilkan per-class precision/recall/F1, bukan hanya macro-F1. Opsi subset test seimbang dapat ditambahkan jika diperlukan.

### 3.6 Evaluasi dan metrik

Evaluasi utama memakai test set yang sama untuk keempat konfigurasi:

- Test set T2-C4: `clean`, `delete`, `insert`, dan `wrong_pitch` dengan interval test.
- T2-C1, C2, C3 juga dievaluasi pada test set ini agar perbandingan apple-to-apple.
- Evaluasi tambahan pada full test set boleh dilaporkan untuk C1-C3.

Metrik yang dilaporkan:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Per-class Precision, Recall, F1
- Confusion Matrix

Prediksi dan confusion matrix disimpan per konfigurasi. Corruption grid tidak menjadi fokus perbandingan ini; jika dijalankan, statusnya evaluasi tambahan.

### 3.7 Artefak yang akan dibuat

| Item | Path |
|---|---|
| T2-C2 | `reports/experiments/T2-C2_seed42/`, `checkpoints/T2-C2_seed42/best.pt` |
| T2-C3 | `reports/experiments/T2-C3_seed42/`, `checkpoints/T2-C3_seed42/best.pt` |
| T2-C4 | `reports/experiments/T2-C4_seed42/`, `checkpoints/T2-C4_seed42/best.pt` |
| Perbandingan | `reports/experiments/task2_config_comparison.md`, `task2_config_comparison.csv` |
| Confusion matrix | `reports/experiments/task2_config_comparison_figures/` |
| Config | `reports/experiments/T2-C*_seed42/config.json` |

### 3.8 Urutan eksekusi

1. Tambahkan dukungan `fusion=full|basic` dan `freeze_encoders` pada model/training.
2. Pastikan encoder frozen tetap `eval()` dan optimizer hanya head untuk C2.
3. Tambahkan filter pitch interval untuk C4.
4. Buat evaluator yang bisa mengevaluasi checkpoint pada full test dan pitch test.
5. Jalankan C2, C3, C4 dengan seed 42.
6. Reuse C1 dari `T2-baseline_seed42`.
7. Gabungkan metrik, per-class scores, dan confusion matrix ke laporan perbandingan.
8. Perbarui `FULL_CONFIG_TASK1_TASK2.md` dengan status dan hasil akhir.

Estimasi compute:

| Konfigurasi | Estimasi |
|---|---|
| T2-C2 | sekitar 45 menit |
| T2-C3 | sekitar 1,5 jam |
| T2-C4 | sekitar 1,5 jam |
| Total | sekitar 4 jam |

### 3.9 Acceptance criteria

- C2, C3, C4 selesai dengan seed 42 dan config tersimpan.
- C2 benar-benar frozen: tidak ada gradient encoder dan BatchNorm tidak berubah.
- C3 memakai basic fusion `64-d` dengan hidden `128`.
- C4 tidak memakai interval test di training.
- Keempat konfigurasi dievaluasi pada test set yang sama.
- Tabel perbandingan memuat Accuracy, Precision, Recall, Macro-F1, dan confusion matrix.
- Tidak ada perubahan pada hasil Task 1.

### 3.10 Keputusan yang masih terbuka

1. **Split pitch T2-C4**: setuju dengan `train |delta| <= 6`, `val 7-8`, `test 9-12`? Atau interval lain?
2. **Test set T2-C4**: pakai seluruh kelas lain + `wrong_pitch` interval test, atau buat subset seimbang per kelas?
3. **T2-C1**: pakai checkpoint `T2-baseline_seed42` yang sudah ada, atau retrain dengan nama `T2-C1_seed42`?

---

## 4. Catatan dan risiko

- Arm `T2-robust`, `T2-robust-strong`, `T2-audio-only`, dan `T2-cross-soundfont` adalah eksperimen tambahan dan tidak termasuk perbandingan empat konfigurasi ini.
- C2 tidak valid jika BatchNorm encoder ikut ter-update; karena itu encoder wajib `eval()` saat training.
- C4 memiliki distribusi kelas tidak seimbang pada test interval; laporkan per-class metrics dan confusion matrix.
- C1, C2, dan C3 sudah melihat interval test C4 selama training; perbandingan menunjukkan generalization gap, bukan performa absolut pada interval baru.
- Semua hasil masih memakai satu seed (`42`); variansi antar-seed belum diukur.
- Task 2 memakai label error sintetis dan test hanya 3 piece; angka bukan performa pada permainan nyata.
