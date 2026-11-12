# Rencana Eksperimen - 5 Konfigurasi per Task

Dokumen ini merinci 5 konfigurasi Task 1 dan 5 konfigurasi Task 2. Seluruh
konfigurasi Task 2 bersifat multimodal (score reference + audio); tidak ada
ablasi audio-only. Tujuannya memenuhi rubrik eksperimen sekaligus menunjukkan
usaha peningkatan model yang terkontrol.

## 1. Tujuan dan pertanyaan penelitian

1. Apakah tuning training + regularisasi meningkatkan Task 1 dibanding baseline?
2. Apakah arsitektur lebih besar (4 blok) + CCA meningkatkan performa?
3. Apakah learned ranking head dengan hard negatives meningkatkan akurasi
   5-kandidat?
4. Apakah kapasitas embedding (64-d) meningkatkan Task 1 dibanding C2?
5. Apakah regularisasi, unfreeze encoder, backbone lebih besar, atau gated
   fusion mengubah performa/kalibrasi Task 2?

## 2. Data dan split

- Split tetap dari artefak: train 17 piece, val 4 piece, test 3 piece.
- Task 1: `score_crops`, `audio_windows`, `pairs_note` untuk training retrieval;
  `pairs_system`, `notehead_systems`, `systems` untuk evaluasi/ranking.
- Task 2: `error_windows`, `task2_windows`, `noteheads` untuk referensi skor.
- Jangan split ulang. Jangan memakai test untuk tuning.

## 3. Metrik

### Task 1

Utama:
- Page-conditioned system R@1, R@5, MRR.
- 5-kandidat accuracy/MRR dari `pairs_system`.

Sekunder:
- Snippet retrieval R@1/5/10/25, MRR, median rank.
- Piece/page identification Rk@1/Rk@5.

### Task 2

- Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
- Top-2 accuracy, binary correct-vs-error.
- ECE (expected calibration error).
- Per-soundfont F1 dan gap val ke test.

## 4. Matriks konfigurasi Task 1

| ID | Konfigurasi | Faktor yang diubah | Tujuan |
|---|---|---|---|
| C1 | Dorfer-lite 3 blok 16/32/64, no CCA, 6 epoch | kontrol | pembanding |
| C2 | Arsitektur sama, 12 epoch, lr 1e-3, wd 1e-3, augmentasi lebih kuat | tuning + regularisasi | uji training recipe |
| C3 | 4 blok 24/48/96/96 + CCA on, jadwal C2 | arsitektur | uji kapasitas + CCA |
| C4 | Embedding C3 + learned ranking head listwise | loss/metode | uji learned ranking |
| C5 | Resep C2 + `embed_dim: 64` | kapasitas embedding | uji dimensi embedding |

Urutan ini membuat perbandingan terkontrol:

- C1 -> C2 mengisolasi training recipe.
- C2 -> C3 mengisolasi arsitektur + CCA.
- C3 -> C4 mengisolasi metode ranking.
- C2 -> C5 mengisolasi kapasitas embedding (cabang, bukan kelanjutan linear).

## 5. Detail konfigurasi Task 1

### C1 - Baseline

```yaml
profile: dorfer-lite
seed: 42
batch_size: 128
eval_batch_size: 256
epochs: 6
early_stop_patience: 3
lr: 2.0e-3
weight_decay: 1.0e-4
grad_clip: 5.0
margin: 0.7
embed_dim: 32
channels: [16, 32, 64]
use_cca: false
augment:
  image_scale: [0.95, 1.05]
  image_translate: 5.0
  audio_time_shift: 1
  audio_tempo_jitter: 0.1
num_workers: 0
fast_dev: false
```

### C2 - Tuning + regularisasi

```yaml
profile: dorfer-lite
seed: 42
batch_size: 128
eval_batch_size: 256
epochs: 12
early_stop_patience: 3
lr: 1.0e-3
weight_decay: 1.0e-3
grad_clip: 5.0
margin: 0.7
embed_dim: 32
channels: [16, 32, 64]
use_cca: false
augment:
  image_scale: [0.9, 1.1]
  image_translate: 5.0
  audio_time_shift: 1
  audio_tempo_jitter: 0.1
num_workers: 0
fast_dev: false
```

### C3 - Arsitektur 4 blok + CCA

```yaml
profile: dorfer-paper-size
seed: 42
batch_size: 128
eval_batch_size: 256
epochs: 12
early_stop_patience: 3
lr: 1.0e-3
weight_decay: 1.0e-3
grad_clip: 5.0
margin: 0.7
embed_dim: 32
channels: [24, 48, 96, 96]
use_cca: true
cca:
  r1: 1.0e-3
  r2: 1.0e-3
  rT: 1.0e-3
  alpha: 1.0
augment:
  image_scale: [0.9, 1.1]
  image_translate: 5.0
  audio_time_shift: 1
  audio_tempo_jitter: 0.1
num_workers: 0
fast_dev: false
```

Catatan: jika CCA menghasilkan loss/skor NaN, fallback ke `use_cca: false` dan
catat sebagai temuan.

### C4 - Learned ranking head

```yaml
experiment: t1_c4_ranking_head
source_checkpoint: checkpoints/experiments/t1_c3_architecture.pt
freeze_encoders: true
features:
  - max_sim
  - mean_sim
  - top5_mean
  - min_sim
  - std_sim
  - pair_best_x
  - pair_best_y
  - pair_best_xy
  - pair_best_absdiff
loss: listwise_softmax
num_candidates: 5
hidden_dim: 128
dropout: 0.2
epochs: 10
lr: 1.0e-3
weight_decay: 1.0e-3
batch_queries: 64
hard_negative_mining:
  enabled: true
  every_epochs: 2
```

Detail C4:
- Sorot kandidat system dari `pairs_system` (1 positif + 4 negatif per query).
- Untuk setiap kandidat, ambil embedding anggota notehead dari `notehead_systems`.
- Hitung fitur similarity, lalu latih MLP/listwise softmax 5-way.
- Hard negative mining: setiap 2 epoch, ganti negatif dengan system skor
  tertinggi yang salah (dari seluruh system pada halaman yang sama).
- Evaluasi: 5-kandidat accuracy/MRR, page-conditioned system R@1/R@5.

### C5 - Kapasitas embedding 64-d

```yaml
run_id: t1_c5_capacity
task: task1
experiment: dual_encoder
profile: dorfer-lite
seed: 42
batch_size: 128
eval_batch_size: 256
epochs: 12
early_stop_patience: 3
lr: 1.0e-3
weight_decay: 1.0e-3
grad_clip: 5.0
margin: 0.7
embed_dim: 64
channels: [16, 32, 64]
use_cca: false
augment:
  image_scale: [0.9, 1.1]
  image_translate: 5.0
  audio_time_shift: 1
  audio_tempo_jitter: 0.1
num_workers: 0
fast_dev: false
```

## 6. Protokol

1. Restore artefak dan jalankan `run_preprocessing.py --verify-only`.
2. Smoke test `fast_dev` untuk seluruh konfigurasi Task 1 dan Task 2.
3. Jalankan C1-C5 secara berurutan.
4. Jalankan T2-C1 sampai T2-C5 secara berurutan.
5. Pilih checkpoint berdasarkan val MRR (Task 1) atau val macro-F1 (Task 2).
6. Evaluasi test sekali untuk konfigurasi terpilih.
7. Ulangi 3 seed (42/123/2024) untuk minimal dua konfigurasi terbaik per task;
   laporkan mean dan std.
8. Simpan semua artefak per run.

## 7. Output per run

```
reports/experiments/<run_id>/
  config.json
  history.csv
  metrics.json
  predictions.csv
  run.log
  figures/
    # Task 1
    learning_curves.png
    snippet_metrics.png
    system_metrics.png
    retrieval_ranks.png
    # Task 2
    learning_curves.png
    confusion.png
    per_class_f1.png
    calibration.png
```

`metrics.json` minimal memuat: task, config, seed, device, checkpoint, epoch
terbaik, val metrics, test metrics, dan durasi training. Untuk Task 2 tambahkan
ECE, per-soundfont F1, dan gap val ke test.

## 8. Analisis dan laporan

- Tabel perbandingan C1-C5: perubahan, val metric, test metric, waktu, seed.
- Tabel perbandingan T2-C1..T2-C5: macro-F1, per-class F1, ECE, per-soundfont
  F1, gap val ke test.
- Kurva learning tiap konfigurasi.
- Bar chart perbandingan metrik utama.
- Distribusi rank dan contoh kegagalan Task 1.
- Analisis error Task 2 per soundfont, `pitch_delta`, dan durasi.
- Batasan: pilot 25 piece, test 3 piece, label Task 2 sintetis, CPU-only.

## 9. Estimasi waktu (CPU, 1 seed)

### Task 1

| Run | Estimasi |
|---|---|
| C1 | 31,7 menit (sudah ada) |
| C2 | 37,8 menit (sudah ada) |
| C3 | 67,8 menit (sudah ada) |
| C4 | 4,3 menit (sudah ada) |
| C5 | 35-45 menit |
| **Total baru** | **+/- 40 menit** |

### Task 2

| Run | Estimasi |
|---|---|
| T2-C1 baseline | 8-12 menit |
| T2-C2 regularized | 8-12 menit |
| T2-C3 unfrozen | 20-30 menit |
| T2-C4 strong encoder | 10-15 menit |
| T2-C5 gated fusion | 8-12 menit |
| **Total baru** | **+/- 1-1,5 jam** |

Seed finalis (123/2024) menambah waktu sesuai jumlah run finalis.

## 10. Task 2 - 5 konfigurasi multimodal

Semua konfigurasi memakai dua modalitas: score reference (crop notehead sumber)
dan audio window. Tidak ada konfigurasi audio-only.

### 10.1 Default bersama

```yaml
task: task2
experiment: multimodal_classifier
seed: 42
device: auto
embed_dim: 32
hidden_dim: 128
dropout: 0.2
num_classes: 4
batch_size: 256
eval_batch_size: 512
epochs: 8
early_stop_patience: 3
lr_encoder: 5.0e-4
lr_head: 1.0e-3
weight_decay: 1.0e-5
grad_clip: 5.0
freeze_sheet: true
train_one_variant_per_source: true
label_smoothing: 0.0
fusion: concat
num_workers: 0
fast_dev: false
```

Catatan sumber checkpoint: `checkpoints/baseline_task1_dual_encoder.pt` tidak ada
di repo saat ini. Runner memakai checkpoint eksperimen Task 1:
`checkpoints/experiments/<run_id>.pt`.

### 10.2 T2-C1 - Baseline multimodal

```yaml
run_id: t2_c1_baseline
source_checkpoint: checkpoints/experiments/t1_c1_baseline.pt
channels: [16, 32, 64]
# sisanya mengikuti default bersama
```

- Kontrol: fusion concat, sheet frozen, 1 varian per source per epoch.
- Reproduksi baseline notebook 04 dengan checkpoint Task 1 dari `reports/experiments`.

### 10.3 T2-C2 - Regularisasi

```yaml
run_id: t2_c2_regularized
source_checkpoint: checkpoints/experiments/t1_c1_baseline.pt
channels: [16, 32, 64]
dropout: 0.3
label_smoothing: 0.05
weight_decay: 1.0e-4
```

- Faktor: regularisasi (dropout + label smoothing + weight decay).
- Sisanya sama dengan T2-C1.

### 10.4 T2-C3 - Unfreeze sheet encoder

```yaml
run_id: t2_c3_unfrozen
source_checkpoint: checkpoints/experiments/t1_c1_baseline.pt
channels: [16, 32, 64]
freeze_sheet: false
lr_encoder: 2.0e-4
```

- Faktor: fine-tune kedua encoder (transfer learning).
- Score bank tidak dipakai; crop skor dimuat per batch.
- Sisanya sama dengan T2-C1.

### 10.5 T2-C4 - Backbone lebih besar

```yaml
run_id: t2_c4_strong_encoder
source_checkpoint: checkpoints/experiments/t1_c3_architecture.pt
channels: [24, 48, 96, 96]
```

- Faktor: kapasitas backbone (4 blok dari C3).
- `freeze_sheet: true`; encoder C3 dipakai sebagai raw encoder.
- Proyeksi CCA tidak dipakai di head Task 2; `FusionClassifier` menormalisasi
  kedua embedding. Ini disengaja agar perbandingan backbone terkontrol.
- Sisanya sama dengan T2-C1.

### 10.6 T2-C5 - Gated fusion

```yaml
run_id: t2_c5_gated_fusion
source_checkpoint: checkpoints/experiments/t1_c1_baseline.pt
channels: [16, 32, 64]
fusion: gated
```

Arsitektur gated fusion:

```
x = normalize(sheet_emb)
y = normalize(audio_emb)
pair = concat([x, y, x*y, |x-y|])          # 4 * embed_dim
gate = sigmoid(Linear(4 * embed_dim -> 4 * embed_dim))
gated = gate * pair
h = ReLU(Linear(4 * embed_dim -> hidden_dim)(gated))
logits = Linear(hidden_dim -> num_classes)(Dropout(h))
```

- Faktor: mekanisme fusion (gated vs concat).
- `Task2Model(..., fusion="concat" | "gated")`; default `concat` agar checkpoint
  baseline tetap kompatibel.
- Sisanya sama dengan T2-C1.

## 11. Risiko dan guardrail

- Test Task 1 hanya 3 piece: jangan menyimpulkan generalisasi ke seluruh MSMD.
- Task 2 mendekati saturasi: peningkatan dinilai dari val, kalibrasi, dan
  robustness, bukan hanya test accuracy.
- CCA dapat tidak stabil; siapkan fallback `use_cca: false`.
- T2-C4 memakai raw encoder C3 tanpa proyeksi CCA; catat sebagai batasan
  perbandingan.
- Gated fusion bekerja pada embedding 32-d; jika tidak stabil, fallback ke
  `fusion: concat` dan catat.
- Jangan tuning di test.
- Jangan membandingkan hasil dengan angka paper secara langsung; baseline ini
  Dorfer-lite dengan 17 train piece.

## 12. Checklist eksekusi

- [x] Artefak `.npy` tersedia dan verifikasi `ok: true`.
- [x] Runner Task 1 `scripts/run_experiment.py` tersedia.
- [ ] Runner Task 2 di `scripts/run_experiment.py` (cabang `task: task2`).
- [x] Config Task 1 C1-C4 tersedia di `configs/experiments/`.
- [ ] Config Task 1 C5 (`t1_c5_capacity.yaml`).
- [ ] Config Task 2 T2-C1..T2-C5 di `configs/experiments/`.
- [x] Smoke test Task 1 lolos.
- [ ] Smoke test Task 2 lolos.
- [x] C1-C4 selesai dan metrics tersimpan.
- [ ] C5 selesai dan metrics tersimpan.
- [ ] T2-C1..T2-C5 selesai dan metrics tersimpan.
- [x] Tabel perbandingan dan figure Task 1 C1-C4.
- [ ] Tabel perbandingan C1-C5 dan T2-C1..T2-C5.
- [x] Laporan batasan baseline (`reports/experiments/comparison/analysis.md`).
