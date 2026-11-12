# Rencana Eksperimen — 4 Konfigurasi

Dokumen ini merinci eksperimen sistematis minimal 4 konfigurasi untuk Task 1,
dengan opsi 4 konfigurasi tambahan untuk Task 2. Tujuannya memenuhi rubrik
eksperimen (minimal 3 model/konfigurasi) sekaligus menunjukkan usaha peningkatan
model yang terkontrol.

## 1. Tujuan dan pertanyaan penelitian

1. Apakah tuning training + regularisasi meningkatkan Task 1 dibanding baseline?
2. Apakah arsitektur lebih besar (4 blok) + CCA meningkatkan performa?
3. Apakah learned ranking head dengan hard negatives meningkatkan akurasi
   5-kandidat?
4. (Opsional) Apakah regularisasi, unfreeze encoder, atau audio-only CNN
   mengubah performa/robustness Task 2?

## 2. Data dan split

- Split tetap dari artefak: train 17 piece, val 4 piece, test 3 piece.
- Task 1: `score_crops`, `audio_windows`, `pairs_note` untuk training retrieval;
  `pairs_system`, `notehead_systems`, `systems` untuk evaluasi/ranking.
- Task 2: `error_windows`, `task2_windows`, `noteheads` untuk referensi.
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

Urutan ini membuat perbandingan terkontrol: C1 ke C2 mengisolasi training recipe,
C2 ke C3 mengisolasi arsitektur, C3 ke C4 mengisolasi metode ranking.

## 5. Detail konfigurasi

### C1 — Baseline

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

### C2 — Tuning + regularisasi

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

### C3 — Arsitektur 4 blok + CCA

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

### C4 — Learned ranking head

```yaml
experiment: t1_c4_ranking_head
source_checkpoint: checkpoints/t1_c3_architecture.pt
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

## 6. Protokol

1. Restore artefak dan jalankan `run_preprocessing.py --verify-only`.
2. Smoke test `fast_dev` untuk Task 1 dan Task 2.
3. Jalankan C1, C2, C3, C4 secara berurutan.
4. Pilih checkpoint berdasarkan val MRR (Task 1) atau val macro-F1 (Task 2).
5. Evaluasi test sekali untuk konfigurasi terpilih.
6. Ulangi 3 seed (42/123/2024) untuk minimal dua konfigurasi terbaik; laporkan
   mean dan std.
7. Simpan semua artefak per run.

## 7. Output per run

```
reports/experiments/<run_id>/
  config.json
  history.csv
  metrics.json
  predictions.csv
  figures/
    learning_curves.png
    retrieval_ranks.png
    confusion.png
```

`metrics.json` minimal memuat: task, config, seed, device, checkpoint, epoch
terbaik, val metrics, test metrics, dan durasi training.

## 8. Analisis dan laporan

- Tabel perbandingan C1-C4: perubahan, val metric, test metric, waktu, seed.
- Kurva learning tiap konfigurasi.
- Bar chart perbandingan metrik utama.
- Distribusi rank dan contoh kegagalan Task 1.
- Analisis error Task 2 per soundfont, `pitch_delta`, dan durasi.
- Batasan: pilot 25 piece, test 3 piece, label Task 2 sintetis, CPU-only.

## 9. Estimasi waktu (CPU, 1 seed)

| Run | Estimasi |
|---|---|
| C1 | 19 menit (sudah ada) |
| C2 | 35-40 menit |
| C3 | 60-80 menit |
| C4 | 10-15 menit |
| **Total baru** | **+/- 2 jam** |

## 10. Opsional — 4 konfigurasi Task 2

| ID | Konfigurasi | Faktor yang diubah |
|---|---|---|
| T2-C1 | Baseline fusion, sheet frozen, 8 epoch | kontrol |
| T2-C2 | + dropout 0.3, label smoothing 0.05, wd 1e-4, early stop | regularisasi |
| T2-C3 | + unfreeze sheet encoder (lr 2e-4) | transfer learning |
| T2-C4 | Audio-only CNN tanpa referensi partitur | ablasi metode |

Metrik: macro-F1, per-class F1, confusion, ECE, per-soundfont F1, gap val ke test.

## 11. Risiko dan guardrail

- Test Task 1 hanya 3 piece: jangan menyimpulkan generalisasi ke seluruh MSMD.
- Task 2 mendekati saturasi: peningkatan dinilai dari val, kalibrasi, dan
  robustness, bukan hanya test accuracy.
- CCA dapat tidak stabil; siapkan fallback.
- Jangan tuning di test.
- Jangan membandingkan hasil dengan angka paper secara langsung; baseline ini
  Dorfer-lite dengan 17 train piece.

## 12. Checklist eksekusi

- [ ] Artefak `.npy` tersedia dan verifikasi `ok: true`.
- [ ] Runner `scripts/run_experiment.py` tersedia.
- [ ] Config C1-C4 tersedia di `configs/experiments/`.
- [ ] Smoke test lolos.
- [ ] C1-C4 selesai dan metrics tersimpan.
- [ ] Tabel perbandingan dan figure dibuat.
- [ ] Laporan batasan ditulis.
