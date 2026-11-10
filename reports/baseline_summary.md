# Baseline Run Summary — Dorfer-lite

Tanggal run: 2026-11-09  
Environment: Python 3.11, PyTorch 2.15.0+cpu, 16 CPU threads  
Hardware: Intel Core Ultra 7 155H, 32 GB RAM, CPU-only

## Konfigurasi

| | Task 1 | Task 2 |
|---|---|---|
| Model | Dorfer-lite dual encoder | Dorfer-lite encoders + fusion classifier |
| Channels | 16/32/64 (3 blok) | 16/32/64 |
| CCA | tidak | tidak |
| Batch | 128 | 256 |
| Epoch | 6 | 8 |
| LR | Adam 2e-3 | encoder 5e-4, head 1e-3 |
| Sheet encoder | dilatih | dibekukan |
| Data/epoch | 28.730 pairs | ±28.730 (1 varian/source) |

## Waktu training

| Tahap | Waktu |
|---|---:|
| Task 1 training (6 epoch) | ±19,3 menit |
| Task 1 evaluasi | ±2–3 menit |
| Task 2 training (8 epoch) | ±9,4 menit |
| Task 2 evaluasi | ±2–3 menit |
| **Total** | **±33–35 menit** |

## Task 1 — Score–Audio Alignment

Best checkpoint: epoch 6, validation MRR `0.00864`.

### Snippet retrieval

| Protokol | R@1 | R@5 | R@10 | R@25 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Paired 2018 A→S | 0.0015 | 0.0084 | 0.0134 | 0.0307 | 0.0087 | 573.5 |
| Paired 2018 S→A | 0.0010 | 0.0050 | 0.0164 | 0.0391 | 0.0092 | 526.0 |
| Extended A→S | 0.0020 | 0.0097 | 0.0171 | 0.0391 | 0.0105 | 515.0 |
| Extended S→A | 0.0030 | 0.0050 | 0.0154 | 0.0352 | 0.0090 | 826.5 |
| Random paired A→S | 0.0005 | 0.0025 | 0.0055 | 0.0178 | 0.0044 | 1018.5 |

Model hanya sekitar 3× random pada R@1/MRR. Exact snippet retrieval masih sangat lemah.

### System localization

| Protokol | R@1 | R@5 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Page-conditioned | 0.2750 | 0.9446 | 0.5152 | 3.0 |
| Full collection (36 system) | 0.0762 | 0.2540 | 0.1876 | 13.0 |
| 5-candidate | 0.2556 | 1.0000 | 0.5191 | 2.0 |

### Identification

| Protokol | Rk@1 | Rk@5 |
|---|---:|---:|
| Piece A→S | 54.2% | 100% |
| Piece S→A | 50.5% | 100% |
| Page A→S | 33.7% | 81.0% |
| Page S→A | 26.8% | 77.2% |

**Interpretasi Task 1:** model belajar lokalisasi kasar (page-conditioned R@5 0,94), tetapi belum belajar pencocokan snippet level-note yang kuat. Penyebab utama: hanya 17 train pieces, model lite 148k parameter, dan 6 epoch. Baseline ini belum bisa dibandingkan langsung dengan angka paper (paper memakai 360 train pieces dan model jauh lebih besar).

## Task 2 — Performance Error Detection

Best checkpoint: epoch 8, validation macro-F1 `0.9643`.

### Test metrics

| Metrik | Nilai |
|---|---:|
| 4-class accuracy | **0.9900** |
| 4-class macro-F1 | **0.9900** |
| Top-2 accuracy | **1.0000** |
| Binary correct-vs-error accuracy | 0.9902 |
| Binary precision / recall / F1 | 0.9932 / 0.9938 / 0.9935 |
| Error-type accuracy among detected errors | 0.9997 |
| Coverage of true errors | 0.9938 |

### Confusion matrix (test)

| true \ pred | correct | wrong_pitch | missing_note | extra_note |
|---|---:|---:|---:|---:|
| correct | 7.907 | 0 | 0 | 165 |
| wrong_pitch | 0 | 8.065 | 7 | 0 |
| missing_note | 0 | 0 | 8.072 | 0 |
| extra_note | 151 | 0 | 0 | 7.921 |

Kesalahan tersisa hampir seluruhnya adalah **correct vs extra_note**.

**Interpretasi Task 2:** model sangat kuat membedakan correct, wrong pitch, missing note, dan extra note. Namun label Task 2 adalah **error sintetis**, bukan anotasi human, dan test hanya 3 piece. Angka 0,99 tidak boleh diklaim sebagai performa pada permainan piano nyata.

## Artefak

- `checkpoints/baseline_task1_dual_encoder.pt` — model Task 1 final (v2, 6 epoch)
- `checkpoints/baseline_task2_multimodal_classifier.pt` — model Task 2 final
- `checkpoints/baseline_task1_test_embeddings.npz` — embedding test Task 1
- `checkpoints/task2_score_bank_{split}.pt` — cache embedding score Task 2
- `reports/baseline_task1_metrics.json`
- `reports/baseline_task2_metrics.json`
- `reports/baseline_task1_history.csv`
- `reports/baseline_task2_history.csv`
- `reports/baseline_task2_error_analysis.csv`
- `reports/baseline_task2_test_predictions.csv`
- `reports/figures/20_*` sampai `28_*`

Backup run sebelumnya:
- `checkpoints/baseline_task1_dual_encoder_v1.pt`
- `checkpoints/baseline_task1_test_embeddings_v1.npz`
- `reports/baseline_task1_metrics_v1.json`
- `reports/baseline_task1_history_v1.csv`

## Catatan teknis

- Task 1 dijalankan 2 kali. Run v1 (8 epoch, tanpa tempo jitter) dan v2 (6 epoch, weight decay 1e-4, tempo jitter 10%). v2 dipakai sebagai final.
- Random baseline pada Task 1 sudah dikoreksi: versi lama hanya mempermutasi kolom kandidat sehingga rank tidak berubah. JSON sudah memuat random baseline yang benar (R@1 0.0005, MRR 0.0044).
- Task 1 snippet retrieval lemah; system localization masih di atas random.
- Task 2 sangat tinggi, tetapi memakai error sintetis dan split pilot.
- Notebook 03 dan 04 sudah dieksekusi penuh; output dan figure tersimpan.
