# Baseline Run Summary — Dorfer (full)

Tanggal run: 2026-11-17
Environment: Python 3.11, PyTorch 2.15.0+cpu, 16 CPU threads
Hardware: Intel Core Ultra 7 155H, 32 GB RAM, CPU-only

## Konfigurasi

| | Task 1 | Task 2 |
|---|---|---|
| Model | Dorfer dual encoder | Dorfer encoders + fusion classifier |
| Channels | 24/48/96/96 (4 blok) | 24/48/96/96 |
| CCA | ya, running covariance `alpha=0.05` | tidak (trunk 32-d) |
| Batch | 100 | 256 |
| Epoch | 8 max (early stop 3) | 8 max (early stop 3) |
| LR | Adam 2e-3 | encoder 3e-4, head 1e-3 |
| Sheet encoder | dilatih | **dilatih** |
| Audio encoder | dilatih | **dilatih** |
| Data/epoch | 28.730 pairs | ±28.730 (1 varian/source) |

## Waktu training

| Tahap | Waktu |
|---|---:|
| Task 1 training (6 epoch berjalan, best epoch 3) | ±42 menit |
| Task 1 evaluasi | ±3 menit |
| Task 2 training (8 epoch, best epoch 8) | ±81 menit |
| Task 2 evaluasi | ±3 menit |
| **Total** | **±2 jam 10 menit** |

## Task 1 — Score–Audio Alignment

Best checkpoint: epoch 3, validation MRR `0.0100` (random ≈ `0.0032`).

### Snippet retrieval

| Protokol | R@1 | R@5 | R@10 | R@25 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Paired 2018 A→S | 0.0010 | 0.0035 | 0.0099 | 0.0342 | 0.0075 | 610.5 |
| Paired 2018 S→A | 0.0030 | 0.0114 | 0.0178 | 0.0391 | 0.0117 | 579.0 |
| Extended A→S | 0.0017 | 0.0077 | 0.0157 | 0.0453 | 0.0104 | 519.0 |
| Extended S→A | 0.0035 | 0.0099 | 0.0213 | 0.0337 | 0.0100 | 1161.0 |
| Random paired A→S | 0.0005 | 0.0020 | 0.0035 | 0.0099 | 0.0037 | 1026.0 |

Snippet MRR hanya sekitar 2–3× random. Exact snippet retrieval masih lemah.

### System localization

| Protokol | R@1 | R@5 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Page-conditioned | 0.2595 | 0.9348 | 0.5043 | 3.0 |
| Full collection (36 system) | 0.0693 | 0.2919 | 0.1950 | 11.0 |
| 5-candidate | 0.2733 | 1.0000 | 0.5298 | 2.0 |

### Identification

| Protokol | Rk@1 | Rk@5 |
|---|---:|---:|
| Piece A→S | 41.2% | 100% |
| Piece S→A | 46.3% | 100% |
| Page A→S | 30.5% | 80.1% |
| Page S→A | 25.7% | 85.9% |

**Interpretasi Task 1:** model belajar lokalisasi kasar, tetapi belum belajar pencocokan snippet level-note yang kuat. Penyebab utama: hanya 17 train pieces. Best epoch 3 lalu val MRR turun, pola overfitting. Baseline ini faithful secara arsitektur, tetapi belum sebanding dengan angka paper yang memakai 360 train pieces dan training jauh lebih panjang.

## Task 2 — Performance Error Detection

Best checkpoint: epoch 8, validation macro-F1 `0.9755`.

### Test metrics

| Metrik | Nilai |
|---|---:|
| 4-class accuracy | **0.9954** |
| 4-class macro-F1 | **0.9954** |
| Top-2 accuracy | 1.0000 |
| Binary correct-vs-error accuracy | 0.9954 |
| Binary precision / recall / F1 | 0.9971 / 0.9968 / 0.9970 |
| Error-type accuracy among detected errors | 1.0000 |
| Coverage of true errors | 0.9968 |

### Confusion matrix (test)

| true \ pred | correct | wrong_pitch | missing_note | extra_note |
|---|---:|---:|---:|---:|
| correct | 8.003 | 0 | 0 | 69 |
| wrong_pitch | 0 | 8.072 | 0 | 0 |
| missing_note | 0 | 0 | 8.072 | 0 |
| extra_note | 78 | 0 | 0 | 7.994 |

Kesalahan tersisa seluruhnya adalah **correct vs extra_note**. Pola ini sama dengan ablation reduced-capacity, jadi bukan efek kapasitas model.

**Interpretasi Task 2:** model kuat membedakan empat kelas. Namun label adalah **error sintetis**, bukan anotasi human, dan test hanya 3 piece. Angka 0,995 tidak boleh diklaim sebagai performa pada permainan piano nyata.

## Artefak

- `checkpoints/baseline_task1_dual_encoder.pt` — model Task 1 final
- `checkpoints/baseline_task2_multimodal_classifier.pt` — model Task 2 final
- `checkpoints/baseline_task1_test_embeddings.npz` — embedding test Task 1
- `reports/baseline_task1_metrics.json`
- `reports/baseline_task2_metrics.json`
- `reports/baseline_task1_history.csv`
- `reports/baseline_task2_history.csv`
- `reports/baseline_task2_error_analysis.csv`
- `reports/baseline_task2_test_predictions.csv`
- `reports/figures/20_*` sampai `28_*`

## Catatan teknis

- Arsitektur mengikuti paper: 4 blok 24/48/96/96, CCA 32-d, margin 0.7, batch 100.
- CCA memakai running covariance `S11/S12/S22` dengan `alpha = 0.05` dan warmup 200 step. Dengan `alpha = 1.0` (nilai paper), CCA menghafal statistik batch pada data pilot dan training tidak menghasilkan representasi yang tergeneralisasi.
- Task 2 melatih kedua encoder; tidak ada komponen yang dibekukan.
- Task 2 tidak memakai CCA karena klasifikasi tidak punya pasangan lintas modalitas; classifier memakai output trunk 32-d.
- Random baseline Task 1 dihitung dengan mempermutasi query sehingga pasangan benar-benar acak.
