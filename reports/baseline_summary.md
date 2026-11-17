# Baseline Run Summary — Dorfer (full)

Run: 2026-11-17 · Python 3.11 · PyTorch 2.15.0+cpu · 16 thread · CPU-only

## Konfigurasi

| | Task 1 | Task 2 |
|---|---|---|
| Model | Dorfer dual encoder | Dorfer encoders + fusion classifier |
| Channels | 24/48/96/96 (4 blok) | 24/48/96/96 |
| CCA | ya, running covariance `alpha=0.05` | tidak (trunk 32-d) |
| Batch | 100 | 256 |
| Epoch | 8 max, early stop 3 | 8 max, early stop 3 |
| LR | Adam 2e-3 | encoder 3e-4, head 1e-3 |
| Sheet encoder | dilatih | dilatih |
| Audio encoder | dilatih | dilatih |

## Waktu

| Tahap | Waktu |
|---|---:|
| Task 1 training (6 epoch, best epoch 3) | ±42 menit |
| Task 1 evaluasi | ±3 menit |
| Task 2 training (8 epoch, best epoch 8) | ±81 menit |
| Task 2 evaluasi | ±3 menit |
| **Total** | **±2 jam 10 menit** |

## Task 1 — Score–Audio Alignment

Best epoch 3, validation MRR `0.0100` (random ≈ `0.0032`).

| Protokol | R@1 | R@5 | MRR | Median rank |
|---|---:|---:|---:|---:|
| Paired A→S | 0.0010 | 0.0035 | 0.0075 | 610.5 |
| Paired S→A | 0.0030 | 0.0114 | 0.0117 | 579.0 |
| Extended A→S | 0.0017 | 0.0077 | 0.0104 | 519.0 |
| Extended S→A | 0.0035 | 0.0099 | 0.0100 | 1161.0 |
| Random paired A→S | 0.0005 | 0.0020 | 0.0037 | 1026.0 |

| System localization | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| Page-conditioned | 0.2595 | 0.9348 | 0.5043 |
| Full collection (36 system) | 0.0693 | 0.2919 | 0.1950 |
| 5-candidate | 0.2733 | 1.0000 | 0.5298 |

| Identification | Rk@1 | Rk@5 |
|---|---:|---:|
| Piece A→S | 41.2% | 100% |
| Piece S→A | 46.3% | 100% |
| Page A→S | 30.5% | 80.1% |
| Page S→A | 25.7% | 85.9% |

**Interpretasi:** snippet MRR hanya 2–3× random dan page-conditioned R@1 setara chance. Best epoch 3 lalu turun → overfitting pada 17 train pieces. Arsitektur faithful; skala data yang membatasi.

## Task 2 — Performance Error Detection

Best epoch 8, validation macro-F1 `0.9755`.

| Metrik | Nilai |
|---|---:|
| 4-class accuracy | **0.9954** |
| 4-class macro-F1 | **0.9954** |
| Top-2 accuracy | 1.0000 |
| Binary error F1 | 0.9970 |
| Error-type accuracy (detected) | 1.0000 |
| Coverage of true errors | 0.9968 |

| true \ pred | correct | wrong_pitch | missing_note | extra_note |
|---|---:|---:|---:|---:|
| correct | 8.003 | 0 | 0 | 69 |
| wrong_pitch | 0 | 8.072 | 0 | 0 |
| missing_note | 0 | 0 | 8.072 | 0 |
| extra_note | 78 | 0 | 0 | 7.994 |

**Interpretasi:** error tersisa hanya `correct` ↔ `extra_note`. Label Task 2 adalah error sintetis dan test hanya 3 piece; angka ini bukan performa pada permainan nyata.

## Catatan

- CCA memakai running covariance `S11/S12/S22`, `alpha=0.05`, warmup 200 step. Dengan `alpha=1.0` (nilai paper), CCA menghafal batch pada data pilot: loss 0.03 tetapi MRR di level random.
- Task 2 melatih kedua encoder; tidak ada komponen frozen dan tidak ada cache embedding.
- Random baseline Task 1 dihitung dengan mempermutasi query.
