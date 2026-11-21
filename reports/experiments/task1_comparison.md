# Task 1 — Comparison (seed 42)

## Gate A (C1 vs baseline historis)

| metric | historical_pct | C1_pct | delta_pp | tolerance_pp | status |
|---|---|---|---|---|---|
| val_mrr | 1.001 | 1.168 | 0.167 | 0.200 | PASS |
| snippet_a2s_R@1 | 0.099 | 0.099 | 0.000 | n/a | n/a |
| system_full_R@1 | 6.925 | 7.507 | 0.582 | 1.000 | PASS |
| system_5cand_R@1 | 27.329 | 29.125 | 1.796 | 2.000 | PASS |

## Primary metrics (%)

| variant | val_mrr | snippet_a2s_R@1 | snippet_a2s_MRR | system_page_R@1 | system_page_MRR | system_full_R@1 | system_full_MRR | system_5cand_R@1 | system_5cand_MRR | piece_id_R@1 | page_id_a2s_R@1 | page_id_s2a_R@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1.168 | 0.099 | 0.682 | 30.785 | 53.949 | 7.507 | 20.601 | 29.125 | 54.950 | 46.494 | 30.017 | 24.777 |
| C2 | 1.377 | 0.297 | 1.199 | 30.104 | 54.358 | 9.093 | 21.591 | 29.993 | 55.105 | 52.713 | 27.837 | 32.309 |
| C3 | 2.264 | 0.496 | 1.744 | 28.147 | 53.043 | 10.803 | 23.252 | 29.559 | 54.150 | 48.897 | 33.461 | 34.242 |
| C4 | 0.774 | 0.099 | 0.688 | 26.722 | 51.321 | 4.274 | 17.261 | 25.483 | 52.317 | 55.575 | 21.977 | 17.047 |

## Delta vs C1 (percentage points)

| variant | snippet_a2s_R@1 | snippet_a2s_MRR | system_full_MRR | system_5cand_R@1 |
|---|---|---|---|---|
| C1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C2 | 0.198 | 0.517 | 0.990 | 0.867 |
| C3 | 0.396 | 1.063 | 2.652 | 0.434 |
| C4 | 0.000 | 0.006 | -3.339 | -3.642 |

## Per-piece consistency vs C1

**snippet_a2s_R@1**

| variant | improved | same | worse | mean delta (pp) |
|---|---|---|---|---|
| C2 | 3 | 0 | 0 | 0.199 |
| C3 | 3 | 0 | 0 | 0.436 |
| C4 | 1 | 1 | 1 | -0.034 |

**system5_R@1**

| variant | improved | same | worse | mean delta (pp) |
|---|---|---|---|---|
| C2 | 1 | 0 | 2 | 0.855 |
| C3 | 1 | 0 | 2 | 0.381 |
| C4 | 1 | 0 | 2 | -4.632 |

**system_full_MRR**

| variant | improved | same | worse | mean delta (pp) |
|---|---|---|---|---|
| C2 | 1 | 0 | 2 | 0.546 |
| C3 | 2 | 0 | 1 | 2.667 |
| C4 | 1 | 0 | 2 | -4.932 |

## Catatan

- CCA tidak diubah: `alpha=0.05`, `warmup_steps=200`, `r1=r2=rT=1e-3`.
- C1 lolos Gate A pada toleransi yang ditetapkan; lintasan epoch berbeda dari historis (best epoch 7 vs 3) karena perbedaan environment/numerik.
- C3 unggul pada val MRR, full MRR, dan snippet; C2 mendekati C3 pada system metrics; C4 lebih buruk.
- Dengan 1 seed dan ambang 5 poin, belum ada arm yang melewati ambang pada metrik sistem; hasil bersifat eksploratif.