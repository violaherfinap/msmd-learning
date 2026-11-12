# Task 1 — Hasil Eksperimen C1–C4

## Setup

- Dataset: MSMD augmented v1.1 no-audio, pilot 24 piece.
- Split tetap: train 17 piece, val 4 piece, test 3 piece (8 halaman test).
- Task 1: 28.730 pasangan positif/epoch; val 2.430 notehead; test 8.072 window.
- CPU-only, seed 42, checkpoint dipilih dari val, test dievaluasi sekali.
- C4 memakai encoder beku dari checkpoint C3 dan melatih ranking head listwise.
- Rencana diperluas menjadi 5 konfigurasi per task: Task 1 C1-C5 (`embed_dim 64`
  sebagai C5) dan Task 2 T2-C1..T2-C5 (semua multimodal, T2-C5 gated fusion).
  Detail di `reports/EXPERIMENT_PLAN.md`.

## Tabel perbandingan

| Run | Experiment | Val MRR | Protokol val | Page R@1 | Page R@5 | 5-cand R@1 | 5-cand MRR | Full R@1 | Snippet MRR | Waktu |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | dual encoder | 0,0078 | snippet | 0,2849 | **0,9610** | 0,3085 | 0,5668 | 0,0723 | **0,0122** | 31,7 mnt |
| C2 | dual encoder | 0,0096 | snippet | 0,2587 | 0,9454 | 0,2992 | 0,5494 | 0,0728 | 0,0094 | 37,8 mnt |
| C3 | dual encoder | **0,0139** | snippet | 0,2520 | 0,9590 | 0,2584 | 0,5338 | 0,0519 | 0,0080 | 67,8 mnt |
| C4 | ranking head | 0,5410 | 5-cand | **0,3303** | 0,9480 | **0,3527** | **0,5914** | **0,1100** | — | **4,3 mnt** |

Catatan: Val MRR C4 adalah MRR 5-kandidat, bukan snippet MRR, jadi tidak
dibandingkan langsung dengan C1–C3.

## Temuan utama

1. **Baseline C1 tetap kompetitif.** C1 unggul di page R@5 (0,9610) dan
   snippet MRR (0,0122). Ini menunjukkan resep Dorfer-lite 3 blok masih kuat
   untuk data pilot yang kecil.
2. **Tuning C2 tidak membantu test.** Val snippet MRR naik dari 0,0078 ke
   0,0096, tetapi page R@1 turun 2,6 poin dan 5-cand R@1 turun 0,9 poin.
   Regularisasi/lr yang lebih rendah tampaknya lebih lambat konvergen pada
   anggaran 12 epoch dan tidak menutup gap ke test.
3. **Arsitektur C3 + CCA menaikkan val, tetapi tidak test.** Val snippet MRR
   naik 78% ke 0,0139, namun page R@1 turun 3,3 poin dan 5-cand R@1 turun
   5,0 poin dibanding C1. Kapasitas tambahan belum terbayar pada 17 piece
   train. CCA tidak menghasilkan NaN, jadi tidak perlu fallback.
4. **C4 ranking head adalah pemenang system localization.** Dengan encoder C3
   yang dibekukan, C4 menaikkan:
   - page R@1: 0,3303 (+4,5 poin vs C1; +7,8 vs C3),
   - 5-cand R@1: 0,3527 (+4,4 poin vs C1; +9,4 vs C3),
   - full collection R@1: 0,1100 (+3,8 poin vs C1; +5,8 vs C3),
   - page MRR: 0,5627 vs 0,5369 (C1),
   - 5-cand MRR: 0,5914 vs 0,5668 (C1).
   Namun page R@5 C4 sedikit di bawah C1 (0,9480 vs 0,9610), jadi C1 masih
   lebih baik pada recall longgar.
5. **Snippet retrieval belum terpecahkan.** Semua run berada di snippet MRR
   0,0080–0,0122. Sinyal utama yang berguna ada di level system, bukan
   notehead. C4 tidak memperbaiki snippet retrieval karena ranking head hanya
   memakai fitur similaritas yang sudah ada.
6. **C4 jauh lebih efisien.** 4,3 menit vs 31,7–67,8 menit untuk dual encoder,
   karena encoder dibekukan dan hanya MLP ranking yang dilatih.

## Analisis error

- Error C1 dan C4 terkonsentrasi pada halaman dengan 5–6 system.
  - Ukuran halaman 6: C1 R@1 0,2207 -> C4 0,2934 (+7,3 poin).
  - Ukuran halaman 5: C1 0,2425 -> C4 0,3152 (+7,3 poin).
  - Ukuran halaman 4: C1 0,3610 -> C4 0,3130 (-4,8 poin).
  - Ukuran halaman 3: C1 0,5735 -> C4 0,4853 (-8,8 poin).
  - Halaman dengan 1 system selalu R@1 = 1,0 (trivial).
- Top confusion hampir semuanya antar-system bertetangga di halaman yang sama,
  misalnya `KumarR p02 s05 -> s01`, `KumarR p02 s04 -> s05`, dan
  `MagnenatS p02 s00 -> s03`. Ini konsisten dengan kesalahan lokasi vertikal
  pada partitur, bukan kesalahan piece/page.
- C4 memperbaiki banyak confusion C1, tetapi masih tertukar pada system yang
  berdekatan dan assume satu halaman.

## Batasan

- Pilot 25 piece, test hanya 3 piece / 8 halaman. Jangan mengklaim generalisasi
  ke seluruh MSMD.
- Satu seed (42) per konfigurasi. Finalis perlu 3 seed (42/123/2024).
- C4 memakai checkpoint C3 yang dipilih dari val snippet MRR; ranking head
  dilatih pada train dan dievaluasi sekali di test.
- Val MRR C4 memakai protokol 5-kandidat, bukan snippet, sehingga bukan
  perbandingan apel-ke-apel dengan C1–C3.
- Task 2 belum dijalankan pada iterasi ini; fokus eksperimen ada di Task 1.
- CPU-only; waktu training bergantung pada beban mesin.

## Artefak

- `reports/experiments/<run_id>/`: `config.json`, `history.csv`,
  `metrics.json`, `predictions.csv`, `predictions_system.csv`, `run.log`,
  `figures/`.
- `reports/experiments/comparison/`: `comparison.csv`, `comparison.md`,
  `comparison.png`, `snippet_mrr.png`, `error_by_page.csv`,
  `error_by_page_size.csv`, `top_confusions.csv`,
  `page_r1_by_page.png`, `page_r1_by_page_size.png`.
- Checkpoint: `checkpoints/experiments/<run_id>.pt` (tidak di-commit).

## Reproduksi

```powershell
python scripts\run_experiment.py --config configs\experiments\t1_c1_baseline.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c2_tuned.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c3_architecture.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c4_ranking_head.yaml
python scripts\compare_experiments.py
python scripts\analyze_task1_errors.py
```

Smoke test:

```powershell
python scripts\run_experiment.py --config configs\experiments\t1_c4_ranking_head.yaml --fast-dev
```

## Langkah berikutnya

1. Tambah C5 (`embed_dim: 64`) untuk melengkapi 5 konfigurasi Task 1.
2. Jalankan 3 seed untuk finalis Task 1 (kandidat C1 dan C4), laporkan mean/std.
3. Fokus perbaikan page R@5 dan system bertetangga: kalibrasi skor,
   hard-negative mining yang lebih agresif, atau fitur posisi vertikal.
4. Port ranking head ke arsitektur C1 jika C3 tetap tidak membantu.
5. Jalankan 5 konfigurasi Task 2 multimodal: T2-C1 baseline, T2-C2 regularisasi,
   T2-C3 unfreeze sheet, T2-C4 backbone 4 blok, T2-C5 gated fusion. Tidak ada
   ablasi audio-only.
6. Jangan tuning di test; test hanya dievaluasi sekali per konfigurasi final.
