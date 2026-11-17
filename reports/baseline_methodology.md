# Baseline Methodology — Dorfer

Dua baseline untuk project MSMD: **Task 1** score–audio alignment dan **Task 2**
performance error detection. Arsitektur mengikuti paper tanpa dikurangi
kapasitas; yang dibatasi hanya skala data dan durasi training.

> Dorfer, M., Hajič jr., J., Arzt, A., Frostel, H., & Widmer, G. (2018).
> *Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and
> Piece Identification.* TISMIR, 1(1), 22–33.
> Implementasi acuan: `CPJKU/audio_sheet_retrieval`.

## 1. Model

| Sheet `80×100` | Audio `92×42` |
|---|---|
| 2× Conv3×3-24 + BN + ELU + MP2 | sama |
| 2× Conv3×3-48 + BN + ELU + MP2 | sama |
| 2× Conv3×3-96 + BN + ELU + MP2 | sama |
| 2× Conv3×3-96 + BN + ELU + MP2 | sama |
| Conv1×1-32 + BN, GAP | Conv1×1-32 + BN, GAP |
| CCA 32-d + L2 | CCA 32-d + L2 |

Loss ranking hinge (Kiros et al.), `α = 0.7`, negatif in-batch:

$$
\mathcal{L} = \frac{1}{N(N-1)}\sum_i\sum_{k\neq i}
\max\{0,\ \alpha - s(x_i, y_i) + s(x_i, y_k)\}
$$

Hyperparameter Task 1: Adam `2e-3`, weight decay `1e-4`, batch 100, maksimum 8
epoch, early stop 3.

**CCA.** Kovarians `S11`, `S12`, `S22` disimpan sebagai running average dengan
`alpha = 0.05`; proyeksi `U/V` dihitung dari statistik running itu. Warmup 200
step memakai proyeksi identity. Paper memakai `alpha = 1.0`, tetapi pada 100
sampel/batch CCA menghafal batch dan tidak tergeneralisasi; `alpha = 0.05`
memakai mekanisme yang sama dari kode resmi.

## 2. Task 1 — Score–Audio Alignment

Input: crop notehead dan window audio dari pasangan positif core
(`pairs_note_*.parquet`, `label == 1`).

Evaluasi:

1. **Snippet retrieval** dua arah: paired 2018 (2.018 kandidat) dan extended
   (8.072 window). Metrik R@1/5/10/25, MRR, median rank.
2. **System localization**: page-conditioned, full collection (36 system),
   5-candidate. Skor system = cosine maksimum terhadap notehead anggotanya.
3. **Identification**: voting top-25 snippet ke piece/page.

## 3. Task 2 — Performance Error Detection

Empat kelas: `correct`, `wrong_pitch`, `missing_note`, `extra_note`. Error
dibuat sintetis dari window audio bersih; referensi adalah crop notehead sumber.

Model:

1. Inisialisasi kedua encoder dari checkpoint Task 1.
2. Latih **kedua encoder** + fusion classifier (tidak ada yang dibekukan).
3. Karena sheet encoder berubah, embedding tidak di-cache; kedua input
   di-forward setiap step.
4. Fusion `[x, y, x*y, abs(x-y)]` → MLP 128 → 4 kelas. CCA tidak dipakai karena
   klasifikasi tidak punya pasangan lintas modalitas.

Training: cross-entropy, encoder `3e-4`, head `1e-3`, batch 256, maksimum 8
epoch, early stop 3. Train memakai satu varian acak per source per epoch;
validasi/test memakai semua varian.

Evaluasi: accuracy, macro-F1, confusion matrix, top-2, binary correct-vs-error,
error-type accuracy dengan coverage, dan error analysis per soundfont/pitch
delta/metode template/durasi.

## 4. Menjalankan

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-baseline.txt

python -m nbconvert --to notebook --execute --inplace notebooks\03_baseline_task1_dual_encoder.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks\04_baseline_task2_multimodal_classifier.ipynb
```

`CONFIG["fast_dev"] = True` menjalankan subset kecil untuk smoke test.

## 5. Output

- `checkpoints/baseline_task1_dual_encoder.pt`
- `checkpoints/baseline_task2_multimodal_classifier.pt`
- `checkpoints/baseline_task1_test_embeddings.npz`
- `reports/baseline_task1_metrics.json`, `reports/baseline_task2_metrics.json`
- `reports/baseline_task1_history.csv`, `reports/baseline_task2_history.csv`
- `reports/baseline_task2_error_analysis.csv`
- `reports/figures/20_*`–`28_*`

## 6. Batasan

- Train hanya 17 pieces; test 3 pieces.
- Baseline faithful secara arsitektur, tidak sebanding dengan angka paper yang
  memakai 360 train pieces.
- Task 2 memakai error sintetis, bukan anotasi human, dan belum end-to-end
  (referensi partitur diberikan langsung).
- Task 1 memakai agregasi notehead → system, bukan classifier system langsung.
- Crop dinormalisasi dengan `mean/std` dataset; paper hanya `/255`.
- Semua soundfont terlihat di train; belum ada evaluasi unseen-soundfont.
