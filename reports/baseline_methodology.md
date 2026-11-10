# Baseline Methodology — Dorfer-lite

Dokumen ini menjelaskan desain dua baseline untuk project MSMD:

- **Task 1 — Music Score–Audio Alignment**
- **Task 2 — Music Performance Error Detection**

Baseline memakai ide, loss, dan protokol evaluasi dari paper utama, tetapi
model diperkecil dan jadwal training dipersingkat agar selesai sekitar 1 jam di
CPU. Profil ini disebut **Dorfer-lite** dan **bukan reproduksi penuh paper**.

> Dorfer, M., Hajič jr., J., Arzt, A., Frostel, H., & Widmer, G. (2018).
> *Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and
> Piece Identification.* Transactions of the International Society for Music
> Information Retrieval, 1(1), 22–33. https://doi.org/10.5334/tismir.12

Referensi implementasi paper penuh:
[`CPJKU/audio_sheet_retrieval`](https://github.com/CPJKU/audio_sheet_retrieval),
khususnya `models/mutopia_ccal_cont_rsz.py`.

---

## 1. Ringkasan model

**Dorfer-lite** tetap memakai satu model dual encoder dengan dua cabang CNN
yang dilatih bersama:

| Sheet image `80×100` | Audio spectrogram `92×42` |
|---|---|
| 2× Conv3×3-16 + BN + ELU + MP2 | 2× Conv3×3-16 + BN + ELU + MP2 |
| 2× Conv3×3-32 + BN + ELU + MP2 | 2× Conv3×3-32 + BN + ELU + MP2 |
| 2× Conv3×3-64 + BN + ELU + MP2 | 2× Conv3×3-64 + BN + ELU + MP2 |
| Conv1×1-32 + BN (linear) | Conv1×1-32 + BN (linear) |
| GAP → 32-d → L2 | GAP → 32-d → L2 |

Perbedaan dari paper penuh:

- 3 blok CNN 16/32/64, bukan 4 blok 24/48/96/96.
- Tanpa CCA embedding layer; langsung L2 + ranking loss.
- 6 epoch, bukan puluhan/1000 epoch.

Fungsi skor: cosine similarity. Loss ranking (Kiros et al.):

$$
\mathcal{L} = \frac{1}{N(N-1)}\sum_i\sum_{k\neq i}
\max\{0,\ \alpha - s(x_i, y_i) + s(x_i, y_k)\}
$$

Hyperparameter lite:

- `α = 0.7`
- batch 128 (Task 1) / 256 (Task 2)
- negatif = semua sampel lain dalam batch
- Adam, lr awal `2e-3` (Task 1), `5e-4` encoder + `1e-3` head (Task 2)
- weight decay `1e-5`

Profil paper penuh masih bisa diaktifkan lewat
`channels=(24, 48, 96, 96)` dan `use_cca=True`, tetapi tidak dipakai pada
baseline lite ini.

---

## 2. Task 1 — Music Score–Audio Alignment

**Tujuan:** diberikan gambar partitur dan potongan audio, tentukan lokasi
partitur (system/baris) yang berkorespondensi.

**Baseline:** Dorfer-lite (dual encoder + ranking loss, tanpa CCA), dilatih pada
positive pairs core:

```
score_crops_{split}.npy  [N, 200, 160, 1]
audio_windows_{split}.npy [N, 42, 92]
pairs_note_{split}.parquet (label == 1)
```

Pasangan positif adalah notehead dan window audio yang benar-benar sama
(`notehead_id` identik). Negatif dibentuk in-batch.

**Evaluasi:**

1. Snippet retrieval dua arah (protokol paper):
   - paired 2018: satu window per notehead, 2.018 kandidat
   - extended: semua 8.072 window terhadap 2.018 crop
   - metrik: R@1/5/10/25, MRR, median rank
2. System localization:
   - **page-conditioned**: diberikan halaman, ranking system di halaman itu
   - full collection: ranking seluruh 36 system test
   - 5-candidate accuracy dari `pairs_system_test`
   - skor system = `max` cosine similarity terhadap notehead anggotanya
3. Piece/page identification:
   - voting top-25 snippet
   - metrik Rk@1/5/10

**Notebook:** `notebooks/03_baseline_task1_dual_encoder.ipynb`

---

## 3. Task 2 — Music Performance Error Detection

**Tujuan:** diberikan referensi partitur dan audio permainan, klasifikasikan
apakah permainan benar atau jenis kesalahannya:

- `0 = correct`
- `1 = wrong pitch`
- `2 = missing note` (delete)
- `3 = extra note` (insert)

**Baseline:** adaptasi dari encoder Dorfer:

1. Muat `sheet encoder` dan `audio encoder` dari checkpoint Task 1.
2. Bekukan `sheet encoder` (crop referensi sama untuk empat varian).
3. Precompute embedding score crop per split.
4. Fine-tune `audio encoder` + fusion classifier:

```
f = [x, y, x⊙y, |x − y|]  →  MLP(128) → 4 kelas
```

dengan `x` = embedding score, `y` = embedding audio, keduanya 32-d dan
L2-normalized.

**Data:**

```
score_crops_{split}[notehead_id -> crop_row]
error_windows_{split}[error_row]
task2_windows_{split}.parquet: notehead_id, error_row, class_id
```

**Training:** cross-entropy, encoder lr `5e-4`, head lr `1e-3`, batch 256,
8 epoch. Dataset train memakai satu varian acak per source window per epoch;
validation/test memakai semua varian.

**Evaluasi:**

- accuracy, macro-F1, per-class precision/recall/F1
- confusion matrix
- top-2 accuracy
- metrik biner correct vs error
- 3-class error-type accuracy among detected errors
- error analysis per soundfont, `pitch_delta`, `template_method`, `fallback`,
  dan durasi

**Notebook:** `notebooks/04_baseline_task2_multimodal_classifier.ipynb`

> **Penting:** Task 2 tidak ada di paper Dorfer. Baseline ini adalah
> **adapted baseline**, bukan model asli paper. Referensi partitur yang dipakai
> adalah crop notehead sumber, bukan seluruh halaman/system.

---

## 4. Struktur artefak yang dipakai

| Artefak | Task 1 | Task 2 |
|---|---|---|
| `score_crops_{split}.npy` | ya | ya (referensi) |
| `audio_windows_{split}.npy` | ya | tidak |
| `error_windows_{split}.npy` | tidak | ya |
| `noteheads_{split}.parquet` | ya | ya |
| `windows_{split}.parquet` | ya | tidak |
| `pairs_note_{split}.parquet` | ya | tidak |
| `notehead_systems_{split}.parquet` | ya (evaluasi) | tidak |
| `systems_{split}.parquet` | ya (evaluasi) | tidak |
| `pairs_system_{split}.parquet` | ya (evaluasi) | tidak |
| `task2_windows_{split}.parquet` | tidak | ya |

Split memakai `group_by_work_group`; tidak ada work group yang muncul di dua
split.

---

## 5. Profil CPU (Dorfer-lite)

| | Task 1 | Task 2 |
|---|---|---|
| Channels | 16/32/64 | 16/32/64 |
| CCA | tidak | tidak |
| Batch | 128 | 256 |
| Max epoch | 6 | 8 |
| Early stop | 3 | 3 |
| LR | Adam 2e-3 | encoder 5e-4, head 1e-3 |
| Inisialisasi | from scratch | encoder Task 1 |
| Sheet encoder | dilatih | dibekukan |
| Data/epoch | 28.730 pairs | ±28.730 (1 varian/source) |

Estimasi waktu di CPU Intel Core Ultra 7 155H: **Task 1 ±15–30 menit**,
**Task 2 ±10–20 menit**, evaluasi ±5–10 menit. Total sekitar **30–60 menit**.

`CONFIG["fast_dev"] = True` menjalankan smoke test pada subset kecil. Untuk
baseline final, set ke `False` dan restart kernel.

---

## 6. Cara menjalankan

```powershell
# 1. Ekstrak artefak ke root repo
#    preproc-v1-core.zip, preproc-v1-task1.zip, preproc-v1-task2.zip

# 2. Install dependency
python -m pip install -r requirements.txt
python -m pip install -r requirements-baseline.txt
# Jika wheel PyTorch untuk Python 3.14 belum tersedia, gunakan venv Python
# 3.11/3.12/3.13 atau jalankan notebook di Colab/Kaggle.

# 3. Jalankan notebook Task 1
python -m nbconvert --to notebook --execute --inplace notebooks\03_baseline_task1_dual_encoder.ipynb

# 4. Jalankan notebook Task 2 setelah checkpoint Task 1 ada
python -m nbconvert --to notebook --execute --inplace notebooks\04_baseline_task2_multimodal_classifier.ipynb
```

Atau buka notebook di Jupyter dan jalankan sel dari atas ke bawah.

---

## 7. Output

- `checkpoints/baseline_task1_dual_encoder.pt`
- `checkpoints/baseline_task2_multimodal_classifier.pt`
- `checkpoints/baseline_task1_test_embeddings.npz`
- `checkpoints/task2_score_bank_{split}.pt` (cache embedding referensi)
- `reports/baseline_task1_metrics.json`
- `reports/baseline_task2_metrics.json`
- `reports/baseline_task1_history.csv`
- `reports/baseline_task2_history.csv`
- `reports/baseline_task2_error_analysis.csv`
- `reports/baseline_task2_test_predictions.csv`
- `reports/figures/20_*` sampai `28_*`

---

## 8. Batasan

- Baseline ini adalah **Dorfer-lite**, bukan reproduksi penuh paper: 3 blok
  16/32/64, tanpa CCA, dan hanya 6 epoch (Task 1) / 8 epoch (Task 2).
- Task 1 memakai agregasi notehead → system, bukan classifier system langsung.
  Ini menjaga input model tetap 160×200 seperti paper.
- Task 1 menormalisasi crop dengan `/255` lalu standardisasi `mean/std` dari
  `norm_stats_note.json`; paper hanya `/255`. Audio sudah dinormalisasi per-bin
  oleh preprocessing, sedangkan paper memakai log-filterbank mentah.
- Task 2 adalah adapted baseline, bukan model asli paper.
- Task 2 memakai referensi notehead crop, belum full page/system, dan belum
  menjalankan pipeline end-to-end Task 1 → Task 2.
- Task 2 train memakai satu varian acak per source per epoch; validation/test
  memakai semua varian.
- Label Task 2 adalah error sintetis, bukan anotasi human.
- Test Task 1 hanya 3 piece / 8 halaman; Task 2 juga memakai split pilot.
- Semua soundfont terlihat di train; belum ada evaluasi unseen-soundfont seperti
  paper.
