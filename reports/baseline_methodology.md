# Baseline Methodology — Dorfer

Dokumen ini menjelaskan desain dua baseline untuk project MSMD:

- **Task 1 — Music Score–Audio Alignment**
- **Task 2 — Music Performance Error Detection**

Baseline memakai arsitektur, loss, dan protokol evaluasi dari paper utama tanpa
mengurangi kapasitas model: 4 blok CNN 24/48/96/96, CCA embedding, margin 0.7,
batch 100. Yang dipersingkat hanya jadwal training dan skala data (subset
pilot), bukan arsitekturnya. Dengan begitu delta model yang diusulkan nanti
tetap terukur terhadap baseline paper.

Run reduced-capacity sebelumnya (3 blok, tanpa CCA) disimpan sebagai ablation
di `reports/architecture_dorfer_lite.md` dan tidak dipakai sebagai baseline.

> Dorfer, M., Hajič jr., J., Arzt, A., Frostel, H., & Widmer, G. (2018).
> *Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval and
> Piece Identification.* Transactions of the International Society for Music
> Information Retrieval, 1(1), 22–33. https://doi.org/10.5334/tismir.12

Referensi implementasi paper penuh:
[`CPJKU/audio_sheet_retrieval`](https://github.com/CPJKU/audio_sheet_retrieval),
khususnya `models/mutopia_ccal_cont_rsz.py`.

---

## 1. Ringkasan model

**Baseline** memakai satu model dual encoder dengan dua cabang CNN yang
dilatih bersama, sesuai paper:

| Sheet image `80×100` | Audio spectrogram `92×42` |
|---|---|
| 2× Conv3×3-24 + BN + ELU + MP2 | 2× Conv3×3-24 + BN + ELU + MP2 |
| 2× Conv3×3-48 + BN + ELU + MP2 | 2× Conv3×3-48 + BN + ELU + MP2 |
| 2× Conv3×3-96 + BN + ELU + MP2 | 2× Conv3×3-96 + BN + ELU + MP2 |
| 2× Conv3×3-96 + BN + ELU + MP2 | 2× Conv3×3-96 + BN + ELU + MP2 |
| Conv1×1-32 + BN (linear) | Conv1×1-32 + BN (linear) |
| GAP → CCA 32-d → L2 | GAP → CCA 32-d → L2 |

Fungsi skor: cosine similarity. Loss ranking (Kiros et al.):

$$
\mathcal{L} = \frac{1}{N(N-1)}\sum_i\sum_{k\neq i}
\max\{0,\ \alpha - s(x_i, y_i) + s(x_i, y_k)\}
$$

Hyperparameter:

- `α = 0.7`
- batch 100 (Task 1) / 256 (Task 2)
- negatif = semua sampel lain dalam batch
- Adam, lr awal `2e-3` (Task 1), `3e-4` encoder + `1e-3` head (Task 2)
- weight decay `1e-4` (Task 1), `1e-5` (Task 2)
- CCA mulai di-update setelah 200 step warmup; kovarians memakai running average `alpha = 0.05`.

CCA (`r1 = r2 = rT = 1e-3`) mengikuti `CCALayer` pada implementasi Lasagne
resmi: kovarians `S11`, `S12`, `S22` disimpan sebagai running average dengan
`alpha = 0.05`, lalu proyeksi `U` dan `V` dihitung dari statistik running
tersebut dan disimpan sebagai buffer non-trainable. Gradien mengalir melalui
fitur yang sudah di-center, bukan melalui proyeksi.

> Catatan implementasi: paper memakai `alpha = 1.0` (statistik satu batch)
> karena dilatih sangat lama pada 360 pieces. Pada subset pilot, `alpha = 1.0`
> membuat CCA menghafal batch dan loss turun tanpa representasi yang
> tergeneralisasi. Karena itu dipakai running average `alpha = 0.05` dengan
> warmup 200 step, mengikuti mekanisme yang sama dari kode resmi.

> Catatan fidelity: paper melatih jauh lebih lama (ratusan epoch) pada 360
> train pieces. Baseline ini memakai 17 train pieces dan budget 8 epoch karena
> keterbatasan CPU. Arsitektur sama; skala data dan durasi training tidak.

---

## 2. Task 1 — Music Score–Audio Alignment

**Tujuan:** diberikan gambar partitur dan potongan audio, tentukan lokasi
partitur (system/baris) yang berkorespondensi.

**Baseline:** Dorfer full (dual encoder + CCA + ranking loss), dilatih pada
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
2. Latih **kedua encoder bersama** fusion classifier; tidak ada komponen yang
   dibekukan (no frozen parameters).
3. Forward score crop dan audio setiap step (tidak ada cache embedding, karena
   sheet encoder ikut berubah).
4. Klasifikasi dari embedding 32-d:

```
f = [x, y, x⊙y, |x − y|]  →  MLP(128) → 4 kelas
```

dengan `x` = embedding score, `y` = embedding audio, keduanya 32-d dan
L2-normalized. CCA tidak dipakai di Task 2 karena klasifikasi tidak memiliki
pasangan lintas modalitas dalam satu batch; yang dipakai adalah output trunk
encoder.

**Data:**

```
score_crops_{split}[notehead_id -> crop_row]
error_windows_{split}[error_row]
task2_windows_{split}.parquet: notehead_id, error_row, class_id
```

**Training:** cross-entropy, encoder lr `3e-4`, head lr `1e-3`, batch 256,
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

## 5. Profil CPU (Dorfer full)

| | Task 1 | Task 2 |
|---|---|---|
| Channels | 24/48/96/96 | 24/48/96/96 |
| CCA | ya (32-d, warmup 200) | tidak (trunk 32-d) |
| Batch | 100 | 256 |
| Max epoch | 8 | 8 |
| Early stop | 3 | 3 |
| LR | Adam 2e-3 | encoder 3e-4, head 1e-3 |
| Inisialisasi | from scratch | encoder Task 1 |
| Sheet encoder | dilatih | **dilatih** |
| Audio encoder | dilatih | **dilatih** |
| Data/epoch | 28.730 pairs | ±28.730 (1 varian/source) |

Waktu aktual di CPU Intel Core Ultra 7 155H: **Task 1 ±42 menit** (6 epoch
berjalan, best epoch 3), **Task 2 ±81 menit** (8 epoch), evaluasi ±6 menit.
Total sekitar **2 jam 10 menit**.

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
- `reports/baseline_task1_metrics.json`
- `reports/baseline_task2_metrics.json`
- `reports/baseline_task1_history.csv`
- `reports/baseline_task2_history.csv`
- `reports/baseline_task2_error_analysis.csv`
- `reports/baseline_task2_test_predictions.csv`
- `reports/figures/20_*` sampai `28_*`

---

## 8. Batasan

- Baseline ini memakai **arsitektur Dorfer penuh** (4 blok 24/48/96/96, CCA,
  margin 0.7, batch 100). Yang tidak sama dengan paper adalah skala data
  (sub-pilot) dan panjang training (8 epoch), bukan kapasitas model.
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
