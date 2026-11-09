# Ringkasan Preprocessing MSMD

Dokumen ini adalah sumber utama untuk kondisi preprocessing saat ini. Dokumen ini
menggantikan laporan core-level lama yang masih memakai CSV dan belum memuat Task 1
dan Task 2.

## Status

- Pipeline preprocessing: **selesai**
- Validasi artefak: **lulus**
- QA visual: **sudah dibuat dan diperiksa**
- Dataset: **25 piece pilot**, 24 piece usable
- Training/model: **tidak termasuk** di repo ini; dikerjakan teman
- HTML current: **reports/preprocessing_report_current.html** (self-contained)

## 1. Ringkasan eksekutif

| Item | Nilai |
|---|---:|
| Total piece dataset | 697 |
| Piece pilot | 25 |
| Piece usable untuk core/Task 1/Task 2 | 24 |
| Performance pilot | 100 |
| Performance usable | 96 |
| Crop notehead | 18.813 |
| Window audio core | 46.522 |
| Pair note-level | 93.044 |
| Halaman Task 1 | 60 |
| System Task 1 | 314 |
| Pair Task 1 | 232.610 |
| Window Task 2 | 186.088 |
| Ukuran artefak | ±2,4 GB |

## 2. Keputusan desain

- Task 1 hanya sampai level **system**, bukan bar, karena `coords/bars_*.npy` kosong.
- Task 2 memakai error **sintetis** dari window bersih, bukan label human.
- Split dilakukan per `work_group`, bukan per performance atau per baris.
- Statistik normalisasi hanya dari train.
- Performance train memakai 2 performance per notehead; val/test memakai semua
  performance yang tersedia pada subset.
- Crop gambar disimpan mentah `uint8`; audio/error window disimpan
  `float16` ternormalisasi dan di-clip.
- File tensor besar tidak masuk Git; distribusinya lewat Release/zip.

## 3. Dataset dan exclusion

- Total piece: 697.
- Total performance: 9.015.
- Piece lengkap: 600.
- Piece dengan MuNG: 593.
- Piece dengan MuNG dan performance lengkap: 586 (potensi full dataset).
- 104 piece tidak punya MuNG.
- 7 piece punya MuNG tetapi tidak punya performance lengkap.
- 96 piece memiliki folder `performances/` kosong.
- 1 piece Beethoven (`BeethovenLv__O111__lvb_sonate_111_1`) memiliki 15 performance
  tetapi semua `features/` kosong.
- Subset pilot: 25 piece, 100 performance, 4 soundfont per piece.
- `GobbaertsL__Les_Etoiles_dOr_No1__Les_Etoiles_dOr_No1` ada di subset dan masuk
  test, tetapi tidak punya MuNG, sehingga dikeluarkan dari core/Task 1.
- Hasil akhir: 24 piece usable, 96 performance.

File metadata:

- `data/subset_pieces.csv`
- `data/subset_performances.csv`
- `data/splits.csv`

Catatan: kolom `features_dir` di `subset_performances.csv` adalah path absolut lama
dan tidak boleh dipakai langsung. Pipeline menurunkan path dari `piece` dan
`perf_name`.

## 4. Split

Split pilot pada level piece:

| Split | Piece di subset | Piece usable | Performance usable |
|---|---:|---:|---:|
| Train | 17 | 17 | 68 |
| Val | 4 | 4 | 16 |
| Test | 4 | 3 | 12 |
| **Total** | **25** | **24** | **96** |

Tidak ada `work_group` yang muncul di dua split. Ini mencegah leakage varian karya
yang sama.

## 5. Core note-level

| Split | Crop | Window | Pair |
|---|---:|---:|---:|
| Train | 14.365 | 28.730 | 57.460 |
| Val | 2.430 | 9.720 | 19.440 |
| Test | 2.018 | 8.072 | 16.144 |
| **Total** | **18.813** | **46.522** | **93.044** |

Detail:

- Crop: `uint8 [N, 200, 160, 1]`.
- Window: `float16 [N, 42, 92]`.
- Image mean/std train: `0.8936307 / 0.2651336`.
- Alignment audio: `T = min(spec_T, midi_T)`.
- 72 dari 100 performance memiliki `spec_T = midi_T + 1`.
- `notehead_id` stabil: `page_id__nh{mung_id}`.
- `pairs_note_*`: 1 positif + 1 negatif dari piece/work_group berbeda.
- `train_perfs_per_note=2`; val/test memakai performance yang tersedia.

## 6. Task 1: system localization

| Split | Halaman | System | Query | Pair (1 positif + 4 negatif) |
|---|---:|---:|---:|---:|
| Train | 44 | 240 | 28.730 | 143.650 |
| Val | 8 | 38 | 9.720 | 48.600 |
| Test | 8 | 36 | 8.072 | 40.360 |
| **Total** | **60** | **314** | **46.522** | **232.610** |

Detail:

- Page tensor: `uint8 [N, 256, 256, 1]`.
- System crop: `uint8 [N, 128, 512, 1]`.
- Assignment notehead ke system:
  - 16.491 polygon,
  - 2.322 fallback nearest y-center,
  - 0 unassigned.
- Statistik train:
  - page mean/std `0.9429144 / 0.1269404`,
  - system mean/std `0.9099208 / 0.2053268`.
- Negatif per query: `same_page`, `same_piece_other_page`, `cross_piece`, `other`.
- `pairs_system_*` berisi 5 baris per `query_id`.

## 7. Task 2: 4-class synthetic errors

| Split | Source | Clean | Wrong pitch | Delete | Insert | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 28.730 | 28.730 | 28.730 | 28.730 | 28.730 | 114.920 |
| Val | 9.720 | 9.720 | 9.720 | 9.720 | 9.720 | 38.880 |
| Test | 8.072 | 8.072 | 8.072 | 8.072 | 8.072 | 32.288 |
| **Total** | **46.522** | **46.522** | **46.522** | **46.522** | **46.522** | **186.088** |

Label:

- `0 = clean`
- `1 = wrong_pitch`
- `2 = delete`
- `3 = insert`

Detail:

- Error window: `float16 [N, 42, 92]`.
- Template pitch dibangun dari **train saja**.
- Template coverage direct per soundfont:
  - ElectricPiano: 52 pitch,
  - YamahaGrandPiano: 47 pitch,
  - acoustic_piano_imis_1: 29 pitch,
  - grand-piano-YDP-20160804: 43 pitch.
- Global pitch range: 31–91.
- Metode template total: direct 171, ridge 33, nearest 0, global 40.
- 42 varian memakai `fallback_overlap`; 0 skip.
- Setiap source window menghasilkan tepat 4 varian kelas.
- `templates_task2.npz` hanya untuk generasi, bukan input training.

## 8. Validasi

Validasi utama:

```powershell
python scripts\run_preprocessing.py --verify-only
```

Hasil: `"ok": true` dan `Validation passed`.

Pemeriksaan yang sudah lulus:

- 60/60 hash artifact cocok dengan `manifest_preprocessing.json`.
- 4 zip release lolos `testzip`.
- 90 sampel window direcompute dari raw: max diff `0.0`.
- 300 sampel crop: `ink_ratio` cocok; tidak ada crop blank.
- 60 system crop + 36 page tensor direcompute: max diff `0.0`.
- 240 sampel assignment notehead: 0 error.
- Semua 46.522 varian `clean` identik dengan source window: max diff `0.0`.
- 0 template target missing.
- 0 `work_group` lintas split.
- Negatif pair tidak berasal dari piece/work_group yang sama.
- Task 2: 0 source window yang kehilangan kelas; 0 skip.

### QA visual

Script:

- `scripts/qa_visual.py`
- `scripts/trace_sample.py`

Output:

- `reports/qa_v2/01_notehead_page_overlay.png`
- `reports/qa_v2/02_notehead_crops.png`
- `reports/qa_v2/03_audio_windows.png`
- `reports/qa_v2/04_task1_system_overlay.png`
- `reports/qa_v2/05_task1_system_crops.png`
- `reports/qa_v2/06_task2_class_grid.png`
- `reports/qa_v2/07_task2_diff_examples.png`
- `reports/qa_v2/qa_summary.json`
- `reports/qa_v2/trace_samples.json`

Temuan QA:

- Overlay notehead dan system cocok dengan halaman asli.
- Crop notehead berisi area not; beberapa crop memuat lebih dari satu system,
  sesuai batasan crop `160x200`.
- Window audio menampilkan onset pada posisi yang benar.
- Task 1 menampilkan 1 positif + 4 negatif; kandidat off-page tidak digambar
  sebagai kotak di halaman query.
- Task 2 menunjukkan perbedaan lokal yang masuk akal:
  - `wrong_pitch` mengubah pola pitch,
  - `delete` menghapus region note,
  - `insert` menambah attack setelah onset,
  - `clean` memiliki diff nol.
- Trace 3 sampel:
  - crop raw vs artifact max diff `0`,
  - window finite dan nonzero,
  - onset di dalam window,
  - `aligned_T` cocok,
  - Task 1 query 5 baris dengan 1 positif,
  - Task 2 lengkap 4 kelas.

## 9. Artefak

Lokasi kerja:

```text
data/processed/
```

Kelompok utama:

- Core:
  - `score_crops_{split}.npy`
  - `audio_windows_{split}.npy`
  - `noteheads_{split}.parquet`
  - `windows_{split}.parquet`
  - `pairs_note_{split}.parquet`
- Task 1:
  - `score_pages_{split}.npy`
  - `system_crops_{split}.npy`
  - `pages_{split}.parquet`
  - `systems_{split}.parquet`
  - `notehead_systems_{split}.parquet`
  - `pairs_system_{split}.parquet`
- Task 2:
  - `error_windows_{split}.npy`
  - `task2_windows_{split}.parquet`
  - `templates_task2.npz`
  - `templates_task2.json`
  - `task2_class_map.json`
- Metadata:
  - `norm_stats.json` (input normalisasi audio untuk regenerate core)
  - `norm_stats_note.json`
  - `norm_stats_task1.json`
  - `manifest_preprocessing.json`
  - `SHA256SUMS.txt`
  - `preprocess_summary_note.json`
  - `preprocess_summary_task1.json`
  - `preprocess_summary_task2.json`

File `.npy` dan `templates_task2.npz` tidak masuk Git. Yang masuk Git adalah
Parquet/JSON kecil, manifest, dan checksum.

### Release

Zip di `dist/`:

- `preproc-v1-core.zip` 963.772.424 byte
- `preproc-v1-task1.zip` 26.649.048 byte
- `preproc-v1-task2.zip` 1.445.473.082 byte
- `preproc-v1-meta.zip` 82.419 byte
- `preproc-v1-*` total ±2,4 GB

Checksum:

- `dist/SHA256SUMS.txt`
- `dist/release_manifest.json`
- `data/processed/SHA256SUMS.txt`

Status upload Release: **belum dilakukan** saat dokumen ini dibuat.

## 10. HTML current dan arsip legacy

Laporan HTML current:

- `reports/preprocessing_report_current.html`
  - self-contained: CSS inline dan 7 gambar QA di-embed,
  - memuat core, Task 1, Task 2, validasi, QA, artefak, dan batasan,
  - bukan hasil generator core-only lama.

File berikut sudah dihapus dari branch kerja karena legacy dan bukan sumber
kebenaran saat ini:

- `reports/preprocessing_report.html`
- `reports/02_note_level_preprocessing.html`
- `reports/preprocessing_report.md`
- `reports/preprocessing_datacard.md`
- `reports/methodology_dataset.md`
- `notebooks/01_eda_preprocessing.ipynb`
- `notebooks/02_note_level_preprocessing.ipynb`
- `reports/figures/`
- CSV/JSON legacy di `data/processed/`

Semua file tersebut tetap tersedia di branch arsip `01-preprocessing` dan
history Git. Jangan dipakai untuk handoff training atau validasi current.

## 11. Keterbatasan dan warning

- Test pilot hanya 3 piece; metrik mudah berubah.
- Task 1 hanya level system, tidak bisa klaim bar-level.
- 2.322 notehead memakai fallback nearest y-center; QA assignment sampel lulus,
  tetapi seluruh fallback belum diperiksa satu per satu secara visual.
- Task 2 memakai label sintetis; belum tentu mewakili error musisi nyata.
- `delete` bisa halus jika window berisi banyak note.
- `insert` bisa menambah pitch yang sudah ada.
- Tidak ada audio mentah; hanya spektrogram dan MIDI. MIDI bukan audio.
- `norm_stats_note.json` memiliki field legacy `audio.segment_frames=100`,
  sedangkan artifact window sekarang **42 frame**. Field itu tidak dipakai loader;
  gunakan `win_frames`/shape tensor sebagai acuan.
- `data/subset_performances.csv` masih menyimpan `features_dir` absolut dari mesin
  lama; jangan dipakai langsung.
- Full dataset belum diproses. Full yang mungkin adalah 586 piece dengan MuNG dan
  performance lengkap, bukan 697.

## 12. Handoff

Panduan untuk teman yang akan memakai artefak dan melakukan training:

- `reports/HANDOFF_TEMAN.md`

Script extract:

- `scripts/extract_release.ps1`

Dokumen ini dan handoff tidak mengubah data dan tidak melakukan training.
Laporan HTML current ada di `reports/preprocessing_report_current.html`.
