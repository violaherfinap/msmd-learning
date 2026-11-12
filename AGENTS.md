# AGENTS.md — MSMD Learning

Acuan kerja untuk agent/kontributor di repo ini. Baca bagian ini sebelum mengubah
kode atau menjalankan training.

## Ringkasan proyek

Repo ini mengerjakan dua task pada dataset MSMD augmented v1.1 (no-audio):

- **Task 1** — score-audio alignment / system localization.
- **Task 2** — performance error detection (4 kelas sintetis: correct, wrong_pitch,
  missing_note, extra_note).

Status: preprocessing selesai dan tervalidasi; baseline Dorfer-lite sudah ada.
Branch kerja: `feature/baseline-dorfer-lite`.

## Aturan wajib

1. Jangan commit `data/processed/*.npy`, `*.npz`, `checkpoints/`, atau artefak besar.
2. Jangan mengubah split. Gunakan `data/splits.csv` dan split per `work_group`.
3. Tuning hanya di val. Test dievaluasi sekali di akhir.
4. Satu konfigurasi eksperimen mengubah satu faktor utama.
5. Setiap run wajib menyimpan config, seed, history, metrics, dan prediksi.
6. Jangan menjalankan `python scripts\run_preprocessing.py --stage all` kecuali
   dataset mentah 18 GB tersedia dan memang diminta.
7. Jangan mengklaim hasil Task 2 sebagai performa dunia nyata: labelnya sintetis.
8. Jangan mengubah `scripts/baseline_dorfer.py` tanpa alasan; tambahkan fungsi
   baru, jangan menduplikasi dataset/model/loss yang sudah ada.

## Environment

- Windows PowerShell 5.1, Python 3.11, PyTorch CPU.
- Hardware acuan: Intel Core Ultra 7 155H, 32 GB RAM, CPU-only.
- Install:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-baseline.txt
```

- Verifikasi artefak:

```powershell
python scripts\run_preprocessing.py --verify-only
```

Harus menghasilkan `"ok": true` dan `Validation passed`.

## Data

- File `.npy`/`.npz` tidak ada di Git. Restore dari Release:

```powershell
.\scripts\extract_release.ps1 -DryRun
.\scripts\extract_release.ps1
```

- Jangan extract ke dalam `data/processed` (akan menjadi path bersarang).
- Aset Release: `preproc-v1-core.zip`, `preproc-v1-task1.zip`,
  `preproc-v1-task2.zip`, `preproc-v1-meta.zip`.
- Pilot: 25 piece dipilih, 24 usable. Train 17 / val 4 / test 3 piece.
- Task 1 test: 8.072 query, 36 system, 8 halaman.
- Task 2 test: 32.288 baris; 4 kelas seimbang sempurna (8.072 per kelas).
- Bentuk artefak:
  - `score_crops_{split}.npy`: `uint8 [N, 200, 160, 1]`
  - `audio_windows_{split}.npy`: `float16 [N, 42, 92]`, sudah normalisasi
  - `system_crops_{split}.npy`: `uint8 [N, 128, 512, 1]`
  - `score_pages_{split}.npy`: `uint8 [N, 256, 256, 1]`
  - `error_windows_{split}.npy`: `float16 [N, 42, 92]`, sudah normalisasi
- Normalisasi:
  - `score_crops`: `(pixel/255 - 0.8936307) / 0.2651336`
  - `system_crops`: `(pixel/255 - 0.9099208) / 0.2053268`
  - `score_pages`: `(pixel/255 - 0.9429144) / 0.1269404`
  - `audio_windows` dan `error_windows`: jangan normalisasi ulang.
- `pairs_note` berisi positif+negatif; filter `label == 1` untuk pasangan positif.
- `pairs_system`: 1 positif + 4 negatif per `query_id`.
- `task2_windows`: 4 varian per `source_window_id`; kolom `class_id` 0-3.

## Struktur penting

- `scripts/baseline_dorfer.py`: dataset, model, loss, metrik, helper embedding.
- `notebooks/03_baseline_task1_dual_encoder.ipynb`: baseline Task 1.
- `notebooks/04_baseline_task2_multimodal_classifier.ipynb`: baseline Task 2.
- `reports/`: metrik, history, figure, QA.
- Target baru (belum ada):
  - `configs/experiments/*.yaml`
  - `scripts/run_experiment.py`
  - `reports/experiments/<run_id>/`

## Baseline acuan

| Task | Metrik | Nilai |
|---|---|---|
| Task 1 | snippet MRR | 0.0086 |
| Task 1 | page-conditioned system R@1 / R@5 | 0.275 / 0.945 |
| Task 1 | 5-kandidat R@1 | 0.256 |
| Task 2 | val macro-F1 | 0.964 |
| Task 2 | test macro-F1 | 0.990 |

Task 2 sisa error hampir hanya correct vs insert; semua soundfont muncul di train.

## Perintah

```powershell
# Verifikasi artefak
python scripts\run_preprocessing.py --verify-only

# Jalankan baseline lewat notebook
python -m nbconvert --to notebook --execute --inplace notebooks\03_baseline_task1_dual_encoder.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks\04_baseline_task2_multimodal_classifier.ipynb

# Smoke test (setelah runner dibuat)
python scripts\run_experiment.py --config configs\experiments\t1_c1_baseline.yaml --fast-dev

# Eksperimen penuh (setelah runner dibuat)
python scripts\run_experiment.py --config configs\experiments\t1_c1_baseline.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c2_tuned.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c3_architecture.yaml
python scripts\run_experiment.py --config configs\experiments\t1_c4_ranking_head.yaml
```

Smoke test manual: set `CONFIG["fast_dev"] = True` di notebook, restart kernel,
jalankan 2 epoch.

## Protokol eksperimen

- Split tetap: train/val/test dari artefak.
- Seed 42 untuk semua run; 3 seed (42/123/2024) untuk finalis.
- Pilih checkpoint berdasarkan val; test sekali.
- Simpan ke `reports/experiments/<run_id>/`:
  `config.json`, `history.csv`, `metrics.json`, `predictions.csv`, `figures/`.
- Metrik Task 1: system R@1/R@5/MRR, 5-kandidat accuracy, snippet R@1/5/MRR.
- Metrik Task 2: macro-F1, per-class F1, confusion, ECE, per-soundfont F1,
  gap val ke test.
- Setiap run baru harus dibandingkan dengan baseline yang sama, bukan dengan
  angka paper.

## Konvensi kode

- Ikuti gaya `scripts/baseline_dorfer.py` (docstring, type hints sederhana).
- Gunakan `pathlib.Path`, bukan string path.
- Gunakan `set_seed`, `configure_torch`, `save_json`, `to_jsonable` yang sudah ada.
- Jangan memakai Unicode arrow/karakter non-ASCII di `print` pada Windows;
  console cp1252 bisa error. Gunakan ASCII atau
  `sys.stdout.reconfigure(encoding="utf-8")`.
- Notebook harus tipis; logika training/evaluasi di `scripts/`.
- `num_workers=0` di Windows.
- Jangan menambah dependency baru tanpa memperbarui `requirements*.txt`.

## Pitfalls

- `.npy` hilang: training tidak bisa jalan; restore Release dulu.
- `data/subset_performances.csv` berisi path lama `C:\Users\HP\...`; jangan dipakai
  sebagai path absolut.
- `norm_stats_note.json` punya field legacy `segment_frames=100`; window sekarang
  42 frame dan field itu tidak dipakai loader.
- `channels` Task 2 harus sama dengan checkpoint Task 1.
- CCA bisa NaN; jika terjadi, fallback ke `use_cca=False` dan catat.
- Jangan menilai Task 2 hanya dari test accuracy; cek val, kalibrasi, dan
  per-soundfont.
- Class balancing tidak relevan untuk Task 2 (sudah seimbang); fokus ke
  hard-negative mining untuk Task 1.
- Test Task 1 hanya 3 piece; jangan overclaim generalisasi.

## Definition of done

- Smoke test `fast_dev` lolos (loss finite, checkpoint tersimpan).
- Minimal 4 konfigurasi Task 1 berjalan dan tersimpan.
- Ada tabel perbandingan + figure + analisis error.
- Laporan menuliskan batasan: pilot 25 piece, test 3 piece, label Task 2 sintetis,
  CPU-only.

## Dokumen acuan

- `reports/EXPERIMENT_PLAN.md` — rencana 4 konfigurasi.
- `reports/baseline_methodology.md` — metodologi baseline.
- `reports/baseline_summary.md` — hasil baseline.
- `reports/HANDOFF_TEMAN.md` — panduan artefak dan loader.
- `README.md` — gambaran umum repo.
