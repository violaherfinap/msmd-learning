# Data Card — Note-Level Preprocessing

## Sumber

- Dataset: MSMD augmented v1.1 (no-audio)
- Subset: 25 piece (24 piece note-level; satu piece tanpa MuNG)
- Unit: crop notehead ↔ window audio di sekitar onset
- Split: group by `work_group`

## Jumlah

| Split | Crop | Window | Pair |
|---|---:|---:|---:|
| Train | 14,365 | 28,730 | 57,460 |
| Val | 2,430 | 9,720 | 19,440 |
| Test | 2,018 | 8,072 | 16,144 |

## Schema

- `score_crops_{split}.npy`: `[N, 200, 160, 1]`, `uint8`, grayscale mentah.
- `audio_windows_{split}.npy`: `[N, 42, 92]`, `float16`, spectrogram ternormalisasi per bin, clip `[-5, 5]`.
- `noteheads_{split}.csv`: metadata crop; `crop_row` adalah index tensor.
- `windows_{split}.csv`: metadata window; `window_row` adalah index tensor.
- `pairs_note_{split}.csv`: pasangan positif/negatif untuk retrieval.
- `pairs_note_hard_{split}.csv`: negatif keras tambahan; tidak mengubah file lama.
- `norm_stats_note.json`: statistik normalisasi gambar/audio.

## Normalisasi

- Gambar: `(pixel / 255 - mean) / std`; mean `0.893631`, std `0.265134`.
- Audio: sudah dinormalisasi saat preprocessing; jangan normalisasi ulang.

## Coverage

- Total piece: 24
- Coverage minimum: 45.32%
- Coverage median: 98.22%
- Coverage maksimum: 100.00%

## Hard negatives

- Train: 14,365
- Val: 2,430
- Test: 2,018

## Batasan

- Test hanya 3 piece.
- Crop 160x200 dapat memuat lebih dari satu system.
- Belum ada raw audio.
- Belum ada model/training pada tahap ini.

## Verifikasi

```powershell
python scripts/validate_preprocessing.py --check-raw-paths
```
