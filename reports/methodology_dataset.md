# Metodologi Dataset

## 1. Sumber Dataset

Dataset yang digunakan adalah **MSMD (Multimodal Sheet Music Dataset)** karya Dorfer et al. (2018), dalam varian *augmented* `msmd_aug_v1-1_no-audio`. Dataset ini berisi partitur piano klasik, performance MIDI, dan fitur turunan audio yang saling terhubung. Varian yang dipakai **tidak menyertakan file audio mentah** (`.flac`/`.wav`); modalitas audio direpresentasikan oleh spektrogram dan piano roll yang sudah dihitung.

## 2. Jumlah dan Struktur Data

- Total folder piece: **697**
- Piece dengan score: **654**
- Piece dengan performance: **601**
- Piece dengan score + performance: **601**
- Piece siap pakai (score + 5 fitur lengkap): **600**
- Total folder performance: **9015**
- Performance dengan 5 fitur lengkap: **9000**
- Halaman partitur (PNG): **2041**
- Jumlah composer: **70**

Setiap piece memiliki struktur:

```
<piece>/
  *.ly, *.norm.ly, *.midi, meta.yml
  scores/<piece>_ly/
    *.pdf, img/*.png, coords/*.npy, mung/*.xml
  performances/<piece>_tempo-XXX_<soundfont>/
    *.midi, features/*.flac_{notes,onsets,durations,midi,spec}.npy
```

## 3. Fitur dan Representasi

- **Visual:** gambar halaman partitur (RGBA, umumnya 835x1181), notasi graph MuNG, dan koordinat notehead.
- **Audio:** `spec.npy` (92 bin log-filterbank x frame, 20 fps), `midi.npy` (128 pitch x frame), `notes.npy` (onset, pitch, durasi, velocity, voice), `onsets.npy`, dan `durations.npy`.
- **Simbolik:** file MIDI dan sumber LilyPond.

## 4. Target/Label

Dataset tidak memiliki kolom target eksplisit. Target diturunkan dari struktur folder:

- `piece_id` sebagai kelas utama untuk tugas pencocokan/retrieval.
- `composer` sebagai label sekunder.
- Label biner pasangan: `1` jika partitur dan audio berasal dari piece yang sama, `0` jika berbeda.
- Pasangan positif bersifat **piece-level**: halaman partitur piece X dipasangkan dengan segmen audio mana pun dari piece X (belum alignment temporal note-per-note).

## 5. Distribusi dan Karakteristik

- Jumlah composer: **70**, dengan distribusi sangat tidak seimbang (Bach mendominasi).
- Jumlah work group: **685**; terdapat **10** work group berisi lebih dari satu piece (varian/encoding karya yang sama).
- Median halaman per piece: **2** (maksimum **24**).
- Median note per performance (sampel 1.200): **567**.
- Median durasi performance (sampel 1.200): **1.5 menit**.
- Frame rate fitur audio: **20 fps**; jumlah frame konsisten dengan durasi.
- Median alignment error rate: **0.0016**.

## 6. Keterbatasan

1. Tidak ada file audio mentah; audio hanya tersedia sebagai fitur turunan.
2. Frame rate 20 fps; `onsets` dan `durations` dalam frame, bukan detik.
3. Koordinat `coords` memakai `(y, x)`, tidak lengkap, dan `bars_*.npy` kosong.
4. Flag `aligned_well` dan `processed` bernilai `false` untuk seluruh piece sehingga tidak dapat dipakai sebagai filter kualitas.
5. Sebagian score tidak memiliki MuNG; sebagian performance tidak memiliki fitur lengkap.
6. Performance dari piece yang sama bukan sampel independen.

## 7. Subset dan Split

Subset yang diproses: **25 piece**, **100 performance** (4 performance per piece: satu per soundfont, tempo terdekat 1000), berasal dari **25 composer**. Split dilakukan pada level `work_group` dengan proporsi 70/15/15:

{
  "test": 4,
  "train": 17,
  "val": 4
}

Split group-level mencegah kebocoran varian karya yang sama antar train/val/test.

## 8. Preprocessing

- **Visual:** konversi RGBA ke grayscale (composite atas latar putih), **pad** ke 384x384 agar aspect ratio terjaga, lalu standardisasi memakai mean/std train.
- **Audio:** `spec.npy` ditranspose ke (frame, 92), dinormalisasi **z-score per-bin** memakai statistik train, di-clip ke [-5, 5], lalu dipotong menjadi segmen **5 detik (100 frame)**.
- **Pencegahan leakage:** statistik normalisasi hanya dihitung dari train; split dilakukan sebelum preprocessing; negatif sampling hanya dari piece lain di split yang sama.

## 9. Artefak

- `data/manifest_all_pieces.csv`
- `data/subset_pieces.csv`, `data/splits.csv`
- `data/processed/norm_stats.json`
- `data/processed/score_pages_{train,val,test}.npy` + `score_index.csv`
- `data/processed/audio_segments_{train,val,test}.npy` + `audio_index.csv`
- `data/processed/pairs_{train,val,test}.csv`
- `data/processed/preprocess_summary.json`

## 10. Hasil Verifikasi

- Total check: **27**, gagal: **0**
- Score tensor: `[384, 384, 1]` (float16)
- Audio tensor: `[100, 92]` (float16)
- Total halaman: **61**, segmen audio: **2844**, pasangan: **976**
