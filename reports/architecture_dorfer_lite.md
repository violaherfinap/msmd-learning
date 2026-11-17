# Ablation: Dorfer-lite (bukan baseline)

> Dokumen ini menyimpan run reduced-capacity sebagai ablation. Baseline resmi memakai Dorfer penuh di `reports/architecture_dorfer.md`.

```mermaid
flowchart LR
    subgraph T1["Task 1 — Score–Audio Alignment"]
        direction LR
        SI["Score crop<br/>1 x 100 x 80"] --> SE["Score Encoder<br/>3 blok 16/32/64<br/>GAP -> 32d -> L2"]
        AI["Audio window<br/>1 x 92 x 42"] --> AE["Audio Encoder<br/>3 blok 16/32/64<br/>GAP -> 32d -> L2"]
        SE --> COS["Cosine similarity"]
        AE --> COS
        COS --> RL["Ranking hinge loss<br/>alpha = 0.7"]
    end

    subgraph T2["Task 2 — Performance Error Detection"]
        direction LR
        SC["Score crop referensi"] --> SE2["Score Encoder<br/>FROZEN"]
        AU["Audio performance"] --> AE2["Audio Encoder<br/>FINE-TUNE"]
        SE2 --> FU["Fusion<br/>x, y, x*y, abs(x-y)"]
        AE2 --> FU
        FU --> MLP["MLP 128"] --> OUT["correct / wrong pitch / missing / extra"]
    end

    SE -. "init weights" .-> SE2
    AE -. "init weights" .-> AE2
```

**Detail satu encoder:** 2× Conv3×3 + BN + ELU + MP2, diulang untuk 16 → 32 → 64 channel, lalu Conv1×1-32 + BN + GAP.

**Catatan:**
- Task 1 dilatih dari nol; Task 2 memakai bobot encoder Task 1.
- Task 2 membekukan Score Encoder dan hanya fine-tune Audio Encoder + fusion classifier.
- Tidak ada CCA layer pada profil Dorfer-lite.

## Angka ablation (dari branch `feature/baseline-dorfer-lite`)

| Metrik | Dorfer-lite | Dorfer full |
|---|---:|---:|
| Task 1 val MRR | 0.0086 | 0.0100 |
| Task 1 snippet R@1 | 0.15% | 0.10% |
| Task 1 snippet MRR | 0.0087 | 0.0075 |
| Task 1 page-conditioned R@1 | 27.5% | 26.0% |
| Task 1 5-candidate R@1 | 25.6% | 27.3% |
| Task 1 piece ID Rk@1 | 54.2% | 41.2% |
| Task 2 4-class accuracy | 99.0% | (lihat summary) |

Angka ini menunjukkan kapasitas bukan faktor pembatas pada 17 train pieces: model kecil dan model penuh memberi hasil sebanding, dan model penuh justru lebih cepat overfit. Karena itu baseline resmi tetap memakai arsitektur penuh untuk keperluan perbandingan, sedangkan profil lite hanya menjadi ablation.
