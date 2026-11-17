# Arsitektur Baseline Dorfer

```mermaid
flowchart LR
    subgraph T1["Task 1 — Score–Audio Alignment"]
        direction LR
        SI["Score crop<br/>1 x 100 x 80"] --> SE["Sheet Encoder<br/>4 blok 24/48/96/96<br/>Conv1x1-32 + GAP"]
        AI["Audio window<br/>1 x 92 x 42"] --> AE["Audio Encoder<br/>4 blok 24/48/96/96<br/>Conv1x1-32 + GAP"]
        SE --> CCA["CCA layer<br/>32d"]
        AE --> CCA
        CCA --> RL["Cosine + ranking hinge<br/>margin 0.7"]
    end

    subgraph T2["Task 2 — Performance Error Detection"]
        direction LR
        SC["Score crop referensi"] --> SE2["Sheet Encoder<br/>TRAINABLE"]
        AU["Audio performance"] --> AE2["Audio Encoder<br/>TRAINABLE"]
        SE2 --> FU["Fusion<br/>x, y, x*y, abs(x-y)"]
        AE2 --> FU
        FU --> MLP["MLP 128"] --> OUT["4 kelas"]
    end

    SE -. "init weights" .-> SE2
    AE -. "init weights" .-> AE2

    classDef navyBox fill:#0B1F3A,stroke:#0B1F3A,color:#FFFFFF,stroke-width:1.5px;
    classDef pinkBox fill:#E85D9E,stroke:#0B1F3A,color:#111111,stroke-width:1.5px;
    classDef whiteNavy fill:#FFFFFF,stroke:#0B1F3A,color:#111111,stroke-width:1.5px;
    classDef whitePink fill:#FFFFFF,stroke:#E85D9E,color:#111111,stroke-width:1.5px;
    classDef outputBox fill:#0B1F3A,stroke:#E85D9E,color:#FFFFFF,stroke-width:2px;

    class SI,SE,SC,SE2 navyBox;
    class AI,AE,AU,AE2 pinkBox;
    class CCA whiteNavy;
    class RL,FU,MLP whitePink;
    class OUT outputBox;
```

**Satu encoder:** 2× Conv3×3 + BN + ELU + MP2, diulang untuk 24 → 48 → 96 → 96
channel, lalu Conv1×1-32 + BN + GAP → 32-d.

**Catatan:**
- Task 1 dilatih dari nol, CCA aktif, warmup 200 step, running covariance `alpha=0.05`.
- Task 2 memakai bobot Task 1 sebagai inisialisasi dan melatih kedua encoder.
- Tidak ada komponen frozen di baseline ini.
- Task 2 tidak memakai CCA; classifier memakai output trunk 32-d.
