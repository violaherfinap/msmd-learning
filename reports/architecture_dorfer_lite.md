# Ablation: Dorfer-lite (bukan baseline)

Baseline resmi memakai Dorfer penuh (`reports/architecture_dorfer.md`). Profil di
bawah ini hanya ablation reduced-capacity dari branch
`feature/baseline-dorfer-lite`: 3 blok 16/32/64, tanpa CCA, dan Task 2 dengan
sheet encoder dibekukan.

| Metrik | Dorfer-lite | Dorfer full |
|---|---:|---:|
| Task 1 val MRR | 0.0086 | 0.0100 |
| Task 1 snippet R@1 | 0.15% | 0.10% |
| Task 1 snippet MRR | 0.0087 | 0.0075 |
| Task 1 page-conditioned R@1 | 27.5% | 26.0% |
| Task 1 5-candidate R@1 | 25.6% | 27.3% |
| Task 1 piece ID Rk@1 | 54.2% | 41.2% |
| Task 2 4-class accuracy | 99.0% | 99.54% |

Hasil sebanding. Ini menunjukkan kapasitas bukan faktor pembatas pada 17 train
pieces; model penuh justru lebih cepat overfit. Angka lite diambil dari branch
tersebut karena artefaknya tidak ikut di-commit di branch ini.
