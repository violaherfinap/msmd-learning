# Mermaid diagrams for baseline slides

Palette: navy `#0B1F3A`, pink `#E85D9E`, white `#FFFFFF`, black `#111111`.

| File | Slide | Orientation |
|---|---|---|
| `slide1_dorfer.mmd` | 1 — Dorfer full architecture (paper) | vertical, right panel |
| `slide2_dorfer_full.mmd` | 2 — Dorfer Task 1 + Task 2, full baseline | horizontal lanes, full slide |
| `slide3_task1_vertical.mmd` | 3 — Dorfer Task 1 (CCA + 4 blocks) | vertical, right panel |
| `slide4_task2_vertical.mmd` | 4 — Dorfer Task 2 (both encoders trainable) | vertical, right panel |

Render at [mermaid.live](https://mermaid.live) or with Mermaid CLI:

```powershell
npx @mermaid-js/mermaid-cli -i slide3_task1_vertical.mmd -o slide3.png -b white -w 2400
```

Recommended placement:
- Slides 1, 3, 4: diagram about 6.2 × 6.0 in (right 55% of a 16:9 slide).
- Slide 2: diagram about 12.5 × 6.2 in (full slide).
- Keep box text at least 14 pt after placement.

There is no frozen component in the baseline. Dashed arrows only mean weight initialisation from Task 1 to Task 2.
