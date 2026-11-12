# Task 1 experiment comparison

| Run | Experiment | Val MRR | Val protocol | Page R@1 | Page R@5 | 5-cand R@1 | 5-cand MRR | Snippet MRR | Minutes |
|---|---|---|---|---|---|---|---|---|---|
| t1_c1_baseline | dual_encoder | 0.0078 | snippet | 0.2849 | 0.9610 | 0.3085 | 0.5668 | 0.0122 | 31.7 |
| t1_c2_tuned | dual_encoder | 0.0096 | snippet | 0.2587 | 0.9454 | 0.2992 | 0.5494 | 0.0094 | 37.8 |
| t1_c3_architecture | dual_encoder | 0.0139 | snippet | 0.2520 | 0.9590 | 0.2584 | 0.5338 | 0.0080 | 67.8 |
| t1_c4_ranking_head | ranking_head | 0.5410 | 5-cand | 0.3303 | 0.9480 | 0.3527 | 0.5914 |  | 4.3 |
