# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 112.1

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 |  0.540329 | 6.13586 | 4.45766 |   0.811495 |           34.9456 |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 |  0.538576 | 5.7917  | 3.47662 |   0.824315 |           52.0452 |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | -0.336752 | 8.05505 | 5.33075 |   0.740709 |           68.6762 |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 |  0.773307 | 4.75701 | 2.95698 |   0.913596 |           94.0905 |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 |  0.489633 | 5.57086 | 3.8835  |   0.624294 |          111.492  |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 |  0.581311 | 8.28062 | 5.76624 |   0.676749 |          112.022  |                5 |
