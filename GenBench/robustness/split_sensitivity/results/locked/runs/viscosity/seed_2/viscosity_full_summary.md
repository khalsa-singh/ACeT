# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 98.9

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 | 0.877489 | 3.22039 | 2.47574 |   0.885885 |           15.5916 |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 | 0.436309 | 6.74697 | 5.15285 |   0.760732 |           37.4592 |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | 0.669269 | 4.79958 | 3.60878 |   0.637314 |           55.0963 |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 | 0.636525 | 4.57979 | 3.81719 |   0.804699 |           84.7918 |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 | 0.784064 | 4.50482 | 3.06428 |   0.566329 |           98.1393 |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 | 0.49722  | 8.70402 | 6.16063 |   0.738819 |           98.8192 |                5 |
