# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 51.4

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 | 0.145378 | 6.29898 | 3.9601  |   0.436003 |           5.83484 |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 | 0.60942  | 4.85248 | 3.28132 |   0.795925 |          24.3532  |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | 0.756958 | 4.88    | 3.59544 |   0.642711 |          37.8661  |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 | 0.655126 | 7.26903 | 4.70283 |   0.854748 |          43.7079  |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 | 0.835759 | 4.25831 | 3.31795 |   0.875685 |          51.0932  |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 | 0.60479  | 7.07383 | 4.29398 |   0.785962 |          51.3422  |                5 |
