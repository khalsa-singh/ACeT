# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 59.4

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |         r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|-----------:|--------:|--------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 |  0.0658623 | 205.034 | 155.681 |   0.568328 |           10.8765 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 | -0.298313  | 247.137 | 175.916 |   0.393414 |           22.1607 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.123463  | 204.449 | 142.089 |   0.764211 |           38.7643 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.307365  | 172.422 | 144.963 |   0.434339 |           46.6034 |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.42923   | 178.753 | 140.021 |   0.609444 |           59.0876 |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.375319  | 228.029 | 170.839 |   0.563636 |           59.3405 |                5 |
