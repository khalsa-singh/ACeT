# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 205.3

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |     rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|---------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | 0.434528  | 171.604  | 141.867  |   0.619019 |           23.5726 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 | 0.606594  | 149.218  | 120.458  |   0.739473 |           89.7948 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 | 0.480881  | 160.561  | 123.977  |   0.74511  |          136.281  |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 | 0.597195  | 165.105  | 136.033  |   0.784891 |          169.969  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 | 0.0688443 | 208.189  | 152.135  |   0.540134 |          204.52   |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 | 0.851446  |  95.2812 |  77.1779 |   0.927273 |          205.255  |                5 |
