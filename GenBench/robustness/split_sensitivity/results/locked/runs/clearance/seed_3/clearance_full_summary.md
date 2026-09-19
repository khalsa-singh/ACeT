# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 91.0

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|--------:|--------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 |  0.666076 | 140.927 | 105.014 |   0.845913 |           28.0777 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.194279 | 200.222 | 165.015 |   0.709263 |           46.346  |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.407396 | 204.243 | 157.039 |   0.635787 |           57.6752 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.573597 | 147.656 | 122.31  |   0.837944 |           76.1732 |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.596225 | 192.977 | 164.514 |   0.783045 |           90.6613 |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 | -0.353405 | 234.792 | 188.376 |   0.145455 |           90.982  |                5 |
