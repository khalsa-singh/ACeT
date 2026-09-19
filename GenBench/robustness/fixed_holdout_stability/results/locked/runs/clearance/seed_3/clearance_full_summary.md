# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 126.0

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |    rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|--------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | -0.319016 | 247.758 | 166.828  |   0.622983 |           29.8945 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.397574 | 165.804 | 139.841  |   0.62672  |           56.8436 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.473451 | 194.694 | 159.593  |   0.668654 |           78.8439 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.23763  | 184.684 | 157.209  |   0.551061 |          103.492  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.530202 | 166.451 | 124.203  |   0.720889 |          125.324  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.700794 | 135.223 |  99.4561 |   0.918182 |          125.944  |                5 |
