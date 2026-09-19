# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 116.4

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | 0.296284 | 173.69  | 136.264  |   0.534953 |           15.1483 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 | 0.577587 | 140.352 | 107.75   |   0.564702 |           33.0112 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 | 0.441286 | 170.267 | 136.311  |   0.6845   |           61.3392 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 | 0.445625 | 209.205 | 151.446  |   0.676643 |           96.4392 |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 | 0.352863 | 174.986 | 143.98   |   0.703286 |          115.571  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 | 0.797945 | 111.122 |  82.1477 |   0.909091 |          116.31   |                5 |
