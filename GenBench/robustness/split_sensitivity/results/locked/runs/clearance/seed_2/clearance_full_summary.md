# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 53.0

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |         r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|-----------:|--------:|--------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | -0.11862   | 244.879 | 184.923 |   0.519878 |           11.853  |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.082963  | 194.868 | 134.043 |   0.612563 |           20.3771 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.333886  | 219.632 | 177.695 |   0.563918 |           29.1446 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.439275  | 216.041 | 174.969 |   0.706532 |           37.5961 |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.0121091 | 231.118 | 168.534 |   0.510076 |           52.7034 |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.309055  | 199.611 | 153.412 |   0.578006 |           52.953  |                5 |
