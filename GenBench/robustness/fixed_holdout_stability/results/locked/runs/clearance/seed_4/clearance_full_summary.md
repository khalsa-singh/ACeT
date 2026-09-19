# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 145.0

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | 0.322433 | 161.309 | 136.585 |   0.648731 |           30.0175 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 | 0.272829 | 191.717 | 137.788 |   0.579832 |           49.7519 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 | 0.68837  | 145.113 | 109.275 |   0.783712 |           89.6745 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 | 0.630965 | 141.741 | 115.996 |   0.614292 |          113.242  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 | 0.460703 | 181.779 | 152.929 |   0.688503 |          144.295  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 | 0.725227 | 129.584 | 107.899 |   0.809091 |          144.93   |                5 |
