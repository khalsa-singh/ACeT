# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 138.1

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |     rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|---------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | 0.249152 | 155.631  | 129.946  |   0.529601 |           27.3643 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 | 0.562212 | 147.142  | 122.976  |   0.793704 |           55.7853 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 | 0.412555 | 154.265  | 124.371  |   0.684343 |           89.8158 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 | 0.435521 | 219.497  | 182.842  |   0.729479 |          108.224  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 | 0.527833 | 174.643  | 135.474  |   0.759572 |          137.395  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 | 0.844129 |  97.5997 |  74.6767 |   0.927273 |          138.02   |                5 |
