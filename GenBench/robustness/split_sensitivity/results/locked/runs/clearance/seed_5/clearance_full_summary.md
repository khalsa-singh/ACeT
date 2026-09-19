# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 53.2

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |    rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|--------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 |  0.436348 | 188.747 | 153.509  |   0.713631 |           9.46657 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.563608 | 178.677 | 126.446  |   0.696722 |          22.2971  |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.649119 | 176.496 | 133.502  |   0.765857 |          36.1225  |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.271124 | 211.502 | 151.976  |   0.555361 |          43.7605  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 | -0.281091 | 220.26  | 175.049  |   0.454697 |          52.9573  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.707594 | 122.55  |  95.9798 |   0.863636 |          53.2129  |                5 |
