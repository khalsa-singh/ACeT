# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 102.4

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |        r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|----------:|--------:|--------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 | -0.441626 | 243.786 | 191.153 |   0.457741 |           14.2248 |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.63599  | 136.568 | 108.456 |   0.758178 |           58.0656 |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.57194  | 161.808 | 126.624 |   0.704875 |           71.2907 |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.426823 | 217.62  | 165.077 |   0.675972 |           90.3537 |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.171139 | 229.056 | 183.67  |   0.547643 |          102.13   |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.484466 | 187.218 | 147.732 |   0.709091 |          102.384  |                5 |
