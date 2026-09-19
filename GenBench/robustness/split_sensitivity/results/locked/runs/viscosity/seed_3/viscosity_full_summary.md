# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 114.1

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 | 0.690972 | 6.08538 | 4.343   |   0.809247 |           19.2672 |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 | 0.730261 | 4.0635  | 3.11393 |   0.813591 |           53.942  |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | 0.816633 | 5.36684 | 3.56365 |   0.908064 |           79.8102 |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 | 0.811529 | 4.58765 | 3.38002 |   0.810436 |           93.3282 |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 | 0.448638 | 5.58699 | 4.0216  |   0.604221 |          113.406  |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 | 0.48748  | 7.54707 | 5.46829 |   0.670376 |          114.062  |                5 |
