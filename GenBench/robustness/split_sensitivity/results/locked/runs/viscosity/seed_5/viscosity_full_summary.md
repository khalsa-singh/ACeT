# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg
- Failed pooling variants: none
- Runtime seconds: 47.9

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 | 0.5206   | 9.99669 | 6.75411 |   0.875727 |           11.5564 |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 | 0.894126 | 3.35403 | 2.58044 |   0.682723 |           16.524  |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | 0.526877 | 4.76912 | 3.31605 |   0.750428 |           27.4552 |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 | 0.743577 | 4.13093 | 2.60676 |   0.846799 |           40.6951 |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 | 0.653051 | 6.22488 | 4.09658 |   0.737203 |           47.6066 |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 | 0.448591 | 6.302   | 4.90931 |   0.533729 |           47.8447 |                5 |
