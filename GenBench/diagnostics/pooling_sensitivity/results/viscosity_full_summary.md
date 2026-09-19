# Viscosity Full Pooling Ablation Summary

- Head: `kan`
- Synthetic rows: 26
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 10
- Completed pooling variants: avg, avgmax, flatten, max
- Failed pooling variants: none
- Runtime seconds: 252.0

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |       r2 |    rmse |     mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|---------:|--------:|--------:|-----------:|------------------:|-----------------:|
| viscosity  | full   | avg       | 1        | cv_validation |  26 | 0.652206 | 5.87752 | 3.71292 |   0.761725 |          14.746   |              nan |
| viscosity  | full   | avg       | 2        | cv_validation |  26 | 0.553364 | 6.03217 | 4.53098 |   0.713772 |          28.2242  |              nan |
| viscosity  | full   | avg       | 3        | cv_validation |  26 | 0.293572 | 9.44114 | 5.71634 |   0.842466 |          44.9568  |              nan |
| viscosity  | full   | avg       | 4        | cv_validation |  26 | 0.801997 | 5.20279 | 3.71916 |   0.903303 |          60.1131  |              nan |
| viscosity  | full   | avg       | 5        | cv_validation |  26 | 0.803967 | 4.4472  | 3.36887 |   0.730137 |          67.0677  |              nan |
| viscosity  | full   | avg       | ensemble | heldout_test  |  23 | 0.740886 | 4.76631 | 3.08323 |   0.875927 |          67.4178  |                5 |
| viscosity  | full   | max       | 1        | cv_validation |  26 | 0.501249 | 7.03842 | 4.38685 |   0.739815 |          16.2761  |              nan |
| viscosity  | full   | max       | 2        | cv_validation |  26 | 0.585693 | 5.80976 | 4.33809 |   0.713626 |          30.6706  |              nan |
| viscosity  | full   | max       | 3        | cv_validation |  26 | 0.660033 | 6.54951 | 4.86462 |   0.835075 |          48.0787  |              nan |
| viscosity  | full   | max       | 4        | cv_validation |  26 | 0.80745  | 5.13064 | 3.49902 |   0.872024 |          64.3339  |              nan |
| viscosity  | full   | max       | 5        | cv_validation |  26 | 0.713573 | 5.37563 | 3.99775 |   0.744353 |          70.9     |              nan |
| viscosity  | full   | max       | ensemble | heldout_test  |  23 | 0.677825 | 5.31475 | 3.46772 |   0.830203 |          71.2174  |                5 |
| viscosity  | full   | flatten   | 1        | cv_validation |  26 | 0.792719 | 4.53747 | 2.91718 |   0.829652 |           7.41667 |              nan |
| viscosity  | full   | flatten   | 2        | cv_validation |  26 | 0.630231 | 5.4886  | 3.96728 |   0.707898 |          21.7535  |              nan |
| viscosity  | full   | flatten   | 3        | cv_validation |  26 | 0.696943 | 6.18376 | 4.74009 |   0.80209  |          28.3261  |              nan |
| viscosity  | full   | flatten   | 4        | cv_validation |  26 | 0.772725 | 5.57411 | 4.38143 |   0.835045 |          46.5843  |              nan |
| viscosity  | full   | flatten   | 5        | cv_validation |  26 | 0.603872 | 6.3218  | 4.33554 |   0.74024  |          52.6303  |              nan |
| viscosity  | full   | flatten   | ensemble | heldout_test  |  23 | 0.643398 | 5.5915  | 3.76419 |   0.797478 |          52.9538  |                5 |
| viscosity  | full   | avgmax    | 1        | cv_validation |  26 | 0.715407 | 5.31674 | 3.49123 |   0.835564 |           7.54967 |              nan |
| viscosity  | full   | avgmax    | 2        | cv_validation |  26 | 0.679816 | 5.10736 | 3.71986 |   0.768401 |          20.2105  |              nan |
| viscosity  | full   | avgmax    | 3        | cv_validation |  26 | 0.565404 | 7.40514 | 5.08178 |   0.791367 |          38.7697  |              nan |
| viscosity  | full   | avgmax    | 4        | cv_validation |  26 | 0.871995 | 4.18325 | 3.57491 |   0.859685 |          53.7013  |              nan |
| viscosity  | full   | avgmax    | 5        | cv_validation |  26 | 0.55524  | 6.69863 | 4.60232 |   0.716658 |          59.9848  |              nan |
| viscosity  | full   | avgmax    | ensemble | heldout_test  |  23 | 0.673177 | 5.35295 | 3.67598 |   0.823239 |          60.3508  |                5 |
