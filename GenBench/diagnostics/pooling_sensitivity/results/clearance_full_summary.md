# Clearance Full Pooling Ablation Summary

- Head: `spline`
- Synthetic rows: 84
- Fold count run: 5
- Max epochs: 1000
- Early-stopping patience: 20
- Completed pooling variants: avg, avgmax, flatten, max
- Failed pooling variants: none
- Runtime seconds: 234.8

Smoke metrics are engineering checks only and are not manuscript evidence.

## Metrics

| endpoint   | mode   | pooling   | fold     | split         |   n |          r2 |    rmse |      mae |   spearman |   elapsed_seconds |   ensemble_folds |
|:-----------|:-------|:----------|:---------|:--------------|----:|------------:|--------:|---------:|-----------:|------------------:|-----------------:|
| clearance  | full   | avg       | 1        | cv_validation |  34 |  0.296284   | 173.691 | 136.264  |   0.534953 |          17.4641  |              nan |
| clearance  | full   | avg       | 2        | cv_validation |  34 |  0.577588   | 140.352 | 107.75   |   0.564702 |          34.2987  |              nan |
| clearance  | full   | avg       | 3        | cv_validation |  34 |  0.441285   | 170.267 | 136.311  |   0.6845   |          51.1105  |              nan |
| clearance  | full   | avg       | 4        | cv_validation |  33 |  0.445625   | 209.205 | 151.446  |   0.676643 |          69.8425  |              nan |
| clearance  | full   | avg       | 5        | cv_validation |  33 |  0.352863   | 174.986 | 143.98   |   0.703286 |          79.4402  |              nan |
| clearance  | full   | avg       | ensemble | heldout_test  |  11 |  0.797945   | 111.122 |  82.1477 |   0.909091 |          79.7719  |                5 |
| clearance  | full   | max       | 1        | cv_validation |  34 |  0.385636   | 162.289 | 123.288  |   0.602185 |          18.7632  |              nan |
| clearance  | full   | max       | 2        | cv_validation |  34 |  0.319411   | 178.153 | 122.576  |   0.521553 |          30.0533  |              nan |
| clearance  | full   | max       | 3        | cv_validation |  34 |  0.549593   | 152.876 | 119.93   |   0.702385 |          38.9569  |              nan |
| clearance  | full   | max       | 4        | cv_validation |  33 |  0.402936   | 217.11  | 168.106  |   0.768343 |          45.1687  |              nan |
| clearance  | full   | max       | 5        | cv_validation |  33 |  0.372395   | 172.325 | 137.887  |   0.73083  |          53.2961  |              nan |
| clearance  | full   | max       | ensemble | heldout_test  |  11 |  0.707933   | 133.6   | 110.297  |   0.872727 |          53.5455  |                5 |
| clearance  | full   | flatten   | 1        | cv_validation |  34 |  0.184165   | 187.016 | 135.815  |   0.530827 |           7.32334 |              nan |
| clearance  | full   | flatten   | 2        | cv_validation |  34 |  0.391657   | 168.432 | 121.173  |   0.557628 |          15.4464  |              nan |
| clearance  | full   | flatten   | 3        | cv_validation |  34 |  0.511247   | 159.25  | 125.085  |   0.682103 |          29.8121  |              nan |
| clearance  | full   | flatten   | 4        | cv_validation |  33 |  0.623111   | 172.495 | 139.32   |   0.847517 |          41.3568  |              nan |
| clearance  | full   | flatten   | 5        | cv_validation |  33 | -0.00228351 | 217.772 | 176.016  |   0.716675 |          48.617   |              nan |
| clearance  | full   | flatten   | ensemble | heldout_test  |  11 |  0.578894   | 160.421 | 115.529  |   0.772727 |          48.8633  |                5 |
| clearance  | full   | avgmax    | 1        | cv_validation |  34 |  0.161828   | 189.559 | 141.876  |   0.495989 |           7.28324 |              nan |
| clearance  | full   | avgmax    | 2        | cv_validation |  34 |  0.362277   | 172.452 | 125.27   |   0.555182 |          16.9135  |              nan |
| clearance  | full   | avgmax    | 3        | cv_validation |  34 |  0.482264   | 163.904 | 137.529  |   0.677518 |          28.1711  |              nan |
| clearance  | full   | avgmax    | 4        | cv_validation |  33 |  0.512298   | 196.222 | 143.377  |   0.781976 |          37.7274  |              nan |
| clearance  | full   | avgmax    | 5        | cv_validation |  33 | -0.0319108  | 220.967 | 173.285  |   0.66689  |          52.2335  |              nan |
| clearance  | full   | avgmax    | ensemble | heldout_test  |  11 |  0.631745   | 150.017 | 126.695  |   0.790909 |          52.5772  |                5 |
