# Figure 2c corrected values

## Caption clause

**c, Model comparison under the matched endpoint workflow. Bars show held-out
test metrics, diamonds show five-fold cross-validation means, and open circles
show individual cross-validation folds for ACeT, ridge regression, support-vector
regression, and random forest.**

## Approved held-out values

| Model | Test R2 | Test RMSE (cP) | Test MAE (cP) |
|---|---:|---:|---:|
| ACeT | 0.7409 | 4.7663 | 3.0832 |
| Ridge | 0.5265 | 6.4434 | 5.0883 |
| SVR | 0.5360 | 6.3782 | 4.2363 |
| Random forest | 0.4787 | 6.7604 | 4.6130 |

## Approved CV summary

| Model | CV R2 mean +/- SD | CV RMSE mean +/- SD | CV MAE mean +/- SD |
|---|---:|---:|---:|
| ACeT | 0.6210 +/- 0.2115 | 6.2002 +/- 1.9173 | 4.2097 +/- 0.9444 |
| Ridge | 0.5328 +/- 0.1283 | 7.0101 +/- 0.7040 | 5.5395 +/- 0.5722 |
| SVR | 0.4350 +/- 0.0800 | 7.7953 +/- 0.9835 | 5.3117 +/- 0.7878 |
| Random forest | 0.6780 +/- 0.1628 | 5.7044 +/- 1.3415 | 3.9728 +/- 0.5572 |

## Interpretation constraint

ACeT has the strongest held-out performance, but random forest has the highest
mean CV R2. The revised text must not say that ACeT outperformed every comparator
in both CV and held-out testing. A defensible interpretation is that ACeT provided
the strongest independent held-out result, whereas random forest showed greater
split sensitivity.
