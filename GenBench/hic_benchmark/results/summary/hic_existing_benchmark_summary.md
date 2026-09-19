# HIC existing benchmark summary

Source: archived HIC OOF prediction metrics and repeat-level performance tables.

| Summary item | n | R2 | Pearson r2 | Spearman rho | RMSE | MAE | 30-min balanced accuracy | 30-min MCC | Repeat R2 mean +/- SD | Repeat MCC mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACeT assays-only OOF performance | 152 | 0.796 | 0.825 | 0.881 | 4.995 | 3.256 | 0.891 | 0.777 | 0.631 +/- 0.085 | 0.773 +/- 0.057 |
| ACeT assays+patch OOF performance | 152 | 0.756 | 0.786 | 0.875 | 5.459 | 3.365 | 0.895 | 0.787 | 0.608 +/- 0.072 | 0.769 +/- 0.042 |
| ACeT patch-only OOF performance | 152 | 0.545 | 0.585 | 0.677 | 7.451 | 5.019 | 0.823 | 0.652 | 0.466 +/- 0.058 | 0.644 +/- 0.036 |
| Bailly fold-matched refit baseline | 152 | 0.506 | 0.507 | 0.652 | 7.763 | 6.444 | 0.777 | 0.550 | 0.503 +/- 0.016 | 0.540 +/- 0.012 |
| Bailly published equation reference | 152 | -0.267 | 0.208 | 0.467 | 12.435 | 10.722 | 0.480 | -0.063 |  +/-  |  +/-  |

Fold-matching assessment: the HIC baseline appears fold-matched for the `bailly_refit` comparison because the shipped JSON files report `analysis_mode=oof_cv`, `cv_repeats=10`, `n_splits=5`, and repeat-level `bailly_refit_*` metrics. The published Bailly equation reference is not a refit baseline; it is a fixed reference equation evaluated on the same n=152 table.
