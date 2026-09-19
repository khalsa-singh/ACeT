# Data

`curated/` contains DataS1-DataS5 exactly once. `fixed_splits/` contains the locked seed-0 manuscript train/test partitions. `robustness_splits/` contains the full-order tables used to regenerate the changing-cohort partitions.

These are the canonical analysis-ready data locations. A small number of immutable producer folders retain path-local numeric input snapshots required to trace the original execution; those snapshots are identified in `MANIFEST.csv` and should not be treated as additional canonical DataS tables.

## DataS2 mouse-exposure target scale

The `AUCt` column in DataS2 and its fixed-split derivatives stores AUC0-672 h in units of `10^4 ng-h/mL` (raw AUC divided by `10^4`). To express a stored value in the `10^6 ng-h/mL` units used in the manuscript and figures, divide it by 100. For example, a stored value of `272.12` corresponds to `2.7212 x 10^6 ng-h/mL`.
