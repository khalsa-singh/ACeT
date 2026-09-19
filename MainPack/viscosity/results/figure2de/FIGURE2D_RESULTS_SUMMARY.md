# Figure 2d current-pipeline summary

- Training seeds: 0, 1, 2, 3, 4
- Permutation repeats per feature per seed: 10
- Dominant mean feature: **DLS kD** (0.8663 ΔR²)
- Seed-0 importance anchor max difference: 0.000001
- Pairwise tests: paired two-sided t-tests across seed-level mean importances; Holm–Šidák adjusted.

## Across-seed summary

| feature                             | feature_display   |   mean_importance |   sd_across_seeds |   n_seeds |   feature_order |
|:------------------------------------|:------------------|------------------:|------------------:|----------:|----------------:|
| DLS Interaction Parameter kD (mL/g) | DLS kD            |         0.866259  |         0.179389  |         5 |               0 |
| SE-UHPLC Main Peak Plates (EP)      | SE-UHPLC Plates   |         0.282136  |         0.140257  |         5 |               1 |
| AC-SINS λmax (nm)                   | AC-SINS Δλmax     |         0.182495  |         0.0620462 |         5 |               2 |
| SE-UHPLC Main Peak FWHM (min)       | SE-UHPLC FWHM     |         0.0413566 |         0.0376545 |         5 |               3 |

## Pairwise tests

| feature_1                           | feature_1_display   | feature_2                      | feature_2_display   |   paired_t_statistic |       raw_p |   holm_sidak_p | significance   |   n_paired_seeds |
|:------------------------------------|:--------------------|:-------------------------------|:--------------------|---------------------:|------------:|---------------:|:---------------|-----------------:|
| DLS Interaction Parameter kD (mL/g) | DLS kD              | SE-UHPLC Main Peak Plates (EP) | SE-UHPLC Plates     |             17.7991  | 5.85433e-05 |    0.000351208 | ***            |                5 |
| DLS Interaction Parameter kD (mL/g) | DLS kD              | AC-SINS λmax (nm)              | AC-SINS Δλmax       |             11.8034  | 0.00029486  |    0.00129455  | **             |                5 |
| DLS Interaction Parameter kD (mL/g) | DLS kD              | SE-UHPLC Main Peak FWHM (min)  | SE-UHPLC FWHM       |             12.201   | 0.000259044 |    0.00129455  | **             |                5 |
| SE-UHPLC Main Peak Plates (EP)      | SE-UHPLC Plates     | AC-SINS λmax (nm)              | AC-SINS Δλmax       |              2.12773 | 0.100464    |    0.100464    | ns             |                5 |
| SE-UHPLC Main Peak Plates (EP)      | SE-UHPLC Plates     | SE-UHPLC Main Peak FWHM (min)  | SE-UHPLC FWHM       |              4.75887 | 0.00891259  |    0.0177457   | *              |                5 |
| AC-SINS λmax (nm)                   | AC-SINS Δλmax       | SE-UHPLC Main Peak FWHM (min)  | SE-UHPLC FWHM       |              9.72202 | 0.000626755 |    0.00187909  | **             |                5 |
