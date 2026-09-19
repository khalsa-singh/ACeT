# Combined regression robustness summary

Two complementary questions are reported separately:

1. **Fixed-holdout training-seed stability:** the manuscript train/test split is unchanged and only the training seed varies; mean ± SD is therefore reported across runs.
2. **Composite-split sensitivity:** the held-out antibodies change across six partitions; error coverage and parity are displayed descriptively, and no aggregate R² is reported.

## Fixed-holdout metrics

| endpoint   |   n_training_seeds | seed_range   |   mean_r2 |   sd_r2 |   min_r2 |   max_r2 |   mean_rmse |   sd_rmse |   min_rmse |   max_rmse |   mean_mae |   sd_mae |   min_mae |   max_mae |   mean_nrmse |   sd_nrmse |   min_nrmse |   max_nrmse |   mean_nmae |   sd_nmae |   min_nmae |   max_nmae |   mean_spearman |   sd_spearman |   min_spearman |   max_spearman |
|:-----------|-------------------:|:-------------|----------:|--------:|---------:|---------:|------------:|----------:|-----------:|-----------:|-----------:|---------:|----------:|----------:|-------------:|-----------:|------------:|------------:|------------:|----------:|-----------:|-----------:|----------------:|--------------:|---------------:|---------------:|
| viscosity  |                  5 | 0-4          |    0.7096 |  0.0436 |   0.6338 |   0.7409 |      5.0355 |    0.3649 |     4.7663 |     5.6663 |     3.2606 |   0.2838 |    2.9231 |    3.6741 |       0.2959 |     0.0214 |      0.2801 |      0.3330 |      0.1916 |    0.0167 |     0.1718 |     0.2159 |          0.8744 |        0.0208 |         0.8485 |         0.9051 |
| clearance  |                  5 | 0-4          |    0.7839 |  0.0684 |   0.7008 |   0.8514 |    113.7621 |   18.1704 |    95.2812 |   135.2230 |    88.2715 |  14.6264 |   74.6767 |  107.8990 |       0.1512 |     0.0242 |      0.1267 |      0.1797 |      0.1173 |    0.0194 |     0.0993 |     0.1434 |          0.8982 |        0.0504 |         0.8091 |         0.9273 |

## Table S13 replacement regression rows

| Endpoint                           | Repeat basis                                                                                       | Summary                                                                                                                                                                            |
|:-----------------------------------|:---------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Viscosity — fixed holdout          | Five independent training seeds on the unchanged 52/23 manuscript split; fixed KAN head            | Held-out R² = 0.710 ± 0.044; RMSE = 5.04 ± 0.36 cP; MAE = 3.26 ± 0.28 cP; Spearman ρ = 0.874 ± 0.021.                                                                              |
| Viscosity — split sensitivity      | Six composite-stratified train/test partitions (seeds 0-5); fixed KAN head and current workflow    | Across 138 held-out prediction instances, 94.9% were within twofold and 100.0% were within threefold error. Test membership changed by partition; no aggregate R² is reported.     |
| Mouse exposure — fixed holdout     | Five independent training seeds on the unchanged 42/11 manuscript split; fixed spline head         | Held-out R² = 0.784 ± 0.068; nRMSE = 0.151 ± 0.024; nMAE = 0.117 ± 0.019; Spearman ρ = 0.898 ± 0.050.                                                                              |
| Mouse exposure — split sensitivity | Six composite-stratified train/test partitions (seeds 0-5); fixed spline head and current workflow | Across 66 held-out prediction instances, 51.5% were within 15% and 74.2% were within 30% error. Each partition used a different 11-antibody test set; no aggregate R² is reported. |
