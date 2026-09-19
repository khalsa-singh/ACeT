# Figure 2e current-pipeline summary

## Feature ablations

| condition   | condition_display   |   mean_test_r2 |   sd_test_r2 |   mean_test_rmse |   sd_test_rmse |   mean_test_mae |   sd_test_mae |   mean_test_spearman |   n_seeds |   condition_order |
|:------------|:--------------------|---------------:|-------------:|-----------------:|---------------:|----------------:|--------------:|---------------------:|----------:|------------------:|
| full        | Full                |       0.709571 |    0.0436437 |          5.03554 |       0.364904 |         3.26061 |     0.283826  |             0.874395 |         5 |                 0 |
| top2        | Top2 (kD, Plates)   |       0.566257 |    0.0957051 |          6.1397  |       0.644464 |         4.25249 |     0.504409  |             0.783167 |         5 |                 1 |
| ht_assays   | HT assays           |       0.510993 |    0.0421388 |          6.54308 |       0.277347 |         4.45924 |     0.180371  |             0.683414 |         5 |                 2 |
| no_kd       | No kD               |       0.285988 |    0.0194477 |          7.91147 |       0.107621 |         5.10259 |     0.0910456 |             0.552625 |         5 |                 3 |

## Paired tests versus full

| reference   | condition   | condition_display   |   paired_t_statistic |       raw_p |   mean_paired_r2_difference |   n_paired_seeds |      holm_p | significance   |
|:------------|:------------|:--------------------|---------------------:|------------:|----------------------------:|-----------------:|------------:|:---------------|
| full        | top2        | Top2 (kD, Plates)   |              2.82787 | 0.0474476   |                    0.143314 |                5 | 0.0474476   | *              |
| full        | ht_assays   | HT assays           |             29.3738  | 7.99768e-06 |                    0.198579 |                5 | 2.3993e-05  | ***            |
| full        | no_kd       | No kD               |             16.2726  | 8.34589e-05 |                    0.423584 |                5 | 0.000166918 | ***            |

## Learning curve

|   n_real |   percent_of_total_75 |   mean_test_r2 |   sd_test_r2 |   mean_test_rmse |   mean_test_mae |   n_seeds |
|---------:|----------------------:|---------------:|-------------:|-----------------:|----------------:|----------:|
|       16 |               21.3333 |       0.539284 |            0 |          6.35555 |         4.01708 |         1 |
|       21 |               28      |       0.601234 |            0 |          5.91284 |         3.98618 |         1 |
|       26 |               34.6667 |       0.636278 |            0 |          5.64704 |         4.09743 |         1 |
|       31 |               41.3333 |       0.6541   |            0 |          5.50696 |         3.81362 |         1 |
|       36 |               48      |       0.675264 |            0 |          5.33583 |         3.74141 |         1 |
|       42 |               56      |       0.598037 |            0 |          5.93649 |         4.23867 |         1 |
|       47 |               62.6667 |       0.630395 |            0 |          5.69253 |         3.88029 |         1 |
|       52 |               69.3333 |       0.740886 |            0 |          4.76631 |         3.08323 |         1 |

## Automated claim checks

- Monotonic non-decreasing mean learning curve: **False**
- Slope across last three learning points: 0.014285 R² per real row
- Full minus no-kD mean R²: 0.4236
- No-kD Holm-adjusted paired P: 0.000166918

Do not finalize the Results wording until these current outputs are reviewed.
