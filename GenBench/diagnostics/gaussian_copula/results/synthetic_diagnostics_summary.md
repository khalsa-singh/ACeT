# Gaussian Copula real-vs-synthetic diagnostics

Generated: 2026-06-30T23:18:39

Scope: SDV Gaussian Copula diagnostic samples were fit only on seed0 training rows for each endpoint. No ACeT model, TensorFlow training, or `model.fit` call was used.

## Summary metrics

| endpoint   | train_path                                                                                                                                                     |   real_training_n |   synthetic_n | synthetic_ratio   |   mean_absolute_correlation_difference |   maximum_absolute_correlation_difference |   maximum_ks_statistic |   mean_ks_statistic |   mean_wasserstein_distance |   pca_synthetic_inside_real_pc_range_fraction |   standardized_synthetic_to_real_pairwise_distance_ratio |   pca_pc1_variance |   pca_pc2_variance |
|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|--------------:|:------------------|---------------------------------------:|------------------------------------------:|-----------------------:|--------------------:|----------------------------:|----------------------------------------------:|---------------------------------------------------------:|-------------------:|-------------------:|
| viscosity  | Data/fixed_splits/viscosity/DataS1_viscosity_seed0_train.csv |                52 |            26 | 1:2               |                                  0.106 |                                     0.499 |                  0.346 |               0.246 |                     120.642 |                                         0.923 |                                                    1.115 |              0.568 |              0.220 |
| clearance  | Data/fixed_splits/mouse_exposure/DataS2_clearance_seed0_train.csv |                42 |            84 | 2:1               |                                  0.172 |                                     0.400 |                  0.190 |               0.140 |                       7.220 |                                         0.857 |                                                    1.038 |              0.581 |              0.196 |

##  interpretation

- For viscosity, synthetic points largely overlap the real training pca range in the first two principal components (0.92 of synthetic rows inside the real PC1/PC2 range). The marginal distributions are broadly preserved; the largest KS statistic is 0.35. The mean absolute Spearman correlation difference is 0.11, so pairwise rank correlations are closely preserved. Median nearest-neighbor distances indicate that synthetic-to-real distances are larger than real-to-real distances (synthetic/real median ratio 1.85). These diagnostics are descriptive and limited by the small endpoint-specific training sets.
- For clearance, synthetic points largely overlap the real training pca range in the first two principal components (0.86 of synthetic rows inside the real PC1/PC2 range). The marginal distributions are broadly preserved; the largest KS statistic is 0.19. The mean absolute Spearman correlation difference is 0.17, so pairwise rank correlations are closely preserved. Median nearest-neighbor distances indicate that synthetic-to-real distances are comparable to real-to-real distances (synthetic/real median ratio 1.05). These diagnostics are descriptive and limited by the small endpoint-specific training sets.

Overall interpretation: these diagnostics support whether the synthetic rows occupy the same broad assay manifold as the real training rows, but they should not be treated as proof of perfect generative fidelity. The small viscosity and clearance training sets limit the precision of KS, Wasserstein, correlation, and nearest-neighbor summaries.

## Files

Each endpoint has PCA, correlation, marginal-distribution, nearest-neighbor, sample, and statistic outputs saved in this directory as CSV plus PNG/PDF figures where requested.
