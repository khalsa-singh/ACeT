# Final supplementary-figure captions

## Figure S1

**Figure S1. Viscosity robustness under fixed-holdout and alternate-split evaluation.** **a,** Absolute log10-error cumulative distributions from five independent training seeds evaluated on the unchanged 23-antibody manuscript holdout; thin lines show individual seeds and the thick line shows their mean. **b,** Corresponding descriptive error distributions across six composite-stratified train/test partitions (split seeds 0–5), for which held-out membership changes by partition. Twofold and threefold reference limits are shown. Mean ± s.d. fixed-holdout metrics are reported in Supplementary Table S13; no aggregate R² is assigned to the changing-cohort analysis.

## Figure S2

**Figure S2. Viscosity parity under fixed-holdout and alternate-split evaluation.** **a,** Predictions for the same 23 held-out antibodies across five independent training seeds; faint points show individual runs, open circles and vertical bars show the across-seed mean ± s.d., blue diamonds show seed 0, and the dashed line indicates unity. **b,** Held-out predictions across six composite-stratified partitions; gray circles show alternate partitions and blue diamonds show the manuscript partition. Dotted and solid reference lines mark 15 and 20 cP, respectively.

## Figure S3

**Figure S3. Mouse-exposure robustness under fixed-holdout and alternate-split evaluation.** **a,** Absolute-percentage-error cumulative distributions from five independent training seeds evaluated on the unchanged 11-antibody manuscript holdout; thin lines show individual seeds and the thick line shows their mean. **b,** Corresponding descriptive error distributions across six composite-stratified train/test partitions with different 11-antibody holdouts. Fifteen-percent and 30% error limits are shown. Mean ± s.d. fixed-holdout metrics are reported in Supplementary Table S13; no aggregate R² is assigned to the changing-cohort analysis.

## Figure S4

**Figure S4. Mouse-exposure parity under fixed-holdout and alternate-split evaluation.** **a,** Predictions for the same 11 held-out antibodies across five independent training seeds; faint points show individual runs, open circles and vertical bars show the across-seed mean ± s.d., orange circles show seed 0, and the dashed line indicates unity. **b,** Held-out predictions across six composite-stratified partitions; gray circles show alternate partitions and orange circles show the manuscript partition. Dotted reference lines mark the 3.9 × 10⁶ ng·h mL⁻¹ threshold used for visual stratification.

## Figure S5

**Figure S5. Clinical-outcome robustness on the temporally external cohort.** **a,** External-cohort balanced accuracy, Approved sensitivity, and Terminated specificity across internal split seeds 0–10. **b,** Empirical cumulative distribution of pooled external negative log-probability assigned to the true class. **c,** Maximum change in external balanced accuracy after leaving out one external antibody, shown across split seeds. Head selection was based on internal cross-validation; the external cohort was not used for model selection.

## Figure S6

**Figure S6. HIC retention-time prediction using assays-only inputs.** Bagged out-of-fold parity for ACeT on the Bailly et al. dataset (n = 152). Predictions were averaged across 10 repeats of fivefold cross-validation; the dashed line indicates unity. Continuous and 30-min triage metrics are reported in Supplementary Tables S2 and S12.

## Figure S7

**Figure S7. HIC retention-time prediction using patch descriptors only.** **a,** Bagged out-of-fold parity for patch-only ACeT. **b,** Fold-matched Bailly patch-linear refit. Predictions were evaluated on the same 152 antibodies using repeated out-of-fold procedures; dashed lines indicate unity.

## Figure S8

**Figure S8. Binary HIC developability triage at 30 min.** Confusion matrices compare assays-only ACeT, the fold-matched Bailly refit, patch-only ACeT, and the corresponding Bailly refit using bagged out-of-fold predictions. Rows are observed classes and columns are predicted classes; Good denotes HIC retention time ≤30 min and Poor denotes >30 min. Cells show counts and row percentages.

## Figure S9

**Figure S9. Relationship between SEC monomer peak-shape metrics and high-concentration viscosity.** **a,** Viscosity versus SE-UHPLC main-peak plate count. **b,** Viscosity versus SE-UHPLC main-peak FWHM in the 75-antibody viscosity dataset (DataS1).

## Figure S10

**Figure S10. Gaussian-Copula real-versus-synthetic diagnostics for the regression endpoints.** Top row, viscosity; bottom row, mouse clearance. For each endpoint, the panels show principal-component overlap between real training rows and synthetic rows, feature-wise Kolmogorov–Smirnov statistics summarizing marginal differences, absolute differences between real and synthetic Spearman-correlation matrices, and nearest-neighbor distances in standardized feature space. These diagnostics are descriptive and assess broad overlap rather than exact reproduction; numerical summaries are reported in Supplementary Table S10.

## Figure S11

**Figure S11. Training and validation loss curves for the main regression models.** **a,** Viscosity. **b,** Mouse clearance. Thin lines show individual fold histories for training and validation loss, and thicker lines show the mean among folds contributing at each epoch; fold-specific early stopping produces differing curve lengths. Curves correspond to the main global-average-pooling configuration and the endpoint-specific cross-validation-selected prediction head.
