# Assay-selection and comparator methods

## Defined viscosity candidate set

The development-only screen uses the assay definitions in `DevelopmentOnlyFeatureSelection/viscosity/reference/candidate_definition.csv` and the original 52-antibody development membership. It ranks the candidates by absolute Pearson correlation with viscosity and retains signed r and r². No held-out file is read by the screen producer. The complete set, rather than only its leading rows, is included. The selected model retains its original four inputs in canonical model order.

## Mouse-exposure development-only pipeline

The source cohort contains 55 antibody rows. The fixed 11 test identities are retained. The two specified training target-outlier rows are removed to obtain 42 development rows. On that development set, feature values outside per-column 1.5×IQR limits are treated as missing. Iterative chained-equations imputation uses a maximum of ten iterations with `sample_posterior=False`; development AUCt is available as auxiliary information and remains unchanged. This produces one completed feature matrix, not pooled inference from multiple imputed datasets.

Assays are ranked by absolute Pearson correlation with development AUCt. On the four selected features, a development-fitted normal-output QuantileTransformer identifies transformed values beyond ±5; those cells are treated as missing on the feature scale and re-imputed without AUCt. The raw 11-row holdout is exported unchanged. Settings and selected MLP head are recorded beside the source. The archived held-out metric summary corresponds to that analysis, not to the main spline model.

## Comparator search

The complete saved trial ledger and parameter lock are retained. Each model class is tuned separately using the cached five development folds: highest mean R², then lowest mean RMSE and model complexity. The search uses development data only. A preceding fixed-control preflight also evaluates the reference holdout; the tuned-model evaluation follows locking. Reporting a class's highest held-out performance is descriptive across the three prespecified classes, not a new class-selection rule.

Per endpoint, the finite search evaluated 1,320 Ridge, 220 RBF-SVR and 150 random-forest configurations. The archived predictions are the numerical reference. Repeated fitting in another environment creates a new run, not a replacement of that evidence.
