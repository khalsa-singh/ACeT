# Mouse intravenous exposure

The main four-assay model uses Heparin_RT, Heparin_pB_buffer, BVP_high and poly_D_lysine with a fixed 42/11 development/test partition and the CV-selected spline head. The target is AUCt, a measure of systemic exposure; filenames retaining “clearance” identify the original PK workflow, not a different target.

```bash
python acet.py demo --formats svg,png,jpg --output-dir reruns/exposure_demo
python acet.py train mouse_exposure --check-inputs
python acet.py train mouse_exposure --output-dir reruns/exposure_training
python acet.py figure3b-data --output-dir reruns/exposure_augmentation_arrays
python acet.py figure3-plots --output-dir reruns/figure3
```

Run from the repository root. The last command requires MATLAB. The stored primary predictions give approximately R² = 0.810019 and nRMSE = 0.143226. AUCt is stored as AUC0–672h/10⁴; divide by 100 for the 10⁶ ng·h/mL plotting scale.

The correct error-CDF source is `FigureSources/main/Figure_3/panel_a/data/figure3a_current_parity_cdf.mat`, with the associated prediction table. The demonstration preserves its ordered-error convention. The full Figure 3b plotting arrays are available separately from the primary result; the original unaugmented head-selection log/checkpoints are not in the archive.

`SensitivityAnalyses/DevelopmentOnlyFeatureSelection/mouse_exposure` contains the separately selected-panel MLP analysis. Its inputs and settings are not interchangeable with the main spline model. See the root guide for SHAP training, matched comparators, pooling and robustness.
