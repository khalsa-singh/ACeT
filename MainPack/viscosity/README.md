# High-concentration viscosity

The main model uses kD, SEC main-peak plates, AC-SINS spectral shift and SEC main-peak FWHM. The fixed split is 52 development / 23 test antibodies. Numeric inputs and the model builder are in `code/source/shared`; the CV-selected reference head is KAN.

From the repository root:

```bash
python acet.py train viscosity --check-inputs
python acet.py train viscosity --output-dir reruns/viscosity_training
python acet.py figure2b-data --output-dir reruns/viscosity_bootstrap
python acet.py figure2c-tables --output-dir reruns/viscosity_comparison_tables
python acet.py figure2de-preflight --output-dir reruns/viscosity_figure2de_inputs
python acet.py figure2de-train --output-dir reruns/viscosity_figure2de_training
python acet.py figure2de-plots --output-dir reruns/viscosity_figure2de_plots
```

Training commands require the study environment; `figure2de-plots` requires MATLAB. The reference result is approximately R² = 0.740886, RMSE = 4.7663 cP and MAE = 3.0832 cP. The bootstrap producer uses the archived held-out predictions. `results/figure2de` contains importance, paired tests, learning-curve and feature-ablation records.

See `SensitivityAnalyses/DevelopmentOnlyFeatureSelection/viscosity` for development-only assay ranking and the selected-panel sensitivity, and the root README for complete schemas and benchmark commands.
