# Figure 2c matched viscosity comparison

The numerical source is `FigureSources/main/Figure_2/panel_c/data/figure2c_corrected_baselines.mat`. It contains five-fold metrics and the held-out metrics for ACeT, Ridge, SVR and random forest in that order.

From the root, `python acet.py figure2c-tables --output-dir reruns/figure2c` validates and reconstructs the CSV/MAT plot-source tables. The MATLAB panel sources read the `figure2c_corrected_baselines.mat` format. A separate development-CV-tuned benchmark is under `SensitivityAnalyses/TunedComparatorStressTest` and Table S16; it does not overwrite this fixed-reference comparison.

The numeric summary and caption values are in this directory's companion documentation. Final EPS files are in `FigureSources/Publication_EPS`.
