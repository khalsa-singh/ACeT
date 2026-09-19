# Model and data diagnostics

`prediction_head_comparison` supports Table S7. `pooling_sensitivity` supports Table S9 and supplies fold histories and predictions. `gaussian_copula` supports Table S10 and Figure S10 with real/synthetic samples, marginal statistics, correlation matrices and neighbor diagnostics. `clinical_uncertainty` supports Table S11. `training_curves` preserves extracted fold loss records.

```bash
python acet.py pooling-viscosity --output-dir reruns/pooling_viscosity
python acet.py pooling-mouse-exposure --output-dir reruns/pooling_mouse
python acet.py gaussian-copula --output-dir reruns/copula_diagnostics
python acet.py supplementary-figures --output-dir reruns/supplementary_figures
```

Pooling commands fit new models, copula diagnostics refit synthesizers, and the supplementary figure command renders archived sources with MATLAB. For a dependency-light numerical check of the stored diagnostics, use `python acet.py verify`.
